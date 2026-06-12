#!/usr/bin/env python3
"""
training_router.py
───────────────────
FastAPI router for fine-tuning data management.

Exposes:
  GET  /training/dpo/candidates              DPO rejection pool
  PUT  /training/dpo/candidates/{id}/chosen  Save a curated chosen half
  GET  /training/qlora/candidates            Filtered GOLD exemplars + excluded

MongoDB:
  knowledge_base.chat_history     Read-only — source of truth
  knowledge_base.dpo_candidates   Write — chosen halves + curated status

Include in your main FastAPI app:
  from training_router import router as training_router
  app.include_router(training_router)
"""

from __future__ import annotations

import os
import re
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("training_router")

# ── MongoDB — exact same pattern as intent_agent.py ──────────────────────────
try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

_mongo_client = None

def _get_mongo_client():
    """Shared MongoClient — same env vars as intent_agent.py."""
    global _mongo_client
    if _mongo_client is not None:
        return _mongo_client
    if not MONGO_AVAILABLE:
        return None
    try:
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        _mongo_client.server_info()
        logger.info("training_router: MongoDB connected")
    except Exception as exc:
        logger.warning("training_router: MongoDB unavailable — %s", exc)
        _mongo_client = None
    return _mongo_client


def _get_db():
    client = _get_mongo_client()
    if client is None:
        return None
    db_name = os.getenv("MONGO_DB", "knowledge_base")
    return client[db_name]


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _fmt_dt(dt) -> Optional[str]:
    """
    Serialize to ISO-8601 with Z suffix. Handles three shapes:
      - Python datetime  (from intent_agent.py datetime.utcnow())
      - {"$date": "..."}  (MongoDB extended JSON / PyMongo raw)
      - plain ISO string
    """
    if dt is None:
        return None
    if isinstance(dt, dict):
        dt = dt.get("$date") or dt.get("date")
        if not dt:
            return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
               f"{dt.microsecond // 1000:03d}Z"
    s = str(dt)
    if not s.endswith("Z") and "+" not in s[-6:]:
        s += "Z"
    return s


def _str_id(doc: dict) -> str:
    """MongoDB ObjectId → plain string."""
    return str(doc["_id"])


# ── QLoRA exclusion logic (mirrors export_gold.py) ───────────────────────────

_JIRA_KEY_RE    = re.compile(r"(?<![A-Z])(?<![a-z])(?<!-)\b([A-Z]{2,5}-\d{1,6})\b")
MIN_ANSWER_LEN  = 120


def _qlora_exclusion_reasons(doc: dict) -> List[str]:
    """
    Return exclusion reasons for a GOLD record.
    Empty list = safe to include as a QLoRA training exemplar.
    Mirrors the logic in export_gold.py exactly.
    """
    e       = doc.get("evaluation") or {}
    sources = doc.get("sources") or []
    answer  = doc.get("answer") or ""
    ss      = doc.get("sentries_summary")           # None | [] | [...]
    intent  = e.get("intent") or doc.get("intent") or ""
    method  = e.get("scoring_method") or ""
    ctx_src = e.get("scoring_context_source") or ""

    reasons: List[str] = []

    # 1. sentries/both without live sentries data at eval time
    if intent in ("sentries", "both") and ss is None:
        reasons.append("no_live_sentries")

    # 2. Scored against empty context (blanket faithfulness=1.0)
    #    Pure RAG against stored_snippets is legitimate — exempt it.
    is_empty_ctx       = ("no_context" in method or ctx_src == "empty")
    is_rag_w_snippets  = (
        intent == "rag"
        and ctx_src in ("stored_snippets", "neo4j_tier2_fallback")
    )
    if is_empty_ctx and not is_rag_w_snippets:
        reasons.append("no_real_context")

    # 3. Answer too short to be a useful exemplar
    if len(answer) < MIN_ANSWER_LEN:
        reasons.append(f"short_answer({len(answer)}c)")

    # 4. sentries/both with Jira sources but no keys cited — generic prose
    jira_sources = [s for s in sources if s.get("source") == "jira"]
    cited_keys   = set(_JIRA_KEY_RE.findall(answer.upper()))
    if (intent in ("sentries", "both")
            and jira_sources
            and not cited_keys
            and len(answer) > 80):
        reasons.append("no_keys_cited_despite_jira_sources")

    return reasons


# ── Failure reason extraction for DPO records ─────────────────────────────────

def _failure_reason(validator_details: dict) -> str:
    """Human-readable failure reason from validator_details checks."""
    if not validator_details:
        return ""
    checks  = validator_details.get("checks") or []
    reasons = []
    for ch in checks:
        if (ch.get("penalty") or 0) > 0:
            note = (ch.get("note") or "")[:80]
            reasons.append(f"{ch['name']}: {note}")
    return " | ".join(reasons[:3])


# ── Document serialisers ──────────────────────────────────────────────────────

def _doc_to_dpo(doc: dict, chosen_map: Dict[str, dict]) -> dict:
    """
    chat_history doc → DPO candidate shape.
    Works whether or not an 'evaluation' sub-doc is present.
    chosen_map: { source_id_str: dpo_candidates_doc }
    """
    e    = doc.get("evaluation") or {}
    eid  = _str_id(doc)
    ch   = chosen_map.get(eid) or {}
    return {
        "id":             eid,
        "query":          doc.get("query", ""),
        "rejected":       doc.get("answer", ""),
        "chosen":         ch.get("chosen", ""),
        "curated":        bool(ch.get("curated", False)),
        "intent":         e.get("intent") or doc.get("intent", ""),
        "failure_tags":   e.get("failure_tags") or [],
        "failure_reason": _failure_reason(e.get("validator_details") or {}),
        "weighted_score": e.get("weighted_score"),           # None when no eval
        "faithfulness":   e.get("faithfulness"),             # None when no eval
        "timestamp":      _fmt_dt(doc.get("timestamp")),
        "total_time_s":   doc.get("total_time_s"),
        "sources_count":  len(doc.get("sources") or []),
    }


def _doc_to_qlora_candidate(doc: dict) -> dict:
    e = doc.get("evaluation") or {}
    return {
        "id":             _str_id(doc),
        "query":          doc.get("query", ""),
        "answer":         doc.get("answer", ""),
        "intent":         e.get("intent") or doc.get("intent", ""),
        "weighted_score": e.get("weighted_score"),
        "faithfulness":   e.get("faithfulness"),
        "scoring_method": e.get("scoring_method", ""),
        "scoring_source": e.get("scoring_context_source", ""),
        "graph_coverage": e.get("graph_coverage"),
        "timestamp":      _fmt_dt(doc.get("timestamp")),
    }


def _doc_to_qlora_excluded(doc: dict, reasons: List[str]) -> dict:
    e = doc.get("evaluation") or {}
    return {
        "id":                _str_id(doc),
        "query":             doc.get("query", ""),
        "intent":            e.get("intent") or doc.get("intent", ""),
        "weighted_score":    e.get("weighted_score"),
        "exclusion_reasons": reasons,
    }


# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/training", tags=["training"])


class ChosenPayload(BaseModel):
    chosen: str


@router.get("/dpo/candidates")
def get_dpo_candidates():
    """
    Return DPO candidates from chat_history.

    Priority:
      1. Docs with evaluation.training_signal == "dpo_rejected"  (evaluated by Hammer)
      2. Fallback: all docs with a non-empty answer and no evaluation yet
         — so the panel works even before Hammer has run.
    """
    db = _get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="MongoDB unavailable")

    # Try evaluated records first
    docs = list(
        db["chat_history"]
        .find(
            {"evaluation.training_signal": "dpo_rejected"},
            {
                "query": 1, "answer": 1, "intent": 1,
                "evaluation": 1, "timestamp": 1,
            },
        )
        .sort("timestamp", -1)
    )

    # Fallback: raw turns that haven't been evaluated yet
    if not docs:
        docs = list(
            db["chat_history"]
            .find(
                {
                    "answer": {"$exists": True, "$ne": ""},
                    "evaluation": {"$exists": False},
                },
                {
                    "query": 1, "answer": 1, "intent": 1,
                    "timestamp": 1, "sources": 1,
                    "sentries_summary": 1, "total_time_s": 1,
                },
            )
            .sort("timestamp", -1)
            .limit(200)
        )

    if not docs:
        return JSONResponse({"candidates": [], "total": 0})

    # Batch-fetch all chosen halves in one query — avoids N+1
    ids = [_str_id(d) for d in docs]
    chosen_map: Dict[str, dict] = {}
    for c in db["dpo_candidates"].find({"source_id": {"$in": ids}}):
        chosen_map[c["source_id"]] = c

    candidates = [_doc_to_dpo(d, chosen_map) for d in docs]
    return JSONResponse({"candidates": candidates, "total": len(candidates)})


@router.put("/dpo/candidates/{doc_id}/chosen")
def update_chosen(doc_id: str, payload: ChosenPayload):
    """
    Save or update the curated chosen half for a DPO candidate.
    Upserts into knowledge_base.dpo_candidates keyed by source_id.
    """
    db = _get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="MongoDB unavailable")

    chosen = payload.chosen.strip()
    if len(chosen) < 50:
        raise HTTPException(
            status_code=422,
            detail="Chosen answer must be at least 50 characters",
        )

    # Validate that the source document actually exists
    try:
        oid = ObjectId(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    if not db["chat_history"].find_one({"_id": oid}, {"_id": 1}):
        raise HTTPException(
            status_code=404,
            detail=f"Source document {doc_id} not found in chat_history",
        )

    db["dpo_candidates"].update_one(
        {"source_id": doc_id},
        {
            "$set": {
                "source_id":  doc_id,
                "chosen":     chosen,
                "curated":    True,
                "curated_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )

    return {"ok": True, "source_id": doc_id, "chosen_len": len(chosen)}


@router.get("/qlora/candidates")
def get_qlora_candidates():
    """
    Return GOLD records split into two lists:
      candidates — pass all export_gold.py criteria, safe to train on
      excluded   — GOLD grade but excluded for a specific reason
    """
    db = _get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="MongoDB unavailable")

    # Load manually excluded ids first
    manually_excluded_ids = {
        str(d["source_id"])
        for d in db["qlora_excluded"].find({}, {"source_id": 1})
    } if "qlora_excluded" in db.list_collection_names() else set()

    docs = list(
        db["chat_history"]
        .find(
            {"evaluation.quality_grade": "GOLD"},
            {
                "query": 1, "answer": 1, "intent": 1,
                "evaluation": 1, "sources": 1,
                "sentries_summary": 1, "timestamp": 1,
            },
        )
        .sort("evaluation.weighted_score", -1)
    )

    candidates: List[dict] = []
    excluded:   List[dict] = []

    for doc in docs:
        doc_id = _str_id(doc)

        # Manual exclusions take priority
        if doc_id in manually_excluded_ids:
            excluded.append({
                **_doc_to_qlora_excluded(doc, ["manually_excluded"]),
                "id": doc_id,
            })
            continue

        reasons = _qlora_exclusion_reasons(doc)
        if not reasons:
            candidates.append(_doc_to_qlora_candidate(doc))
        else:
            excluded.append(_doc_to_qlora_excluded(doc, reasons))

    return JSONResponse({
        "candidates": candidates,
        "excluded":   excluded,
        "total_gold": len(candidates) + len(excluded),
    })


@router.delete("/qlora/candidates/{doc_id}")
def exclude_qlora_candidate(doc_id: str):
    """
    Manually exclude a QLoRA candidate. Persisted in qlora_excluded collection
    so it survives server restarts and won't reappear on refresh.
    """
    db = _get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="MongoDB unavailable")

    try:
        oid = ObjectId(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    if not db["chat_history"].find_one({"_id": oid}, {"_id": 1}):
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    db["qlora_excluded"].update_one(
        {"source_id": doc_id},
        {"$set": {
            "source_id":   doc_id,
            "reason":      "manually_excluded",
            "excluded_at": datetime.now(timezone.utc),
        }},
        upsert=True,
    )
    return {"ok": True, "source_id": doc_id, "action": "excluded"}


@router.post("/qlora/candidates/{doc_id}/restore")
def restore_qlora_candidate(doc_id: str):
    """
    Restore a manually excluded QLoRA candidate back to the candidates list.
    """
    db = _get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="MongoDB unavailable")

    result = db["qlora_excluded"].delete_one({"source_id": doc_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"No manual exclusion found for {doc_id}")

    return {"ok": True, "source_id": doc_id, "action": "restored"}

#!/usr/bin/env python3
"""
Hammer Dataset Builder  —  Neo4j edition
════════════════════════════════════════
Queries scored chat_history documents and exports training datasets for:

  QLoRA  — (prompt, response) pairs   from responses with score >= 0.80
  DPO    — (prompt, chosen, rejected) from responses with score <  0.50
  GRPO   — sentry call logs           from all sentries/both intent docs

Prompts are reconstructed to match the exact Weaver templates in
agent/weaver_node.py so the fine-tuned model learns the right format.

Context retrieval — two-tier (matches evaluator.py exactly):
  Tier 1 — Stored context_snippets from MongoDB (the actual passages the
           LLM saw at inference time, written by intent_agent.py).  Used
           whenever the doc has snippets — no Neo4j call.
  Tier 2 — Neo4j hybrid_search() re-retrieval (fallback) when no snippets
           are stored or when the caller has no doc handle.

For sentries intent: context summary is reconstructed from stored sources.

Output files (all JSONL, one JSON object per line):
  datasets/qlora_train.jsonl    — QLoRA supervised fine-tuning
  datasets/dpo_candidates.jsonl — DPO rejected halves (chosen = "")
  datasets/grpo_rollout.jsonl   — GRPO reward signal data

Provenance fields are included in every record so datasets are fully
traceable back to the originating chat_history document.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
MONGO_URI         = os.getenv("MONGO_URI",         "mongodb://localhost:27017/")
MONGO_DB          = os.getenv("MONGO_DB",          "knowledge_base")

# Neo4j configuration
NEO4J_URI         = os.getenv("NEO4J_URI",         "bolt://localhost:7687")
NEO4J_USER        = os.getenv("NEO4J_USER",        "neo4j")
NEO4J_PASSWORD    = os.getenv("NEO4J_PASSWORD",    "your_password")
NEO4J_DATABASE    = os.getenv("NEO4J_DATABASE",    "neo4j")

# Graph retrieval parameters — mirror neo4j_search.py / evaluator.py defaults
GRAPH_HOPS        = int(os.getenv("HAMMER_GRAPH_HOPS",       "2"))
MAX_GRAPH_EXPAND  = int(os.getenv("HAMMER_GRAPH_MAX_EXPAND", "20"))

EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL",   "Snowflake/snowflake-arctic-embed-m")
ARCTIC_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

THRESHOLD_QLORA   = float(os.getenv("THRESHOLD_QLORA", "0.80"))
THRESHOLD_DPO     = float(os.getenv("THRESHOLD_DPO",   "0.50"))
OUTPUT_DIR        = Path(os.getenv("HAMMER_DATASET_DIR", "datasets"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hammer.dataset_builder")

# ─── Optional dependencies ────────────────────────────────────────────────────
try:
    from rag.neo4j_search import Neo4jSearch as _Neo4jSearch
    NEO4J_SEARCH_AVAILABLE = True
except ImportError:
    try:
        from neo4j_search import Neo4jSearch as _Neo4jSearch   # flat layout fallback
        NEO4J_SEARCH_AVAILABLE = True
    except ImportError:
        NEO4J_SEARCH_AVAILABLE = False
        logger.warning(
            "neo4j_search not importable — Tier 2 context re-retrieval disabled. "
            "Stored snippets (Tier 1) will still work for scored documents."
        )

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# ─── Lazy singletons ──────────────────────────────────────────────────────────
_mongo_client       = None
_neo4j_search_inst  = None   # Neo4jSearch singleton
_embed_model        = None


def _get_db():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI)
    return _mongo_client[MONGO_DB]


def _get_neo4j_search():
    """
    Return a shared Neo4jSearch instance, or None if unavailable.

    Neo4jSearch loads the embedding model on first instantiation, so we keep
    a singleton and only create it once per dataset_builder process.
    Mirrors evaluator._get_neo4j_search() exactly so both modules retrieve
    context the same way.
    """
    global _neo4j_search_inst
    if not NEO4J_SEARCH_AVAILABLE:
        return None
    if _neo4j_search_inst is None:
        try:
            _neo4j_search_inst = _Neo4jSearch()
            logger.info("Neo4j search instance ready for Tier 2 context re-retrieval")
        except Exception as e:
            logger.warning("Neo4jSearch init failed: %s", e)
    return _neo4j_search_inst


def _get_embed_model():
    global _embed_model
    if not EMBEDDINGS_AVAILABLE:
        return None
    if _embed_model is None:
        try:
            _embed_model = SentenceTransformer(EMBEDDING_MODEL)
            _embed_model.max_seq_length = 512
        except Exception:
            pass
    return _embed_model


# ═══════════════════════════════════════════════════════════════════════════════
# Context reconstruction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _retrieve_context(
    query: str,
    top_k: int = 8,
    doc: Optional[Dict] = None,
) -> List[str]:
    """
    Two-tier context retrieval — mirrors evaluator.retrieve_context() so that
    training data context matches the context the evaluator used to score it.

    Tier 1 — Stored context_snippets (fast, no Neo4j call):
        If the chat_history doc has context_snippets (written by intent_agent.py
        at inference time), use them directly.  These are the actual passages
        the LLM saw when generating the answer, making them the ground truth
        for both scoring and training.

    Tier 2 — Neo4j hybrid_search() re-retrieval (fallback):
        If no snippets are stored or no doc is provided, re-query Neo4j using
        the original query.  Hybrid search (vector + multi-hop graph) matches
        the production retrieval path.

    Args:
        query:  Original user query.
        top_k:  Passages to retrieve (Tier 2 only).
        doc:    Full chat_history document (used for Tier 1).

    Returns:
        List of passage text strings.  Always a list (possibly empty), never None.
    """
    # ── Tier 1: stored snippets ──────────────────────────────────────────────
    if doc is not None:
        snippets = doc.get("context_snippets") or []
        if snippets:
            texts = [s.get("text", "") if isinstance(s, dict) else str(s) for s in snippets]
            texts = [t for t in texts if t and len(t) > 15]
            if texts:
                return texts

    # ── Tier 2: Neo4j hybrid_search() re-retrieval ───────────────────────────
    searcher = _get_neo4j_search()
    if searcher is None:
        logger.debug("_retrieve_context: Neo4j unavailable — returning empty list")
        return []
    try:
        hits = searcher.hybrid_search(
            query=query,
            top_k=top_k,
            hops=GRAPH_HOPS,
            max_expand=MAX_GRAPH_EXPAND,
        )
        texts = [h.get("text", "") for h in hits if h.get("text") and len(h.get("text", "")) > 15]
        logger.debug(
            "_retrieve_context: Tier 2 — %d Neo4j hits → %d texts (hops=%d)",
            len(hits), len(texts), GRAPH_HOPS,
        )
        return texts
    except Exception as e:
        logger.warning("Neo4j context re-retrieval failed: %s", e)
        return []


def _build_rag_context(passages: List[str]) -> str:
    """Format retrieved passages into the Weaver rag_context block."""
    if not passages:
        return "[No context retrieved]"
    parts = []
    for i, p in enumerate(passages, 1):
        parts.append(f"[{i}] {p[:600]}")
    return "\n\n".join(parts)


def _build_sentries_context(sources: List[Dict]) -> str:
    """
    Reconstruct a sentries_context block from stored source metadata.

    Note: this is a best-effort reconstruction — the raw sentry payloads
    are not stored in chat_history. The reconstructed context gives the
    fine-tuned model enough format signal even without full payloads.
    """
    if not sources:
        return "[No live API data retrieved]"
    parts = []
    for s in sources:
        src   = s.get("source",  "unknown").upper()
        stype = s.get("source_type", s.get("source", ""))
        title = s.get("title",   "No title")
        url   = s.get("url",     "N/A")
        proj  = s.get("project", "")
        score = s.get("score",   0.0)
        parts.append(
            f"[{src}] {stype}\n"
            f"  Title   : {title}\n"
            f"  URL     : {url}\n"
            f"  Project : {proj}\n"
            f"  Score   : {score:.3f}"
        )
    return "\n\n".join(parts)


def _build_history_block(history: List[Dict]) -> str:
    """History block — delegates to the live weaver's formatter so training
    prompts match inference exactly (FLATTENED FIX: label/format had drifted)."""
    if not history:
        return ""
    return _weaver_format_history(history[-3:])   # last 3 turns = HISTORY_MAX_TURNS


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt reconstruction  (mirrors agent/weaver_node.py exactly)
# ═══════════════════════════════════════════════════════════════════════════════

# FLATTENED FIX: the local template copies had DRIFTED from the live weaver
# (the old _BOTH_PROMPT said "combining the two data sources" while production
# says "LIVE API DATA is the primary source of truth"), so every exported
# training record taught a prompt format inference never uses. Templates are
# now imported from the weaver module itself — drift is structurally impossible.
import sys as _sys
_HERE_DB = os.path.dirname(os.path.abspath(__file__))
_ROOT_DB = os.path.dirname(_HERE_DB)
for _p in (_ROOT_DB, os.path.join(_ROOT_DB, "agent")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from weaver_node import (
    _RAG_ONLY_PROMPT,
    _SENTRIES_ONLY_PROMPT,
    _BOTH_PROMPT,
    _format_history as _weaver_format_history,
)


def reconstruct_prompt(
    doc:     Dict[str, Any],
    history: Optional[List[Dict]] = None,
) -> str:
    """
    Reconstruct the exact Weaver prompt for a chat_history document.

    For rag/both intents: context comes from stored snippets (Tier 1) or
    Neo4j hybrid_search() fallback (Tier 2) — see _retrieve_context.
    For sentries intent: uses stored source metadata.
    """
    query   = doc.get("query",   "")
    intent  = doc.get("intent",  "both")
    sources = doc.get("sources", [])
    history = history or []

    history_block = _build_history_block(history)

    if intent == "rag":
        passages = _retrieve_context(query, doc=doc)
        rag_context = _build_rag_context(passages)
        return _RAG_ONLY_PROMPT.format(
            history_block=history_block,
            rag_context=rag_context,
            query=query,
        )

    elif intent == "sentries":
        sentries_context = _build_sentries_context(sources)
        return _SENTRIES_ONLY_PROMPT.format(
            history_block=history_block,
            sentries_context=sentries_context,
            query=query,
        )

    else:   # "both" or unknown
        # Fix: was previously _retrieve_context(query) — no doc passed, so Tier 1
        # was skipped and "both" intent training prompts had empty rag_context.
        # Pass the doc so stored snippets are used (matches evaluator behaviour).
        passages         = _retrieve_context(query, doc=doc)
        rag_context      = _build_rag_context(passages)
        sentries_context = _build_sentries_context(sources)
        return _BOTH_PROMPT.format(
            history_block=history_block,
            rag_context=rag_context,
            sentries_context=sentries_context,
            query=query,
        )


def _doc_hash(doc_id: Any) -> str:
    """Stable 12-char hex hash for document ID."""
    return hashlib.md5(str(doc_id).encode()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════════════
# Export functions
# ═══════════════════════════════════════════════════════════════════════════════

def export_qlora(
    output_path: Optional[Path] = None,
    min_score:   float = THRESHOLD_QLORA,
    limit:       Optional[int] = None,
) -> int:
    """
    Export QLoRA supervised fine-tuning dataset.

    Query: scored docs with weighted_score >= min_score AND no error.
    Format per line:
      {
        "prompt":    "<full weaver prompt>",
        "response":  "<answer>",
        "metadata":  { doc_hash, score, intent, timestamp, scoring_method }
      }

    Returns: number of records exported.
    """
    output_path = output_path or (OUTPUT_DIR / "qlora_train.jsonl")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    db     = _get_db()
    # 2D classification: GOLD docs only (no failure tags + high score + high faithfulness)
    # Backward-compat fallback: docs scored before the migration that have
    # training_signal=qlora_positive but no quality_grade are still eligible.
    cursor = db["chat_history"].find(
        {
            "$and": [
                {"evaluation.weighted_score": {"$gte": min_score}},
                {"error": None},
                {"answer": {"$exists": True}},
                {"$or": [
                    {"evaluation.quality_grade": "GOLD"},
                    # Backward-compat for legacy docs not yet re-scored
                    {"$and": [
                        {"evaluation.quality_grade": {"$exists": False}},
                        {"evaluation.training_signal": "qlora_positive"},
                    ]},
                ]},
            ],
        },
        sort=[("evaluation.weighted_score", -1)],
    )
    if limit:
        cursor = cursor.limit(limit)

    count = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for doc in cursor:
            try:
                prompt = reconstruct_prompt(doc)
                eval_d = doc.get("evaluation", {})
                record = {
                    "prompt":   prompt,
                    "response": doc["answer"],
                    "metadata": {
                        "doc_hash":         _doc_hash(doc["_id"]),
                        "score":            eval_d.get("weighted_score"),
                        "intent":           doc.get("intent", "unknown"),
                        "timestamp":        doc.get("timestamp", "").isoformat() if hasattr(doc.get("timestamp"), "isoformat") else str(doc.get("timestamp", "")),
                        "scoring_method":   eval_d.get("scoring_method", ""),
                        "faithfulness":     eval_d.get("faithfulness"),
                        "answer_relevance": eval_d.get("answer_relevance"),
                        # New 2D fields
                        "quality_grade":    eval_d.get("quality_grade"),
                        "failure_tags":     eval_d.get("failure_tags", []),
                    },
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                logger.warning("Skipping doc %s: %s", doc.get("_id"), e)

    logger.info("QLoRA export: %d records → %s", count, output_path)
    return count


def export_dpo(
    output_path:    Optional[Path] = None,
    max_score:      float = THRESHOLD_DPO,
    limit:          Optional[int] = None,
    mark_in_mongo:  bool = True,
) -> int:
    """
    Export DPO rejected-half dataset.

    Query: scored docs with weighted_score < max_score.
    Format per line:
      {
        "prompt":   "<full weaver prompt>",
        "chosen":   "",             ← MUST be filled manually
        "rejected": "<bad answer>",
        "metadata": { doc_hash, score, intent, training_signal, failure_reason }
      }

    Also writes pending records to MongoDB dpo_candidates collection so
    they appear in the review queue.

    Returns: number of records exported.
    """
    output_path = output_path or (OUTPUT_DIR / "dpo_candidates.jsonl")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    db         = _get_db()
    collection = db["chat_history"]
    dpo_coll   = db["dpo_candidates"]

    # 2D classification: FAILED docs with training-relevant critical tags only.
    # We deliberately EXCLUDE retrieval_miss — those are retriever-side problems,
    # not model behaviour problems, so they have no value as DPO rejections.
    # Hallucination and tool_misuse are model-behaviour issues that DPO can
    # actually correct (model learns to prefer chosen over rejected).
    cursor = collection.find(
        {
            "$and": [
                {"answer": {"$exists": True}},
                {"$or": [
                    {"$and": [
                        {"evaluation.quality_grade": "FAILED"},
                        {"evaluation.failure_tags": {
                            "$in": ["hallucination", "tool_misuse"]
                        }},
                    ]},
                    # Backward-compat for legacy docs
                    {"$and": [
                        {"evaluation.quality_grade": {"$exists": False}},
                        {"evaluation.training_signal": "dpo_rejected"},
                        {"evaluation.weighted_score": {"$lt": max_score}},
                    ]},
                ]},
            ],
        },
        sort=[("evaluation.weighted_score", 1)],   # worst first
    )
    if limit:
        cursor = cursor.limit(limit)

    count = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for doc in cursor:
            try:
                prompt = reconstruct_prompt(doc)
                eval_d = doc.get("evaluation", {})

                # Use new tag-based failure reason when available, else legacy
                tags = eval_d.get("failure_tags", [])
                failure_reason = ", ".join(tags) if tags else _infer_failure_reason(doc)

                record = {
                    "prompt":   prompt,
                    "chosen":   "",   # ← human must fill this in
                    "rejected": doc["answer"],
                    "metadata": {
                        "doc_hash":          _doc_hash(doc["_id"]),
                        "score":             eval_d.get("weighted_score"),
                        "intent":            doc.get("intent", "unknown"),
                        "faithfulness":      eval_d.get("faithfulness"),
                        "answer_relevance":  eval_d.get("answer_relevance"),
                        "temporal":          eval_d.get("temporal_consistency"),
                        # New 2D fields
                        "quality_grade":     eval_d.get("quality_grade"),
                        "failure_tags":      tags,
                        # Legacy / debugging
                        "failure_reason":    failure_reason,
                        "validator_details": eval_d.get("validator_details"),
                        "timestamp":         str(doc.get("timestamp", "")),
                        "query":             doc.get("query", ""),
                    },
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                if mark_in_mongo:
                    dpo_coll.update_one(
                        {"doc_hash": _doc_hash(doc["_id"])},
                        {"$setOnInsert": {
                            "doc_hash":       _doc_hash(doc["_id"]),
                            "query":          doc.get("query", ""),
                            "rejected":       doc["answer"][:500],
                            "prompt_preview": prompt[:200],
                            "score":          eval_d.get("weighted_score"),
                            "failure_tags":   tags,
                            "failure_reason": failure_reason,
                            "chosen":         None,
                            "reviewed":       False,
                            "created_at":     datetime.utcnow(),
                        }},
                        upsert=True,
                    )

                count += 1
            except Exception as e:
                logger.warning("Skipping doc %s: %s", doc.get("_id"), e)

    logger.info("DPO export: %d records → %s", count, output_path)
    if mark_in_mongo:
        logger.info("DPO candidates written to MongoDB dpo_candidates collection")
    return count


def export_grpo(
    output_path: Optional[Path] = None,
    limit:       Optional[int] = None,
) -> int:
    """
    Export GRPO rollout data (sentry call logs).

    Query: all docs where intent in (sentries, both) — no score threshold.
    Per the report: "Execution reward computed at training time — no score threshold."

    Format per line:
      {
        "query":           "<user query>",
        "intent":          "sentries" | "both",
        "answer":          "<generated answer>",
        "sources_summary": [{ source, title, score, url }],
        "timing":          { ... },
        "reward_hint":     float 0.0–1.0   (from validator score if available)
        "metadata":        { doc_hash, timestamp, total_time_s }
      }

    Returns: number of records exported.
    """
    output_path = output_path or (OUTPUT_DIR / "grpo_rollout.jsonl")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    db     = _get_db()
    cursor = db["chat_history"].find(
        {"intent": {"$in": ["sentries", "both"]}, "answer": {"$exists": True}},
        sort=[("timestamp", -1)],
    )
    if limit:
        cursor = cursor.limit(limit)

    count = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for doc in cursor:
            try:
                eval_d     = doc.get("evaluation", {})
                sources    = doc.get("sources", [])

                # Reward hint from validator score (if available) or fallback
                if eval_d.get("scoring_method") == "validator":
                    reward_hint = eval_d.get("faithfulness", 0.5)
                elif eval_d.get("weighted_score") is not None:
                    reward_hint = eval_d["weighted_score"]
                else:
                    reward_hint = None   # computed at training time

                record = {
                    "query":   doc.get("query", ""),
                    "intent":  doc.get("intent", "sentries"),
                    "answer":  doc.get("answer", ""),
                    "sources_summary": [
                        {
                            "source":  s.get("source",  ""),
                            "title":   s.get("title",   "")[:100],
                            "score":   s.get("score",   0.0),
                            "url":     s.get("url",     ""),
                            "project": s.get("project", ""),
                        }
                        for s in sources[:10]
                    ],
                    "timing":       doc.get("timing", {}),
                    "reward_hint":  reward_hint,
                    "metadata": {
                        "doc_hash":    _doc_hash(doc["_id"]),
                        "timestamp":   str(doc.get("timestamp", "")),
                        "total_time_s":doc.get("total_time_s"),
                    },
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                logger.warning("Skipping doc %s: %s", doc.get("_id"), e)

    logger.info("GRPO export: %d records → %s", count, output_path)
    return count


# ═══════════════════════════════════════════════════════════════════════════════
# Seed dataset extractor (step 3 from the doc — no scoring needed)
# ═══════════════════════════════════════════════════════════════════════════════

def export_seed_dataset(
    output_path: Optional[Path] = None,
    limit: int = 30,
) -> int:
    """
    Extract the best demo/production responses to use as QLoRA seed data.

    Uses the MongoDB query from the evaluation report:
      total_time_s < 10  AND  answer exists  AND  no error
      sorted by timestamp descending, limited to `limit`

    These are exported even before the evaluator has run, so they
    have no evaluation.weighted_score yet.

    Returns: number of records exported.
    """
    output_path = output_path or (OUTPUT_DIR / "seed_train.jsonl")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    db     = _get_db()
    cursor = db["chat_history"].find(
        {
            "total_time_s": {"$lt": 10},
            "answer":       {"$exists": True},
            "error":        None,
        },
        sort=[("timestamp", -1)],
    ).limit(limit)

    count = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for doc in cursor:
            try:
                prompt = reconstruct_prompt(doc)
                record = {
                    "prompt":   prompt,
                    "response": doc["answer"],
                    "metadata": {
                        "doc_hash":    _doc_hash(doc["_id"]),
                        "total_time_s":doc.get("total_time_s"),
                        "intent":      doc.get("intent", "unknown"),
                        "timestamp":   str(doc.get("timestamp", "")),
                        "source":      "seed_export",
                    },
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                logger.warning("Skipping doc %s: %s", doc.get("_id"), e)

    logger.info("Seed export: %d records → %s", count, output_path)
    return count


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _infer_failure_reason(doc: Dict) -> str:
    """
    Heuristic: infer why a response scored low, to help human reviewers
    prioritise which DPO chosen responses to write first.
    """
    eval_d = doc.get("evaluation", {})
    answer = doc.get("answer", "")

    # API refusal — highest priority for DPO (the report calls this out)
    if re.search(r"i\s+don.t\s+have\s+access|i\s+can.t\s+(access|call|query)", answer, re.I):
        return "api_refusal"

    # Low faithfulness with RAG context
    if eval_d.get("faithfulness", 1) < 0.40:
        return "low_faithfulness"

    # Low temporal consistency
    if eval_d.get("temporal_consistency", 1) < 0.50:
        return "temporal_inconsistency"

    # Low answer relevance
    if eval_d.get("answer_relevance", 1) < 0.40:
        return "off_topic"

    # Syntax error in code
    if eval_d.get("code_correctness") is not None and eval_d["code_correctness"] < 0.50:
        return "code_syntax_error"

    return "low_overall_score"


def dataset_stats() -> Dict[str, Any]:
    """Print current dataset readiness stats from MongoDB.

    Reports both the new 2D classification (quality_grade + failure_tags) and
    the legacy training_signal counts for backward compatibility.
    """
    db = _get_db()
    collection = db["chat_history"]

    total       = collection.count_documents({})
    scored      = collection.count_documents({"evaluation": {"$exists": True}})

    # New 2D classification counts
    grades = {}
    for g in ("GOLD", "SILVER", "BRONZE", "FAILED"):
        grades[g] = collection.count_documents({"evaluation.quality_grade": g})
    graded_total = sum(grades.values())

    # Failure tag distribution via aggregation
    tag_counts: Dict[str, int] = {}
    try:
        pipe = [
            {"$match": {"evaluation.failure_tags": {"$exists": True}}},
            {"$unwind": "$evaluation.failure_tags"},
            {"$group": {"_id": "$evaluation.failure_tags", "n": {"$sum": 1}}},
        ]
        for row in collection.aggregate(pipe):
            tag_counts[row["_id"]] = row["n"]
    except Exception as e:
        logger.debug("tag aggregation failed: %s", e)

    # Legacy training_signal counts (still useful for legacy docs not re-scored)
    qlora_ready = collection.count_documents({"evaluation.training_signal": "qlora_positive"})
    dpo_ready   = collection.count_documents({"evaluation.training_signal": "dpo_rejected"})
    grpo_ready  = collection.count_documents({"intent": {"$in": ["sentries", "both"]}})
    dpo_pending = db["dpo_candidates"].count_documents({"reviewed": False, "chosen": None})

    return {
        "total_responses":    total,
        "scored":             scored,
        "unscored":           total - scored,
        # New 2D classification
        "grades":             grades,
        "graded_total":       graded_total,
        "failure_tags":       tag_counts,
        # Legacy backward-compat
        "qlora_ready":        qlora_ready,
        "dpo_ready":          dpo_ready,
        "grpo_ready":         grpo_ready,
        "dpo_pending_review": dpo_pending,
        "qlora_threshold":    THRESHOLD_QLORA,
        "dpo_threshold":      THRESHOLD_DPO,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hammer Dataset Builder")
    sub = parser.add_subparsers(dest="cmd")

    p_qlora = sub.add_parser("qlora",  help="Export QLoRA training set")
    p_qlora.add_argument("--limit", type=int, default=None)
    p_qlora.add_argument("--out",   type=str, default=None)

    p_dpo   = sub.add_parser("dpo",   help="Export DPO candidate set")
    p_dpo.add_argument("--limit", type=int, default=None)
    p_dpo.add_argument("--out",   type=str, default=None)

    p_grpo  = sub.add_parser("grpo",  help="Export GRPO rollout set")
    p_grpo.add_argument("--limit", type=int, default=None)
    p_grpo.add_argument("--out",   type=str, default=None)

    p_seed  = sub.add_parser("seed",  help="Export seed dataset (pre-scoring)")
    p_seed.add_argument("--limit", type=int, default=30)
    p_seed.add_argument("--out",   type=str, default=None)

    p_stats = sub.add_parser("stats", help="Print dataset readiness stats")

    args = parser.parse_args()

    if args.cmd == "qlora":
        export_qlora(Path(args.out) if args.out else None, limit=args.limit)
    elif args.cmd == "dpo":
        export_dpo(Path(args.out) if args.out else None, limit=args.limit)
    elif args.cmd == "grpo":
        export_grpo(Path(args.out) if args.out else None, limit=args.limit)
    elif args.cmd == "seed":
        export_seed_dataset(Path(args.out) if args.out else None, limit=args.limit)
    elif args.cmd == "stats":
        import pprint
        pprint.pprint(dataset_stats())
    else:
        parser.print_help()

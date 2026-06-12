#!/usr/bin/env python3
"""
enterprise_benchmark.py
═══════════════════════
Production-grade static benchmark for the AI assistant.

Reads from MongoDB chat_history (or a JSON export).
Scores every record across 8 dimensions using deterministic heuristics.
Computes F1 / Precision / Recall per dimension.
Writes every FAILING record to its category MongoDB collection,
ready for Hammer follow-up scoring and training dataset export.

Pipeline position:
    ┌────────────────┐
    │  chat_history  │  (MongoDB — production queries)
    └───────┬────────┘
            │  this script
            ▼
    ┌────────────────────────────────────────────────────────────┐
    │  STATIC BENCHMARK  (offline, deterministic, no LLM)        │
    │  • F1 / Precision / Recall per dimension                   │
    │  • Triggered categories → failing records exported         │
    │  • Failing records written to MongoDB by category          │
    └───────┬────────────────────────────────────────────────────┘
            │  failing records
            ▼
    ┌────────────────────────────────────────────────────────────┐
    │  HAMMER  (faithfulness, answer_relevance, code_score)      │
    │  • RAGAS full scoring on flagged records                   │
    │  • Continuous bias monitoring                              │
    │  • Dataset builder → qlora_train / dpo_candidates          │
    └───────────────────────────────────────────────────────────-┘

MongoDB collections written by this script:
    benchmark_failures_refusal          Cat A  → DPO
    benchmark_failures_format           Cat D  → QLoRA
    benchmark_failures_fabrication      Cat B  → DPO
    benchmark_failures_honest_empty     Cat F  → DPO
    benchmark_failures_french           Cat E  → QLoRA
    benchmark_failures_synthesis        Cat I  → QLoRA  (triggered)
    benchmark_failures_complex_params   Cat H  → GRPO
    benchmark_failures_length_quality   —      → QLoRA
    benchmark_runs                             → audit log of every run

Usage:
    # From MongoDB (recommended for production):
    python enterprise_benchmark.py

    # From JSON export:
    python enterprise_benchmark.py --json chat_history.json

    # Last N records only:
    python enterprise_benchmark.py --last 300

    # Dry run — score and report but do not write to MongoDB:
    python enterprise_benchmark.py --dry-run

    # After the run, feed triggered categories to Hammer:
    python -m hammer.run_hammer score --mode fast --collection benchmark_failures_synthesis
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# ── Optional MongoDB ──────────────────────────────────────────────────────────
try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB  = os.getenv("MONGO_DB",  "knowledge_base")

# ── Thresholds (from BENCHMARK_EVALUATION_FRAMEWORK_v2) ──────────────────────
THRESHOLDS: Dict[str, Tuple[float, str]] = {
    "refusal":        (0.85, "DPO"),
    "format":         (0.80, "QLoRA"),
    "fabrication":    (0.80, "DPO"),
    "honest_empty":   (0.80, "DPO"),
    "french":         (0.85, "QLoRA"),
    "synthesis":      (0.75, "QLoRA"),
    "complex_params": (0.70, "GRPO"),
    "length_quality": (0.90, "QLoRA"),
}

DIMS = list(THRESHOLDS.keys())

# Per-intent scoring weights for weighted_score
SCORE_WEIGHTS = {
    "rag":      {"format": 0.30, "fabrication": 0.25, "length_quality": 0.20, "french": 0.25},
    "sentries": {"refusal": 0.35, "format": 0.30, "honest_empty": 0.15, "length_quality": 0.20},
    "both":     {"synthesis": 0.40, "format": 0.25, "refusal": 0.15, "length_quality": 0.20},
}

# MongoDB failure collection name per dimension
FAILURE_COLLECTIONS: Dict[str, str] = {
    "refusal":        "benchmark_failures_refusal",
    "format":         "benchmark_failures_format",
    "fabrication":    "benchmark_failures_fabrication",
    "honest_empty":   "benchmark_failures_honest_empty",
    "french":         "benchmark_failures_french",
    "synthesis":      "benchmark_failures_synthesis",
    "complex_params": "benchmark_failures_complex_params",
    "length_quality": "benchmark_failures_length_quality",
}

# ── Pattern libraries ─────────────────────────────────────────────────────────
_STRUCTURE_SIGNALS = [
    "**title**", "issue id", "state:", "gitlab.com", "atlassian.net",
    "atlassian", "**state**", "**url**", "**milestone**", "- **",
    "url:", "| issue", "|---", "**author**", "iid:", "merge request",
    "pipeline", "**priority**", "**assignee**", "**reporter**",
]
_REFUSAL_PHRASES = [
    "i don't have access", "i cannot access", "i can't access",
    "as an ai", "i don't have real-time", "unable to retrieve",
    "i cannot retrieve", "i can't retrieve", "mismatch between the context",
    "i cannot query", "i don't have the ability", "i am not able to",
    "je ne peux pas accéder", "je n'ai pas accès",
]
_INFRA_PHRASES = [
    "aucune donnée", "données en temps réel", "check credentials",
    "sentry dispatcher not available", "rag system not available",
    "unauthorized error", "api token has expired", "connection refused",
    "no api data was retrieved", "veuillez vérifier",
    "live api data returned no", "no live api data", "api request returned an error",
    "no live api results", "not possible to determine", "could not be retrieved"
]
_EMPTY_PHRASES = [
    "no results", "no open issues", "no issues found", "not found",
    "no merge requests", "no pipelines", "no tickets", "no data found",
    "does not exist", "no branches", "no commits", "no open tickets",
    "0 tickets", "0 issues", "no open bugs", "could not find",
    "no open mr", "no failed pipelines", "nothing found", "no results found",
    "no jira tickets", "there are no", "aucun", "aucune issue",
    "aucun ticket", "aucun résultat",
]
_HEDGING_RE = re.compile(
    r"\b(typically|usually|generally|in most cases|would be|should be|"
    r"might be|often|you would typically|you would need to|"
    r"in standard deployments|commonly used|would typically|"
    r"généralement|habituellement|en général)\b",
    re.I,
)
_FRENCH_Q_RE = re.compile(
    r"[àâäéèêëîïôùûüçœ]|résume|montre|explique|liste|comment fonctionne|"
    r"c'est quoi|qu'est|quelle|quels|quelles|donne.?moi|cherche|affiche|"
    r"montre.?moi|qu'il y a|est.?ce que",
    re.I,
)
_FRENCH_ANS_RE = re.compile(
    r"[àâäéèêëîïôùûüç]{2,}|voici|les \w+|des \w+|sont |dans |pour |avec |"
    r"aucune|notre|entre|selon|suivant|également|ainsi|alors|"
    r"le fichier|les tickets|les problèmes|voici les",
    re.I,
)
_LIVE_SIGNALS = [
    "issue id", "gitlab.com", "atlassian", "state:", "milestone",
    "opened", "closed", "auth-", "ecom-", "paym-", "**title**",
    "iid:", "merge request", "pipeline", "jira.net", "sprint",
    "billing-", "infra-", "iam-", "chat-", "portal-",
]
_DOC_SIGNALS = [
    "according to", "documentation", "architecture", "confluence",
    "runbook", "based on our", "policy says", "standard requires",
    "knowledge base", "the guide", "the spec", "per our",
    "selon notre", "d'après notre", "notre documentation",
]
_JIRA_KEY_RE   = re.compile(r"\b[A-Z]{2,6}-\d+\b")
_GITLAB_URL_RE = re.compile(
    r"gitlab\.com/[\w\-]+/[\w\-]+/-/(?:issues|merge_requests)/\d+"
)
_COMPLEX_Q_RE  = re.compile(
    r"(both.*label|label.*and|priority.*and|status.*and|"
    r"across.*project.*label|filter.*priority|label.*status|"
    r"and.*open.*label|critical.*and|open.*bug.*label|"
    r"across all projects.*with|with.*label.*and.*priority|"
    r"last \d+ days|created.*after|before.*sprint|"
    r"no assignee.*and|unassigned.*and.*label)",
    re.I,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Dimension scoring
# ═══════════════════════════════════════════════════════════════════════════════

def _is_infra(ans: str) -> bool:
    return any(p in ans.lower() for p in _INFRA_PHRASES)

def _is_refusal(ans: str) -> bool:
    return any(p in ans.lower() for p in _REFUSAL_PHRASES)

def _is_empty(ans: str) -> bool:
    return any(p in ans.lower() for p in _EMPTY_PHRASES)

def _has_structure(ans: str) -> bool:
    al = ans.lower()
    return any(s in al for s in _STRUCTURE_SIGNALS)

def _is_hedged_generic(ans: str, sources: list) -> bool:
    """True if the answer hedges generically without grounding in real data."""
    if sources:
        return False
    if len(ans) < 100:
        return False
    matches = _HEDGING_RE.findall(ans[:600])
    if not matches:
        return False
    has_ids  = bool(_JIRA_KEY_RE.search(ans))
    has_urls = bool(_GITLAB_URL_RE.search(ans))
    return len(matches) >= 2 and not has_ids and not has_urls


def score_record(r: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Score a single chat_history record across all benchmark dimensions.
    Returns {dim: float(0.0–1.0) | None (not applicable)}.
    """
    q       = (r.get("query") or "").strip()
    ans     = str(r.get("answer") or "").strip()
    intent  = r.get("intent", "both")
    sources = r.get("sources") or []
    error   = r.get("error")  # Detect hard API failures

    ans_lo  = ans.lower()
    q_lo    = q.lower()

    # Flag as infra if there's a pipeline error OR the model reported an API failure
    infra   = bool(error) or _is_infra(ans)
    refusal = _is_refusal(ans) and not infra
    empty   = _is_empty(ans)

    scores: Dict[str, Optional[float]] = {d: None for d in DIMS}

    # ── Cat A: refusal ────────────────────────────────────────────────────────
    # Only assessed on sentries intent — if the model has live data and refuses,
    # that is a DPO target.
    if intent == "sentries" and not infra:
        scores["refusal"] = 0.0 if refusal else 1.0

    # ── Cat D: format / field completeness ────────────────────────────────────
    # Sentry responses MUST include IDs, URLs, states. RAG prose is acceptable.
    if not infra and not refusal and len(ans) > 30:
        has_struct   = _has_structure(ans)
        has_table    = "|" in ans and "---" in ans
        has_numbered = bool(re.search(r"\d+\.\s+\*\*", ans))
        has_code     = "```" in ans
        is_count     = bool(re.match(r"^(there are|there is|\d+|answer:|il y a)", ans_lo[:50]))
        is_conv      = len(ans) < 80

        if intent == "sentries":
            if empty or is_count or is_conv:
                scores["format"] = 1.0
            elif has_code:
                scores["format"] = 1.0   # file content — code block is correct format
            elif has_struct or has_table or has_numbered:
                scores["format"] = 1.0
            else:
                scores["format"] = 0.0   # plain prose for multi-item sentry → fail
        else:
            # RAG / both — prose acceptable, just needs substance
            scores["format"] = 1.0 if len(ans) > 80 else 0.5

    # ── Cat B: fabrication ────────────────────────────────────────────────────
    # Is the model making things up when it should cite real data?
    if intent == "sentries":
        if infra:
            scores["fabrication"] = None    # infra failure — can't assess
        elif empty:
            scores["fabrication"] = 1.0     # honest empty = not fabricating
        elif _is_hedged_generic(ans, sources):
            scores["fabrication"] = 0.0     # hedged generic = fabrication signal
        elif sources or _JIRA_KEY_RE.search(ans) or _GITLAB_URL_RE.search(ans):
            scores["fabrication"] = 1.0     # grounded with IDs/URLs
        elif refusal:
            scores["fabrication"] = 1.0     # refusal ≠ fabrication
        else:
            # No sources, no IDs, not hedged — uncertain
            scores["fabrication"] = 0.7

    # ── Cat F: honest empty ───────────────────────────────────────────────────
    # When the API returns no data, does the model admit it or invent content?
    if empty and len(ans) < 500 and not infra:
        scores["honest_empty"] = 1.0
    elif _is_hedged_generic(ans, sources) and len(ans) > 200 and intent == "sentries":
        scores["honest_empty"] = 0.0    # should have said "no results" but invented

    # ── Cat E: French language consistency ───────────────────────────────────
    # French question → French answer required.
    # ── Cat E: French language consistency ───────────────────────────────────
    if _FRENCH_Q_RE.search(q):
        if infra:
            scores["french"] = None
        else:
            fr_ans = bool(_FRENCH_ANS_RE.search(ans[:400]))
            scores["french"] = 1.0 if fr_ans else 0.0

    # ── Cat I: cross-source synthesis ────────────────────────────────────────
    if intent == "both" and len(ans) > 40:
        if infra:
            scores["synthesis"] = None
        else:
            has_live = any(s in ans_lo for s in _LIVE_SIGNALS)
            has_doc  = any(s in ans_lo for s in _DOC_SIGNALS)

            if has_live and has_doc:
                scores["synthesis"] = 1.0
            elif has_live or has_doc:
                scores["synthesis"] = 0.5   
            else:
                scores["synthesis"] = 0.0   

    # ── Cat H: complex parameter generation ──────────────────────────────────
    if _COMPLEX_Q_RE.search(q) and intent == "sentries":
        if infra:
            scores["complex_params"] = None
        else:
            has_data   = _has_structure(ans) or bool(_JIRA_KEY_RE.search(ans))
            is_hedged2 = bool(_HEDGING_RE.search(ans[:400]))
            if has_data and not is_hedged2:
                scores["complex_params"] = 1.0
            elif is_hedged2:
                scores["complex_params"] = 0.0
            else:
                scores["complex_params"] = 0.5
    # ── Length quality ────────────────────────────────────────────────────────
    # Response proportionate to query — neither too short nor bloated.
    ans_len = len(ans)
    q_len   = len(q)
    if infra or refusal:
        scores["length_quality"] = None
    elif ans_len < 10:
        scores["length_quality"] = 0.0
    elif ans_len > 6_000:
        scores["length_quality"] = 0.3      # probably bloated
    elif q_len < 25 and ans_len > 3_000:
        scores["length_quality"] = 0.5      # short question, very long answer
    else:
        scores["length_quality"] = 1.0

    return scores


def compute_weighted_score(
    scores: Dict[str, Optional[float]], intent: str
) -> Optional[float]:
    """Compute single quality score from dimension scores."""
    weights = SCORE_WEIGHTS.get(intent, SCORE_WEIGHTS["both"])
    total_w = total_s = 0.0
    for dim, w in weights.items():
        v = scores.get(dim)
        if v is not None:
            total_s += v * w
            total_w += w
    return round(total_s / total_w, 4) if total_w > 0 else None


def training_signal(
    ws: Optional[float],
    scores: Dict[str, Optional[float]],
    intent: str,
) -> str:
    """Route a record to a training dataset bucket."""
    if ws is None:
        return "unevaluated"
    # Specific failure-mode routing (takes priority over score thresholds)
    if intent == "both" and (scores.get("synthesis") or 0) < 0.5:
        return "qlora_target_synthesis"
    if intent == "sentries" and scores.get("refusal") == 0.0:
        return "dpo_target_refusal"
    if scores.get("complex_params") == 0.0:
        return "grpo_target_params"
    # Score-based routing
    if ws >= 0.80:
        return "qlora_positive"
    if ws < 0.50:
        return "dpo_rejected"
    return "neutral"


# ═══════════════════════════════════════════════════════════════════════════════
# F1 / Precision / Recall
# ═══════════════════════════════════════════════════════════════════════════════

def compute_f1(
    values: List[float], threshold: float
) -> Dict[str, float]:
    """
    Binary F1 using threshold as pass/fail boundary.
    Single-category per-record design → FP = 0 by construction.
    F1 = 2×Recall / (1+Recall) = Recall when Precision = 1.0.
    """
    if not values:
        return {"n": 0, "tp": 0, "fn": 0, "fp": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "mean": 0.0, "min": 0.0, "max": 0.0}
    tp = sum(1 for v in values if v >= threshold)
    fn = sum(1 for v in values if v < threshold)
    fp = 0
    precision = 1.0  # by design (single-category, no cross-contamination)
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0.0
    return {
        "n":         len(values),
        "tp":        tp,
        "fn":        fn,
        "fp":        fp,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "mean":      round(sum(values) / len(values), 4),
        "min":       round(min(values), 4),
        "max":       round(max(values), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_ts(r: Dict) -> datetime:
    ts = r.get("timestamp") or {}
    ts_str = ts.get("$date", str(ts)) if isinstance(ts, dict) else str(ts)
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def load_from_mongo(last: Optional[int] = None) -> List[Dict]:
    """Load chat_history records from MongoDB."""
    if not MONGO_AVAILABLE:
        raise RuntimeError("pymongo not installed — use --json instead")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    db     = client[MONGO_DB]
    coll   = db["chat_history"]
    cursor = coll.find({}, sort=[("timestamp", DESCENDING)])
    if last:
        cursor = cursor.limit(last)
    records = list(cursor)
    # Reverse to get chronological order
    records.reverse()
    for r in records:
        r["_id_str"] = str(r.get("_id", ""))
        r.pop("_id", None)
    print(f"  Loaded {len(records)} records from MongoDB chat_history")
    return records


def load_from_json(path: str, last: Optional[int] = None) -> List[Dict]:
    """Load from JSON export file."""
    with open(path, encoding="utf-8") as f:
        raw = f.read().rstrip().rstrip(",") + "\n]"
        # Handle both array and newline-delimited JSON
        raw = raw.strip()
        if not raw.startswith("["):
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            data = [json.loads(l) for l in lines]
        else:
            data = json.loads(raw)
    data.sort(key=_parse_ts)
    if last:
        data = data[-last:]
    for r in data:
        r["_id_str"] = str(r.get("_id", r.get("conversation_id", "")))
    print(f"  Loaded {len(data)} records from {path}")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation engine
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(records: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Score every record, compute F1 per dimension, collect failures.
    Returns (per_record_results, aggregate).
    """
    per_record:   List[Dict]              = []
    dim_values:   Dict[str, List[float]]  = defaultdict(list)
    dim_failures: Dict[str, List[Dict]]   = defaultdict(list)
    ws_values:    List[float]             = []
    signal_counts = Counter()

    for r in records:
        scores  = score_record(r)
        ws      = compute_weighted_score(scores, r.get("intent", "both"))
        signal  = training_signal(ws, scores, r.get("intent", "both"))

        signal_counts[signal] += 1
        if ws is not None:
            ws_values.append(ws)

        # Accumulate per-dimension values and failures
        for dim, val in scores.items():
            if val is None:
                continue
            dim_values[dim].append(val)
            thr, _ = THRESHOLDS[dim]
            if val < thr:
                # This record FAILED this dimension — collect for export
                failure_record = _build_failure_record(r, dim, val, ws, signal, scores)
                dim_failures[dim].append(failure_record)

        row = {
            "query":          (r.get("query") or "")[:200],
            "intent":         r.get("intent", ""),
            "answer_len":     len(str(r.get("answer") or "")),
            "weighted_score": ws,
            "signal":         signal,
            "id":             r.get("_id_str", ""),
            "ts":             str(_parse_ts(r))[:19],
        }
        for d in DIMS:
            row[f"dim_{d}"] = scores[d]
        per_record.append(row)

    # Per-dimension F1
    dim_metrics = {}
    for dim in DIMS:
        vals = dim_values[dim]
        if not vals:
            continue
        thr, _ = THRESHOLDS[dim]
        dim_metrics[dim] = compute_f1(vals, thr)

    ws_metrics = compute_f1(ws_values, 0.80) if ws_values else {}

    aggregate = {
        "run_at":          datetime.now(timezone.utc).isoformat(),
        "total_records":   len(records),
        "intent_dist":     dict(Counter(r.get("intent") for r in records)),
        "signal_counts":   dict(signal_counts),
        "weighted_score":  ws_metrics,
        "dimensions":      dim_metrics,
        "failure_counts":  {d: len(v) for d, v in dim_failures.items()},
    }

    return per_record, aggregate, dim_failures


def _build_failure_record(
    r: Dict,
    dim: str,
    dim_score: float,
    ws: Optional[float],
    signal: str,
    all_scores: Dict[str, Optional[float]],
) -> Dict:
    """Build the failure document to write to MongoDB."""
    thr, method = THRESHOLDS[dim]
    return {
        "source_id":        r.get("_id_str", ""),
        "conversation_id":  r.get("conversation_id", ""),
        "dimension":        dim,
        "training_method":  method,
        "dim_score":        dim_score,
        "threshold":        thr,
        "gap":              round(thr - dim_score, 4),
        "weighted_score":   ws,
        "signal":           signal,
        "intent":           r.get("intent", ""),
        "query":            (r.get("query") or "")[:500],
        "answer":           (str(r.get("answer") or ""))[:1000],
        "answer_len":       len(str(r.get("answer") or "")),
        "sources":          r.get("sources", []),
        "all_dim_scores":   {d: v for d, v in all_scores.items() if v is not None},
        "flagged_at":       datetime.now(timezone.utc).isoformat(),
        "hammer_scored":    False,   # Hammer sets this to True after follow-up scoring
        "hammer_score":     None,
        "included_in_training": False,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MongoDB export
# ═══════════════════════════════════════════════════════════════════════════════

def write_failures_to_mongo(
    dim_failures: Dict[str, List[Dict]],
    run_id:       str,
    dry_run:      bool = False,
) -> Dict[str, int]:
    """
    Write failing records to their category collections in MongoDB.
    Uses upsert on source_id + dimension to avoid duplicates across runs.
    Returns {collection_name: records_written}.
    """
    if not MONGO_AVAILABLE:
        print("  [WARN] pymongo not available — skipping MongoDB write")
        return {}

    if dry_run:
        totals = {}
        for dim, records in dim_failures.items():
            col = FAILURE_COLLECTIONS[dim]
            totals[col] = len(records)
            print(f"  [DRY] Would write {len(records):>4} → {col}")
        return totals

    client  = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    db      = client[MONGO_DB]
    totals  = {}

    for dim, records in dim_failures.items():
        if not records:
            continue
        col  = FAILURE_COLLECTIONS[dim]
        coll = db[col]
        # Ensure index for upsert
        coll.create_index([("source_id", ASCENDING), ("dimension", ASCENDING)],
                          unique=True, background=True)
        inserted = updated = 0
        for rec in records:
            rec["run_id"] = run_id
            result = coll.update_one(
                {"source_id": rec["source_id"], "dimension": dim},
                {"$setOnInsert": rec},
                upsert=True,
            )
            if result.upserted_id:
                inserted += 1
            else:
                updated += 1
        total = inserted + updated
        totals[col] = total
        print(f"  ✓ {col:<45} {total:>4}  ({inserted} new, {updated} existing)")

    # Write run audit log
    db["benchmark_runs"].insert_one({
        "run_id":          run_id,
        "run_at":          datetime.now(timezone.utc).isoformat(),
        "total_evaluated": sum(len(v) for v in dim_failures.values()),
        "failure_counts":  {d: len(v) for d, v in dim_failures.items()},
    })
    print(f"  ✓ Run logged to benchmark_runs (run_id={run_id})")
    return totals


# ═══════════════════════════════════════════════════════════════════════════════
# Console report
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(aggregate: Dict, per_record: List[Dict]) -> None:
    total   = aggregate["total_records"]
    intents = aggregate["intent_dist"]
    sigs    = aggregate["signal_counts"]
    dims    = aggregate["dimensions"]
    ws      = aggregate.get("weighted_score", {})

    W = 90
    print()
    print("╔" + "═" * W + "╗")
    print("║" + "  ENTERPRISE BENCHMARK — AI Assistant — Static Assessment".center(W) + "║")
    print("╚" + "═" * W + "╝")
    print(f"\n  Records evaluated : {total}")
    print(f"  Intent dist       : {intents}")
    print(f"  Run at            : {aggregate.get('run_at','')[:19]} UTC")
    print()

    # Training signal distribution
    print("  TRAINING SIGNAL DISTRIBUTION")
    print("  " + "─" * 70)
    for sig, n in sorted(sigs.items(), key=lambda x: -x[1]):
        bar = "█" * min(35, int(n / max(total, 1) * 35))
        pct = n / max(total, 1) * 100
        print(f"  {sig:<35} {n:>4}  {pct:>5.1f}%  {bar}")
    print()

    # Dimension F1 table
    hdr = (f"  {'Dimension':<20} {'n':>4}  {'Pass':>5}  {'Fail':>5}  "
           f"{'Recall':>7}  {'F1':>7}  {'Thr':>5}  {'Status':<15}  Method")
    sep = "  " + "─" * (len(hdr) - 2)
    print(sep)
    print(hdr)
    print(sep)

    for dim in DIMS:
        m = dims.get(dim)
        if not m:
            continue
        thr, method = THRESHOLDS[dim]
        if m["f1"] < thr:
            status = "⚠  TRIGGERED"
        else:
            status = "   OK"
        print(
            f"  {dim:<20} {m['n']:>4}  {m['tp']:>5}  {m['fn']:>5}  "
            f"{m['recall']:>7.3f}  {m['f1']:>7.3f}  {thr:>5.2f}  "
            f"{status:<15}  {method}"
        )
    print(sep)

    if ws:
        print(f"\n  OVERALL WEIGHTED SCORE  n={ws.get('n',0)}"
              f"  mean={ws.get('mean',0):.3f}"
              f"  recall(≥0.80)={ws.get('recall',0):.3f}"
              f"  F1={ws.get('f1',0):.3f}")
    print()

    # Decision matrix
    print("  FINE-TUNING DECISION MATRIX")
    print("  " + "─" * 70)
    triggered = [(d, m) for d, m in dims.items() if m["f1"] < THRESHOLDS[d][0]]
    if not triggered:
        print("  ✓ All dimensions above threshold — no fine-tuning triggered")
        print("    QLoRA may still be prepared for scale (context overflow at 50+ projects)")
    else:
        for dim, m in sorted(triggered, key=lambda x: THRESHOLDS[x[0]][0] - x[1]["f1"], reverse=True):
            thr, method = THRESHOLDS[dim]
            gap  = thr - m["f1"]
            fail = m["fn"]
            col  = FAILURE_COLLECTIONS[dim]
            print(f"  ⚠  {method:<7} TRIGGERED  dim={dim:<20} "
                  f"F1={m['f1']:.3f}  thr={thr:.2f}  gap={gap:.3f}  "
                  f"failures={fail}  → {col}")
    print()

    # Failure samples for triggered dims
    triggered_dims = [d for d, m in triggered]
    if triggered_dims:
        print("  FAILURE SAMPLES (first 5 per triggered dimension):")
        print("  " + "─" * 70)
        for dim in triggered_dims[:4]:
            thr, _ = THRESHOLDS[dim]
            fails = [r for r in per_record
                     if r.get(f"dim_{dim}") is not None
                     and r[f"dim_{dim}"] < thr][:5]
            print(f"\n  [{dim}]")
            for row in fails:
                score_str = f"{row[f'dim_{dim}']:.2f}"
                print(f"    [{row['intent']:<8}] score={score_str} | "
                      f"Q: {row['query'][:70]}")
        print()

    # Pipeline advice
    print("  NEXT STEPS")
    print("  " + "─" * 70)
    if triggered:
        for dim, m in triggered:
            _, method = THRESHOLDS[dim]
            col = FAILURE_COLLECTIONS[dim]
            print(f"  {method:<8} → Review {col} in MongoDB")
            print(f"           → Run Hammer RAGAS mode on these records for full scoring")
            print(f"           → Export to training dataset via dataset_builder")
    print()
    print("  PIPELINE:")
    print("   1.  python enterprise_benchmark.py            [this script]")
    print("   2.  python -m hammer.run_hammer score --mode fast")
    print("   3.  python -m hammer.run_hammer score --mode ragas")
    print("   4.  python -m hammer.run_hammer dataset qlora-percentile")
    print("   5.  python -m hammer.run_hammer dataset dpo")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# JSONL dataset export (local files)
# ═══════════════════════════════════════════════════════════════════════════════

def export_local_datasets(per_record: List[Dict], out_dir: str) -> None:
    """Write JSONL files per training signal to a local directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    buckets: Dict[str, List] = defaultdict(list)
    for row in per_record:
        sig = row.get("signal", "neutral")
        buckets[sig].append({
            "query":          row["query"],
            "intent":         row["intent"],
            "weighted_score": row["weighted_score"],
            "dims":           {d: row.get(f"dim_{d}") for d in DIMS
                               if row.get(f"dim_{d}") is not None},
        })
    print(f"\n  Local JSONL exports → {out}/")
    for sig, rows in sorted(buckets.items(), key=lambda x: -len(x[1])):
        fpath = out / f"{sig}.jsonl"
        with fpath.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  ✓  {sig:<35} {len(rows):>4} rows  → {fpath.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="enterprise_benchmark",
        description="Enterprise static benchmark — scores chat_history and exports failures to MongoDB",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--json",  metavar="FILE",  help="Load from JSON/JSONL export instead of MongoDB")
    src.add_argument("--csv",   metavar="FILE",  help="Load from CSV export (columns: query, answer, intent)")

    parser.add_argument("--last",    type=int,   default=None, help="Evaluate last N records only")
    parser.add_argument("--intent",  default=None, help="Filter by intent: rag | sentries | both")
    parser.add_argument("--dry-run", action="store_true",
                        help="Score and report but do NOT write failures to MongoDB")
    parser.add_argument("--datasets", metavar="DIR", default=None,
                        help="Also export JSONL training datasets to this local directory")
    parser.add_argument("--json-out", metavar="FILE", default=None,
                        help="Write aggregate JSON to this path")
    parser.add_argument("--quiet",  action="store_true",
                        help="Suppress detailed report (just show decision matrix)")
    args = parser.parse_args()

    print()
    print("  Enterprise Benchmark — AI Assistant")
    print("  " + "─" * 50)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.json:
        records = load_from_json(args.json, args.last)
    elif args.csv:
        import csv as csv_mod
        records = []
        with open(args.csv, encoding="utf-8") as f:
            for row in csv_mod.DictReader(f):
                records.append(row)
        if args.last:
            records = records[-args.last:]
        print(f"  Loaded {len(records)} records from {args.csv}")
    else:
        if not MONGO_AVAILABLE:
            print("  ERROR: pymongo not installed. Use --json or install pymongo.")
            sys.exit(1)
        try:
            records = load_from_mongo(args.last)
        except Exception as e:
            print(f"  ERROR connecting to MongoDB: {e}")
            print("  Use --json to load from a file export instead.")
            sys.exit(1)

    if args.intent:
        records = [r for r in records if r.get("intent") == args.intent]
        print(f"  Filtered to intent={args.intent}: {len(records)} records")

    if not records:
        print("  No records to evaluate.")
        sys.exit(0)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"  Scoring {len(records)} records across {len(DIMS)} dimensions...")
    per_record, aggregate, dim_failures = evaluate(records)

    # ── Report ────────────────────────────────────────────────────────────────
    if not args.quiet:
        print_report(aggregate, per_record)

    # ── Write failures to MongoDB ─────────────────────────────────────────────
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    total_failures = sum(len(v) for v in dim_failures.values())

    if total_failures > 0:
        print(f"  Writing {total_failures} failing records to MongoDB collections...")
        write_failures_to_mongo(dim_failures, run_id, dry_run=args.dry_run)
    else:
        print("  ✓ No failures to write — all dimensions passed.")

    # ── Local JSONL export ────────────────────────────────────────────────────
    if args.datasets:
        export_local_datasets(per_record, args.datasets)

    # ── JSON aggregate ────────────────────────────────────────────────────────
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2, default=str)
        print(f"\n  ✓ Aggregate JSON → {args.json_out}")

    # ── Final summary ─────────────────────────────────────────────────────────
    triggered = [(d, m) for d, m in aggregate["dimensions"].items()
                 if m["f1"] < THRESHOLDS[d][0]]
    print()
    if triggered:
        print(f"  SUMMARY: {len(triggered)} dimension(s) triggered.")
        print(f"  Failing records are now in MongoDB — run Hammer next:")
        print(f"    python -m hammer.run_hammer score --mode fast")
    else:
        print("  SUMMARY: All dimensions passed. System is performing well.")
        print("  Prepare QLoRA pipeline for scale when project count approaches 50.")
    print()


if __name__ == "__main__":
    main()

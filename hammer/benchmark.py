#!/usr/bin/env python3
"""
Hammer Benchmark
════════════════
Runs a held-out evaluation set against the current system and compares
results to the registered baseline before any adapter is deployed.

Per the evaluation report:
  "Every adapter must improve weighted_score and not regress faithfulness
   below 0.80."

Workflow:
  1. Load eval set from JSONL file (held-out, never used for training).
  2. Score each sample with the evaluator metrics.
  3. Compare aggregate scores to the baseline in model_versions.
  4. Return a deployment decision: should_deploy (bool).
  5. Register results in model_versions if a version name is given.

Eval set format (one JSON object per line):
  {
    "query":            "<user question>",
    "expected_answer":  "<ideal response>",      ← used for comparison
    "generated_answer": "<model response>",       ← what we're scoring
    "contexts":         ["<passage1>", ...],      ← optional: pre-loaded context
    "intent":           "rag" | "sentries" | "both",
    "sources":          [ ... ]                   ← optional: source metadata
  }

If "generated_answer" is absent but "expected_answer" is present, the
sample is scored as if the system produced the expected answer (useful
for verifying the eval harness before a model is available).

Deployment rules (from the report):
  MUST improve:    weighted_score vs baseline
  MUST NOT breach: faithfulness < 0.80 on average
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
MONGO_URI      = os.getenv("MONGO_URI",      "mongodb://localhost:27017/")
MONGO_DB       = os.getenv("MONGO_DB",       "knowledge_base")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",   "qwen3").strip()

# Deployment gates
GATE_MIN_FAITHFULNESS   = float(os.getenv("GATE_MIN_FAITHFULNESS",   "0.80"))
# Require at least +0.005 improvement over baseline — equal score is not enough
GATE_MIN_SCORE_DELTA    = float(os.getenv("GATE_MIN_SCORE_DELTA",    "0.005"))
# GATE_ALLOW_EQUAL=false: a zero-delta adapter (no improvement) must NOT be deployed.
# Override with env var GATE_ALLOW_EQUAL=true only for explicit A/B baseline pinning.
GATE_ALLOW_EQUAL        = os.getenv("GATE_ALLOW_EQUAL", "false").lower() == "true"
# Per-intent sentries faithfulness floor (was hardcoded 0.75)
GATE_MIN_SENTRIES_FAITH = float(os.getenv("GATE_MIN_SENTRIES_FAITH", "0.75"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hammer.benchmark")


def _get_db():
    return MongoClient(MONGO_URI)[MONGO_DB]


# ═══════════════════════════════════════════════════════════════════════════════
# Eval set loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_eval_set(path: Path) -> List[Dict]:
    """Load and validate a JSONL eval set file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval set not found: {path}")

    samples = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping invalid JSON on line %d: %s", i, e)
                continue

            # Validate required fields
            if "query" not in sample:
                logger.warning("Skipping line %d: missing 'query'", i)
                continue
            if "expected_answer" not in sample and "generated_answer" not in sample:
                logger.warning("Skipping line %d: missing expected_answer or generated_answer", i)
                continue

            samples.append(sample)

    logger.info("Loaded %d eval samples from %s", len(samples), path)
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def _token_f1(reference: str, hypothesis: str) -> float:
    """
    Token-level F1 between reference and hypothesis answers.

    Used as an additional reference-based signal in the benchmark when
    expected_answer is available. Complements RAGAS/validator (which are
    reference-free) with a lexical overlap check.

    Algorithm: bag-of-words overlap, stop-words stripped.
    Returns 0.0–1.0.
    """
    _STOP = {"the", "a", "an", "is", "are", "was", "were", "it", "its",
             "this", "that", "to", "of", "in", "for", "on", "and", "or",
             "be", "been", "being", "have", "has", "had", "do", "does", "did"}

    def _tok(text: str):
        return [t.lower() for t in re.findall(r"\b\w+\b", text) if t.lower() not in _STOP]

    ref_toks  = _tok(reference)
    hyp_toks  = _tok(hypothesis)
    if not ref_toks or not hyp_toks:
        return 0.0

    ref_counts = {}
    for t in ref_toks:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    hyp_counts = {}
    for t in hyp_toks:
        hyp_counts[t] = hyp_counts.get(t, 0) + 1

    overlap = sum(min(ref_counts.get(t, 0), hyp_counts.get(t, 0)) for t in hyp_counts)
    precision = overlap / len(hyp_toks) if hyp_toks else 0.0
    recall    = overlap / len(ref_toks)  if ref_toks  else 0.0

    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def score_sample(sample: Dict) -> Dict[str, Any]:
    """
    Score a single eval sample using the evaluator metrics.

    Returns a result dict with faithfulness, answer_relevance, temporal,
    code, weighted_score, scoring_method, and (when expected_answer present)
    token_f1 as a reference-based lexical overlap metric.
    """
    from hammer.evaluator import (
        score_ragas,
        score_temporal_consistency,
        score_code_correctness,
        compute_weighted_score,
    )
    from hammer.validator import validate_sentry_answer, entity_match_relevance

    query     = sample["query"]
    answer    = sample.get("generated_answer") or sample.get("expected_answer", "")
    reference = sample.get("expected_answer", "")
    intent    = sample.get("intent", "rag")
    sources   = sample.get("sources", [])

    result: Dict[str, Any] = {
        "query":  query,
        "intent": intent,
        "answer_preview": answer[:100],
    }

    # Temporal + code (always)
    temporal = score_temporal_consistency(answer)
    code     = score_code_correctness(answer)
    result["temporal_consistency"] = temporal
    result["code_correctness"]     = code

    if intent in ("rag", "both"):
        # Use pre-loaded contexts if available in the eval sample, otherwise
        # fall back to Neo4j hybrid_search via retrieve_context (Tier 2).
        contexts = sample.get("contexts") or []
        if not contexts:
            try:
                from hammer.evaluator import retrieve_context
                contexts = retrieve_context(query, top_k=5, doc=None)
            except Exception:
                contexts = []

        rag_scores = score_ragas(query, answer, contexts)
        result["faithfulness"]     = rag_scores["faithfulness"]
        result["answer_relevance"] = rag_scores["answer_relevance"]
        result["scoring_method"]   = rag_scores["method"]
        result["num_contexts"]     = len(contexts)

    elif intent == "sentries":
        # FLATTENED FIX: query + sentries_summary forwarded so the validator's
        # query-echo exemptions and summary cross-check run inside the gate too.
        v_score, v_details = validate_sentry_answer(
            answer=answer,
            sources=sources,
            sentries_summary=sample.get("sentries_summary"),
            query=query,
        )
        result["faithfulness"]      = v_score
        result["answer_relevance"]  = entity_match_relevance(query, answer)
        result["scoring_method"]    = "validator"
        result["validator_details"] = v_details

    else:
        result["faithfulness"]     = 0.50
        result["answer_relevance"] = 0.50
        result["scoring_method"]   = "unknown_intent"

    result["weighted_score"] = compute_weighted_score(
        faithfulness    =result["faithfulness"],
        answer_relevance=result["answer_relevance"],
        temporal        =temporal,
        code            =code,
    )

    # Reference-based F1 — only when a gold expected_answer is available.
    # This is a benchmark-only signal (not used in production scoring).
    # It catches lexical regressions that semantic metrics can miss.
    if reference and answer and reference != answer:
        result["token_f1"] = _token_f1(reference, answer)
    else:
        result["token_f1"] = None

    return result


def run_benchmark(
    eval_set_path:  Path,
    version_name:   Optional[str] = None,
    training_method:Optional[str] = None,
    hyperparams:    Optional[Dict] = None,
    register:       bool = True,
) -> Dict[str, Any]:
    """
    Run the full benchmark and return a deployment decision.

    Args:
        eval_set_path:   Path to JSONL held-out eval set.
        version_name:    Name for this adapter in model_versions (e.g. "qwen3-lora-v1").
                         If None, results are not registered.
        training_method: "QLoRA" | "DPO" | "GRPO" (for model_versions record).
        hyperparams:     Dict of training hyperparameters for the registry record.
        register:        Write results to model_versions collection.

    Returns:
        {
          should_deploy:   bool,
          reason:          str,
          scores:          { weighted_score, faithfulness, answer_relevance, temporal },
          baseline:        { weighted_score, faithfulness } or None,
          delta:           { weighted_score, faithfulness } or None,
          per_sample:      [ ... ],
          version_name:    str or None,
        }
    """
    logger.info("=" * 60)
    logger.info("Hammer Benchmark")
    logger.info("=" * 60)

    # Load eval set
    samples = load_eval_set(eval_set_path)
    if not samples:
        return {
            "should_deploy": False,
            "reason":        "empty_eval_set",
            "scores":        {},
            "baseline":      None,
        }

    # Score all samples
    logger.info("Scoring %d samples...", len(samples))
    per_sample = []
    for i, sample in enumerate(samples, 1):
        result = score_sample(sample)
        per_sample.append(result)
        if i % 10 == 0:
            logger.info("  %d/%d scored", i, len(samples))

    # ── Aggregate overall ─────────────────────────────────────────────────────
    def avg(key: str, subset=None) -> float:
        src = subset if subset is not None else per_sample
        vals = [r[key] for r in src if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    scores = {
        "weighted_score":    avg("weighted_score"),
        "faithfulness":      avg("faithfulness"),
        "answer_relevance":  avg("answer_relevance"),
        "temporal":          avg("temporal_consistency"),
        "token_f1":          avg("token_f1"),   # None values auto-excluded by isinstance check
        "num_samples":       len(per_sample),
    }

    # ── Per-intent breakdown ──────────────────────────────────────────────────
    intents = set(r.get("intent", "rag") for r in per_sample)
    per_intent: Dict[str, Any] = {}
    for intent_label in sorted(intents):
        subset = [r for r in per_sample if r.get("intent") == intent_label]
        per_intent[intent_label] = {
            "n":               len(subset),
            "weighted_score":  avg("weighted_score",   subset),
            "faithfulness":    avg("faithfulness",     subset),
            "answer_relevance":avg("answer_relevance", subset),
            "token_f1":        avg("token_f1",         subset),
        }

    logger.info("")
    logger.info("Benchmark Results (overall, n=%d):", scores["num_samples"])
    logger.info("  weighted_score   : %.4f", scores["weighted_score"])
    logger.info("  faithfulness     : %.4f", scores["faithfulness"])
    logger.info("  answer_relevance : %.4f", scores["answer_relevance"])
    logger.info("  token_f1         : %.4f", scores["token_f1"])
    logger.info("  temporal         : %.4f", scores["temporal"])
    logger.info("")
    logger.info("Per-intent breakdown:")
    for label, pi in per_intent.items():
        logger.info(
            "  %-10s  n=%-3d  score=%.4f  faith=%.4f  ar=%.4f  f1=%.4f",
            label, pi["n"], pi["weighted_score"], pi["faithfulness"],
            pi["answer_relevance"], pi["token_f1"],
        )

    # Compare to baseline
    db           = _get_db()
    baseline_doc = db["model_versions"].find_one(
        {"deployed": True},
        sort=[("deployed_at", -1)],
    )
    baseline = None
    delta    = None

    if baseline_doc:
        b_scores = baseline_doc.get("benchmark", {})
        baseline = {
            "weighted_score": b_scores.get("weighted_score", 0.0),
            "faithfulness":   b_scores.get("faithfulness",   0.0),
            "version":        baseline_doc.get("version",    "baseline"),
        }
        delta = {
            "weighted_score": round(scores["weighted_score"] - baseline["weighted_score"], 4),
            "faithfulness":   round(scores["faithfulness"]   - baseline["faithfulness"],   4),
        }
        logger.info("")
        logger.info("vs Baseline (%s):", baseline["version"])
        logger.info("  Δ weighted_score   : %+.4f", delta["weighted_score"])
        logger.info("  Δ faithfulness     : %+.4f", delta["faithfulness"])

    # ── Deployment gates ──────────────────────────────────────────────────────
    should_deploy = True
    reason        = "all_gates_passed"

    # Gate 1: overall faithfulness must not drop below 0.80
    if scores["faithfulness"] < GATE_MIN_FAITHFULNESS:
        should_deploy = False
        reason = (
            f"faithfulness={scores['faithfulness']:.4f} "
            f"below minimum={GATE_MIN_FAITHFULNESS:.2f}"
        )

    # Gate 2: per-intent faithfulness — sentries must not drop below 0.75
    # A QLoRA run could improve RAG (0.95→0.97) but regress sentries (0.79→0.71)
    # and still pass the overall gate. This gate catches that.
    if should_deploy and "sentries" in per_intent:
        sentries_faith = per_intent["sentries"]["faithfulness"]
        if sentries_faith < GATE_MIN_SENTRIES_FAITH:
            should_deploy = False
            reason = (
                f"sentries faithfulness={sentries_faith:.4f} "
                f"below per-intent minimum={GATE_MIN_SENTRIES_FAITH:.2f}"
            )

    # Gate 3: weighted_score must improve vs baseline by at least GATE_MIN_SCORE_DELTA
    # FLATTENED FIX: GATE_ALLOW_EQUAL previously had NO effect (both branches ran
    # the identical comparison). Allow-equal now admits any delta >= 0 (explicit
    # baseline pinning); strict mode requires >= GATE_MIN_SCORE_DELTA.
    if should_deploy and delta is not None:
        _gate3_threshold = 0.0 if GATE_ALLOW_EQUAL else GATE_MIN_SCORE_DELTA
        if delta["weighted_score"] < _gate3_threshold:
            should_deploy = False
            _mode = "allow-equal (>=0)" if GATE_ALLOW_EQUAL else f">= +{GATE_MIN_SCORE_DELTA:.3f}"
            reason = (
                f"score_delta={delta['weighted_score']:+.4f} "
                f"fails gate ({_mode})"
            )

    logger.info("")
    logger.info("Deployment decision : %s", "✓ DEPLOY" if should_deploy else "✗ DO NOT DEPLOY")
    logger.info("Reason              : %s", reason)

    # ── Register in model_versions ────────────────────────────────────────────
    if register and version_name:
        versions = db["model_versions"]
        versions.update_one(
            {"version": version_name},
            {"$set": {
                "version":          version_name,
                "training_method":  training_method or "unknown",
                "base_model":       OLLAMA_MODEL,
                "hyperparams":      hyperparams or {},
                "benchmark": {
                    "weighted_score":   scores["weighted_score"],
                    "faithfulness":     scores["faithfulness"],
                    "answer_relevance": scores["answer_relevance"],
                    "token_f1":         scores["token_f1"],
                    "temporal":         scores["temporal"],
                    "num_samples":      scores["num_samples"],
                    "per_intent":       per_intent,
                    "eval_set":         str(eval_set_path),
                },
                "baseline_comparison": {
                    "baseline_version": baseline["version"] if baseline else None,
                    "delta":            delta,
                },
                "should_deploy":    should_deploy,
                "deploy_reason":    reason,
                "deployed":         False,
                "deployed_at":      None,
                "benchmarked_at":   datetime.utcnow(),
                "created_at":       datetime.utcnow(),
            }},
            upsert=True,
        )
        logger.info("✓ Registered in model_versions: %s", version_name)

    return {
        "should_deploy":  should_deploy,
        "reason":         reason,
        "scores":         scores,
        "per_intent":     per_intent,
        "baseline":       baseline,
        "delta":          delta,
        "per_sample":     per_sample,
        "version_name":   version_name,
        "benchmarked_at": datetime.utcnow().isoformat(),
    }


def mark_deployed(version_name: str) -> bool:
    """
    Mark an adapter version as deployed in model_versions.
    Call this AFTER successful Ollama model swap.

    Returns True if the version was found and updated.
    """
    db     = _get_db()
    result = db["model_versions"].update_one(
        {"version": version_name},
        {"$set": {"deployed": True, "deployed_at": datetime.utcnow()}},
    )
    if result.matched_count:
        logger.info("✓ Marked as deployed: %s", version_name)
        return True
    logger.warning("Version not found: %s", version_name)
    return False


def list_versions() -> List[Dict]:
    """Return all model versions sorted by benchmark date."""
    db  = _get_db()
    return list(
        db["model_versions"].find(
            {},
            {
                "_id": 0,
                "version": 1,
                "training_method": 1,
                "benchmark.weighted_score": 1,
                "benchmark.faithfulness": 1,
                "should_deploy": 1,
                "deployed": 1,
                "deployed_at": 1,
                "benchmarked_at": 1,
            },
        ).sort("benchmarked_at", -1)
    )


def create_sample_eval_set(output_path: Optional[Path] = None) -> Path:
    """
    Create a minimal sample eval set JSONL to get started.
    The responses are intentionally left empty — fill them in after
    running the system on these queries.
    """
    output_path = Path(output_path or "datasets/eval_set_template.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = [
        # RAG samples
        {"query": "How do I authenticate with OAuth?",
         "expected_answer": "", "generated_answer": "", "intent": "rag",
         "contexts": []},
        {"query": "What is the database schema for users?",
         "expected_answer": "", "generated_answer": "", "intent": "rag",
         "contexts": []},
        {"query": "How do I submit a merge request?",
         "expected_answer": "", "generated_answer": "", "intent": "rag",
         "contexts": []},
        # Sentries samples
        {"query": "Show open issues in auth-service",
         "expected_answer": "", "generated_answer": "", "intent": "sentries",
         "sources": []},
        {"query": "List recent pipelines for ecommerce-backend",
         "expected_answer": "", "generated_answer": "", "intent": "sentries",
         "sources": []},
        # Both samples
        {"query": "What is AUTH-12 about and how should I fix it?",
         "expected_answer": "", "generated_answer": "", "intent": "both",
         "contexts": [], "sources": []},
        {"query": "Explain the API rate limiting and show current bugs",
         "expected_answer": "", "generated_answer": "", "intent": "both",
         "contexts": [], "sources": []},
    ]

    with output_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info("Sample eval set written to %s", output_path)
    logger.info("Fill in 'generated_answer' fields by running the assistant on each query.")
    return output_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hammer Benchmark")
    sub    = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run benchmark against eval set")
    p_run.add_argument("eval_set", type=str, help="Path to JSONL eval set")
    p_run.add_argument("--version",  type=str, default=None, help="Adapter version name to register")
    p_run.add_argument("--method",   type=str, default=None, help="Training method (QLoRA/DPO/GRPO)")
    p_run.add_argument("--no-register", action="store_true",  help="Do not write to model_versions")
    p_run.add_argument("--out",      type=str, default=None,  help="Write full result JSON to file")

    p_deploy = sub.add_parser("deploy", help="Mark a version as deployed")
    p_deploy.add_argument("version", type=str)

    p_list = sub.add_parser("list", help="List all registered versions")

    p_init = sub.add_parser("init-eval-set", help="Create a template eval set")
    p_init.add_argument("--out", type=str, default=None)

    args = parser.parse_args()

    if args.cmd == "run":
        result = run_benchmark(
            eval_set_path   = Path(args.eval_set),
            version_name    = args.version,
            training_method = args.method,
            register        = not args.no_register,
        )
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info("Full result written to %s", args.out)
        print("\nshould_deploy:", result["should_deploy"])
        print("reason:       ", result["reason"])

    elif args.cmd == "deploy":
        mark_deployed(args.version)

    elif args.cmd == "list":
        import pprint
        pprint.pprint(list_versions())

    elif args.cmd == "init-eval-set":
        create_sample_eval_set(Path(args.out) if args.out else None)

    else:
        parser.print_help()

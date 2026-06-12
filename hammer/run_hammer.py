#!/usr/bin/env python3
"""
Hammer Orchestrator  (run_hammer.py)
══════════════════════════════════════
Central CLI for all Hammer operations. Runs asynchronously alongside
production — never touches the LangGraph inference graph.

Commands:
  score        — Run the evaluator on all unscored chat_history docs
  check        — Rolling quality check (alerts if score regresses)
  dataset      — Export training datasets (qlora / dpo / grpo / seed / all)
  benchmark    — Run the adapter gate benchmark before deployment
  deploy       — Mark a version as deployed in model_versions
  status       — Print full system status (scores, dataset readiness, versions)
  watch        — Run continuously: score + check every N hours

Usage:
  python hammer/run_hammer.py score
  python hammer/run_hammer.py score --limit 50 --force
  python hammer/run_hammer.py check
  python hammer/run_hammer.py dataset all
  python hammer/run_hammer.py dataset qlora --out my_train.jsonl
  python hammer/run_hammer.py benchmark datasets/eval_set.jsonl --version qwen3-lora-v1
  python hammer/run_hammer.py deploy qwen3-lora-v1
  python hammer/run_hammer.py status
  python hammer/run_hammer.py watch --interval 6
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hammer.run")

# ─── Make sure project root is importable ─────────────────────────────────────
_HERE    = Path(__file__).resolve().parent          # hammer/
_PROJECT = _HERE.parent                             # project root
for _p in [str(_HERE.parent), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ═══════════════════════════════════════════════════════════════════════════════
# Command handlers
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_score(args) -> None:
    """Run the evaluator on unscored chat_history documents."""
    from hammer.evaluator import run_evaluator
    summary = run_evaluator(
        batch_size    = args.batch,
        force_rescore = args.force,
        limit         = args.limit,
        dry_run       = args.dry_run,
        workers       = args.workers,
    )
    logger.info("Done: %s", json.dumps(summary))


def cmd_check(args) -> None:
    """Rolling quality check — alerts if regression detected."""
    from hammer.evaluator import rolling_quality_check
    result = rolling_quality_check(
        window          = args.window,
        alert_threshold = args.threshold,
    )
    print(json.dumps(result, indent=2, default=str))
    if result.get("should_alert"):
        logger.warning(
            "⚠ ALERT: Quality regression! "
            "rolling=%.4f  baseline=%.4f  drop=%.4f",
            result["rolling_avg"],
            result.get("baseline", 0),
            result.get("regression", 0),
        )
        sys.exit(2)   # non-zero exit so CI/cron jobs can catch it


def cmd_dataset(args) -> None:
    """Export training datasets."""
    from hammer.dataset_builder import (
        export_qlora, export_dpo, export_grpo,
        export_seed_dataset, dataset_stats,
    )

    which = args.which.lower()
    out   = Path(args.out) if args.out else None

    if which in ("qlora", "all"):
        n = export_qlora(out, limit=args.limit)
        logger.info("QLoRA: %d records", n)

    if which in ("dpo", "all"):
        n = export_dpo(out, limit=args.limit)
        logger.info("DPO:   %d records", n)

    if which in ("grpo", "all"):
        n = export_grpo(out, limit=args.limit)
        logger.info("GRPO:  %d records", n)

    if which in ("seed", "all"):
        n = export_seed_dataset(out, limit=args.limit or 30)
        logger.info("Seed:  %d records", n)

    if which == "stats":
        import pprint
        pprint.pprint(dataset_stats())

    if which not in ("qlora", "dpo", "grpo", "seed", "all", "stats"):
        logger.error("Unknown dataset type: %s. Use: qlora / dpo / grpo / seed / all / stats", which)
        sys.exit(1)


def cmd_benchmark(args) -> None:
    """Run the adapter gate benchmark."""
    from hammer.benchmark import run_benchmark

    result = run_benchmark(
        eval_set_path   = Path(args.eval_set),
        version_name    = args.version,
        training_method = args.method,
        register        = not args.no_register,
    )

    print(f"\n{'─' * 50}")
    print(f"  should_deploy  : {'✓ YES' if result['should_deploy'] else '✗ NO'}")
    print(f"  reason         : {result['reason']}")
    print(f"  weighted_score : {result['scores'].get('weighted_score', '?')}")
    print(f"  faithfulness   : {result['scores'].get('faithfulness',   '?')}")
    print(f"  token_f1       : {result['scores'].get('token_f1',       '?')}")
    if result.get("delta"):
        print(f"  Δ score        : {result['delta']['weighted_score']:+.4f}")
        print(f"  Δ faithfulness : {result['delta']['faithfulness']:+.4f}")
    if result.get("per_intent"):
        print(f"  Per-intent:")
        for label, pi in result["per_intent"].items():
            print(f"    {label:<10} n={pi['n']:<3} score={pi['weighted_score']:.4f}  "
                  f"faith={pi['faithfulness']:.4f}  ar={pi['answer_relevance']:.4f}  "
                  f"f1={pi['token_f1']:.4f}")
    print(f"{'─' * 50}\n")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)

    sys.exit(0 if result["should_deploy"] else 1)


def cmd_deploy(args) -> None:
    """Mark a model version as deployed."""
    from hammer.benchmark import mark_deployed
    ok = mark_deployed(args.version)
    sys.exit(0 if ok else 1)


def cmd_status(args) -> None:
    """Print full Hammer system status."""
    from hammer.evaluator import rolling_quality_check
    from hammer.dataset_builder import dataset_stats
    from hammer.benchmark import list_versions

    print("\n" + "═" * 60)
    print("  HAMMER STATUS")
    print("═" * 60)

    # Quality check
    try:
        qc = rolling_quality_check()
        print(f"\n  Rolling quality (last {qc.get('window', 50)} responses):")
        print(f"    rolling_avg  : {qc.get('rolling_avg', 'N/A')}")
        print(f"    baseline     : {qc.get('baseline', 'N/A')}")
        print(f"    regression Δ : {qc.get('regression', 'N/A')}")
        print(f"    should_alert : {qc.get('should_alert', False)}")
    except Exception as e:
        print(f"\n  Quality check failed: {e}")

    # Dataset readiness
    try:
        stats = dataset_stats()
        print(f"\n  Dataset readiness (2D classification):")
        print(f"    total responses    : {stats['total_responses']}")
        print(f"    scored             : {stats['scored']}")
        print(f"    unscored           : {stats['unscored']}")
        print(f"\n  Quality grades:")
        grades = stats.get("grades", {})
        gtotal = stats.get("graded_total", 0)
        for g in ("GOLD", "SILVER", "BRONZE", "FAILED"):
            n = grades.get(g, 0)
            pct = (100 * n / gtotal) if gtotal else 0
            print(f"    {g:<7} : {n:>4}  ({pct:5.1f}%)")
        print(f"\n  Failure tags (non-zero):")
        for t, n in sorted(stats.get("failure_tags", {}).items(), key=lambda x: -x[1]):
            critical = " [CRITICAL]" if t in ("hallucination", "tool_misuse", "retrieval_miss") else ""
            print(f"    {t:<18} : {n:>4}{critical}")
        print(f"\n  Training-data pool:")
        print(f"    QLoRA (GOLD)       : {grades.get('GOLD', 0)}")
        print(f"    DPO candidates     : {stats['dpo_pending_review']}")
        print(f"\n  Legacy (training_signal, backward-compat):")
        print(f"    qlora_positive     : {stats['qlora_ready']}")
        print(f"    dpo_rejected       : {stats['dpo_ready']}")
    except Exception as e:
        print(f"\n  Dataset stats failed: {e}")

    # Model versions
    try:
        versions = list_versions()
        print(f"\n  Model versions ({len(versions)} registered):")
        for v in versions[:5]:
            deployed_flag = "✓ DEPLOYED" if v.get("deployed") else "  staging"
            score = v.get("benchmark", {}).get("weighted_score", "?")
            faith = v.get("benchmark", {}).get("faithfulness",   "?")
            print(f"    {deployed_flag}  {v.get('version','?'):<25}  "
                  f"score={score}  faith={faith}  "
                  f"method={v.get('training_method','?')}")
    except Exception as e:
        print(f"\n  Version list failed: {e}")

    print("\n" + "═" * 60 + "\n")


def cmd_watch(args) -> None:
    """
    Run Hammer continuously: score + quality check every N hours.
    Intended to be started as a background process or systemd service.
    """
    interval_s = args.interval * 3600
    logger.info("Starting continuous Hammer watch (interval=%dh)", args.interval)

    while True:
        logger.info("─" * 60)
        logger.info("Hammer cycle started at %s", datetime.now().isoformat())

        # Score new documents
        try:
            from hammer.evaluator import run_evaluator
            summary = run_evaluator(batch_size=20)
            logger.info("Score cycle: %s", json.dumps(summary))
        except Exception as e:
            logger.error("Score cycle failed: %s", e)

        # Quality check
        try:
            from hammer.evaluator import rolling_quality_check
            qc = rolling_quality_check()
            if qc.get("should_alert"):
                logger.warning("QUALITY ALERT: regression=%.4f", qc.get("regression", 0))
        except Exception as e:
            logger.error("Quality check failed: %s", e)

        logger.info("Hammer cycle done. Sleeping %dh...", args.interval)
        time.sleep(interval_s)


def cmd_init_eval_set(args) -> None:
    """Create a template eval set."""
    from hammer.benchmark import create_sample_eval_set
    out = Path(args.out) if args.out else None
    create_sample_eval_set(out)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI wiring
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="run_hammer",
        description="Hammer — Evaluation & Training Data Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── score ────────────────────────────────────────────────────────────────
    p_score = sub.add_parser("score", help="Score unscored chat_history documents")
    p_score.add_argument("--limit",   type=int,  default=None, help="Max docs to score")
    p_score.add_argument("--batch",   type=int,  default=20,   help="Log progress every N docs")
    p_score.add_argument("--force",   action="store_true",     help="Re-score already-scored docs")
    p_score.add_argument("--dry-run", action="store_true",     help="Score but do not write to MongoDB")
    p_score.add_argument("--workers", type=int,  default=2,    help="Parallel worker threads (default: 2; matches cloud endpoint concurrency limit)")

    # ── check ────────────────────────────────────────────────────────────────
    p_check = sub.add_parser("check", help="Rolling quality check")
    p_check.add_argument("--window",    type=int,   default=50,   help="Number of recent responses")
    p_check.add_argument("--threshold", type=float, default=0.05, help="Regression threshold for alert")

    # ── dataset ──────────────────────────────────────────────────────────────
    p_data = sub.add_parser(
        "dataset",
        help="Export training datasets (qlora / dpo / grpo / seed / all / stats)",
    )
    p_data.add_argument("which", type=str,
                        help="qlora | dpo | grpo | seed | all | stats")
    p_data.add_argument("--limit", type=int,  default=None, help="Max records to export")
    p_data.add_argument("--out",   type=str,  default=None, help="Output file path")

    # ── benchmark ────────────────────────────────────────────────────────────
    p_bench = sub.add_parser("benchmark", help="Run adapter gate benchmark")
    p_bench.add_argument("eval_set",       type=str,            help="Path to JSONL eval set")
    p_bench.add_argument("--version",      type=str, default=None)
    p_bench.add_argument("--method",       type=str, default=None,
                         help="Training method: QLoRA | DPO | GRPO")
    p_bench.add_argument("--no-register",  action="store_true", help="Skip model_versions write")
    p_bench.add_argument("--out",          type=str, default=None, help="Write JSON result to file")

    # ── deploy ───────────────────────────────────────────────────────────────
    p_deploy = sub.add_parser("deploy", help="Mark a version as deployed")
    p_deploy.add_argument("version", type=str)

    # ── status ───────────────────────────────────────────────────────────────
    sub.add_parser("status", help="Print full Hammer system status")

    # ── watch ────────────────────────────────────────────────────────────────
    p_watch = sub.add_parser("watch", help="Run Hammer continuously (background mode)")
    p_watch.add_argument("--interval", type=int, default=6, help="Hours between cycles (default: 6)")

    # ── init-eval-set ────────────────────────────────────────────────────────
    p_init = sub.add_parser("init-eval-set", help="Create a template eval set JSONL")
    p_init.add_argument("--out", type=str, default=None)

    args = parser.parse_args()

    dispatch = {
        "score":        cmd_score,
        "check":        cmd_check,
        "dataset":      cmd_dataset,
        "benchmark":    cmd_benchmark,
        "deploy":       cmd_deploy,
        "status":       cmd_status,
        "watch":        cmd_watch,
        "init-eval-set":cmd_init_eval_set,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

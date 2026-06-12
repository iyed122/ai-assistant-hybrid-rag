#!/usr/bin/env python3
"""
benchmark_calculator.py
═══════════════════════
Computes Precision, Recall, F1 and derives fine-tuning strategy
recommendations after running the 200-prompt benchmark from
BENCHMARK_EVALUATION_FRAMEWORK_v2.

Workflow
────────
1.  Run all 200 prompts through your system.
2.  For each prompt, record: PASS (1) or FAIL (0).
3.  Feed the results to this script — either via CSV or the
    interactive helper function.
4.  The script computes per-category F1 and prints:
    • A full breakdown table
    • Which fine-tuning strategies are triggered (DPO / QLoRA / GRPO)
    • A plain-language recommendation for your supervisor

Usage
─────
  # Interactive demo (built-in sample results):
  python benchmark_calculator.py --demo

  # From CSV (columns: prompt_id, result — 1=PASS 0=FAIL):
  python benchmark_calculator.py --csv results.csv

  # From JSON (list of {"id": "P001", "result": 1}):
  python benchmark_calculator.py --json results.json

  # Print empty CSV template to fill in:
  python benchmark_calculator.py --template > my_results.csv
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# CATEGORY DEFINITIONS
# Matches BENCHMARK_EVALUATION_FRAMEWORK_v2 exactly.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Category:
    name: str
    short:       str          # single letter
    prompt_ids:  List[str]    # e.g. ["P001", ..., "P022"]
    failure_mode: str
    f1_threshold: float       # below this → fine-tuning triggered
    method:       str         # DPO / QLoRA / GRPO
    justification: str        # argument to supervisor if triggered


CATEGORIES: List[Category] = [
    Category(
        name="Refusal Behavior",
        short="A",
        prompt_ids=[f"P{i:03d}" for i in range(1, 23)],
        failure_mode="Model refuses despite data in context",
        f1_threshold=0.85,
        method="DPO",
        justification=(
            "DPO is required. Refusal persists despite explicit prompt prohibition. "
            "This is a base model prior overriding your instructions. "
            "DPO makes correct (non-refusal) behavior a native preference, not a fragile instruction."
        ),
    ),
    Category(
        name="Fabrication Detection",
        short="B",
        prompt_ids=[f"P{i:03d}" for i in range(23, 45)],
        failure_mode="Model invents IDs / keys / URLs not in retrieved data",
        f1_threshold=0.80,
        method="DPO",
        justification=(
            "DPO is required. The model fabricates specifics (IDs, URLs, commit hashes) not present "
            "in the retrieved payload. Preference training grounds answers strictly in context data."
        ),
    ),
    Category(
        name="Routing Accuracy",
        short="C",
        prompt_ids=[f"P{i:03d}" for i in range(45, 73)],
        failure_mode="Wrong source queried (RAG vs live API)",
        f1_threshold=0.90,
        method="QLoRA",
        justification=(
            "QLoRA is required. The intent classifier routes to the wrong source on a measurable "
            "fraction of queries. Baking correct source-selection into the model weights at zero "
            "inference overhead is more reliable than prompt engineering."
        ),
    ),
    Category(
        name="Format / Field Completeness",
        short="D",
        prompt_ids=[f"P{i:03d}" for i in range(73, 95)],
        failure_mode="Missing required fields (IDs, URLs, status, assignee) in response",
        f1_threshold=0.80,
        method="QLoRA",
        justification=(
            "QLoRA is required. Responses inconsistently omit required fields from live API payloads. "
            "Fine-tuning on format-complete examples bakes the desired structure into weights."
        ),
    ),
    Category(
        name="Language Consistency (French)",
        short="E",
        prompt_ids=[f"P{i:03d}" for i in range(95, 113)],
        failure_mode="Responds in English when question was asked in French",
        f1_threshold=0.85,
        method="QLoRA",
        justification=(
            "QLoRA is required. The base model follows language instructions ~70% of the time. "
            "QLoRA on multilingual examples bakes language-following into weights."
        ),
    ),
    Category(
        name="Empty-Result Behavior",
        short="F",
        prompt_ids=[f"P{i:03d}" for i in range(113, 131)],
        failure_mode="Fabricates data when API returns empty results",
        f1_threshold=0.80,
        method="DPO",
        justification=(
            "DPO is required. Empty-result fabrication is a preference failure — the model prefers "
            "plausible-sounding answers over honest empty-result acknowledgement. "
            "DPO trains the correct preference explicitly."
        ),
    ),
    Category(
        name="Token Pressure (Both-intent)",
        short="G",
        prompt_ids=[f"P{i:03d}" for i in range(131, 149)],
        failure_mode="Quality degrades when context is near-full (RAG + live data combined)",
        f1_threshold=0.70,
        method="QLoRA",
        justification=(
            "QLoRA is required. Token budget analysis shows both-intent queries use ~1,700–1,800 "
            "of 2,048 tokens, leaving no room for few-shot examples. "
            "QLoRA bakes the synthesis behavior into weights at zero inference cost."
        ),
    ),
    Category(
        name="Complex Parameter Generation",
        short="H",
        prompt_ids=[f"P{i:03d}" for i in range(149, 175)],
        failure_mode="Wrong JQL / status filter / project parameter generated",
        f1_threshold=0.70,
        method="GRPO",
        justification=(
            "GRPO is required. The deterministic sentry rule system (~1,230 lines) cannot generalize "
            "to multi-condition queries. GRPO teaches parameter generation through a reward function, "
            "enabling correct JQL / filter construction without hand-written rules."
        ),
    ),
    Category(
        name="Cross-Source Synthesis",
        short="I",
        prompt_ids=[f"P{i:03d}" for i in range(175, 201)],
        failure_mode="Uses only one source; fails to combine RAG docs with live ticket data",
        f1_threshold=0.75,
        method="QLoRA",
        justification=(
            "QLoRA is required. The model treats RAG data and live sentry data as separate outputs, "
            "rarely synthesizing them into a unified answer. "
            "QLoRA on synthesis examples bakes the combined-source behavior into weights."
        ),
    ),
]

TOTAL_PROMPTS = 200


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CategoryMetrics:
    category:  Category
    results:   Dict[str, int]   # prompt_id → 1 (PASS) or 0 (FAIL)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def tp(self) -> int:
        """True Positives = PASS responses (correct behavior)."""
        return sum(v for v in self.results.values() if v == 1)

    @property
    def fn(self) -> int:
        """False Negatives = FAIL responses (missed correct behavior)."""
        return sum(1 for v in self.results.values() if v == 0)

    @property
    def fp(self) -> int:
        """
        False Positives in the benchmark context:
        prompts outside this category that were incorrectly answered
        in a way that would count against this category.
        For simplicity this is 0 in the per-category model
        (each prompt belongs to exactly one category).
        """
        return 0

    @property
    def precision(self) -> float:
        """Fraction of PASS responses that are genuine (TP / (TP + FP))."""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Fraction of answerable prompts correctly answered (TP / (TP + FN))."""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def triggered(self) -> bool:
        return self.f1 < self.category.f1_threshold

    @property
    def coverage(self) -> float:
        """Fraction of prompts in this category that have results."""
        return len(self.results) / len(self.category.prompt_ids)


# ──────────────────────────────────────────────────────────────────────────────
# LOADING RESULTS
# ──────────────────────────────────────────────────────────────────────────────

def parse_csv(path: str) -> Dict[str, int]:
    results = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("prompt_id", row.get("id", "")).strip().upper()
            val = row.get("result", row.get("pass", "")).strip()
            if pid and val:
                results[pid] = 1 if val in ("1", "PASS", "pass", "true", "True") else 0
    return results


def parse_json(path: str) -> Dict[str, int]:
    with open(path) as f:
        data = json.load(f)
    results = {}
    if isinstance(data, list):
        for item in data:
            pid = item.get("id", item.get("prompt_id", "")).strip().upper()
            val = item.get("result", item.get("pass"))
            if pid is not None and val is not None:
                results[pid] = 1 if val in (1, True, "PASS", "pass") else 0
    elif isinstance(data, dict):
        for pid, val in data.items():
            results[pid.upper()] = 1 if val in (1, True, "PASS", "pass") else 0
    return results


def assign_to_categories(raw: Dict[str, int]) -> List[CategoryMetrics]:
    metrics = []
    for cat in CATEGORIES:
        cat_results = {pid: raw[pid] for pid in cat.prompt_ids if pid in raw}
        metrics.append(CategoryMetrics(category=cat, results=cat_results))
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# REPORTING
# ──────────────────────────────────────────────────────────────────────────────

PASS_EMOJI   = "✓"
FAIL_EMOJI   = "✗"
TRIGGERED    = "⚠  TRIGGERED"
NOT_TRIGGERED = "   ok"

COLS = {
    "cat":     (5,  "Cat"),
    "name":    (30, "Category"),
    "prompts": (8,  "n"),
    "tp":      (6,  "PASS"),
    "fn":      (6,  "FAIL"),
    "prec":    (10, "Precision"),
    "rec":     (9,  "Recall"),
    "f1":      (8,  "F1"),
    "thr":     (8,  "Thr"),
    "status":  (20, "Status"),
    "method":  (8,  "Method"),
}


def fmt(val, width, align="<"):
    return f"{str(val):{align}{width}}"


def divider(widths, char="─"):
    return "┼".join(char * (w + 2) for w in widths)


def header_row():
    cells = [fmt(v, list(COLS.values())[i][0]) for i, (k, v) in enumerate(COLS.items())]
    return "  ".join(cells)


def print_report(metrics_list: List[CategoryMetrics], raw_total: int) -> None:
    overall_pass = sum(m.tp for m in metrics_list)
    overall_fail = sum(m.fn for m in metrics_list)
    overall_total = overall_pass + overall_fail

    triggered_cats = [m for m in metrics_list if m.triggered]

    print()
    print("╔══════════════════════════════════════════════════════════════════════════════════╗")
    print("║            BENCHMARK EVALUATION RESULTS — STRATEGY DECISION MATRIX             ║")
    print("╚══════════════════════════════════════════════════════════════════════════════════╝")
    print(f"\n  Prompts evaluated: {raw_total} / {TOTAL_PROMPTS}   "
          f"PASS: {overall_pass}   FAIL: {overall_fail}   "
          f"Overall: {overall_pass/overall_total*100:.1f}%" if overall_total > 0 else "")
    print()

    # ── Per-category table ────────────────────────────────────────────
    header = (
        f"  {'Cat':<5}  {'Category':<28}  {'n':>4}  "
        f"{'PASS':>5}  {'FAIL':>5}  "
        f"{'Prec':>7}  {'Recall':>7}  {'F1':>7}  "
        f"{'Thr':>5}  {'Status':<20}  {'Method':<7}"
    )
    sep = "  " + "─" * (len(header) - 2)
    print(sep)
    print(header)
    print(sep)

    for m in metrics_list:
        status_str = TRIGGERED if m.triggered else NOT_TRIGGERED
        cov_str = "" if m.coverage >= 0.99 else f" ({m.coverage:.0%})"
        print(
            f"  {m.category.short:<5}  {m.category.name:<28}  "
            f"{m.total:>4}{cov_str}  "
            f"{m.tp:>5}  {m.fn:>5}  "
            f"{m.precision:>7.3f}  {m.recall:>7.3f}  {m.f1:>7.3f}  "
            f"{m.category.f1_threshold:>5.2f}  {status_str:<20}  {m.category.method:<7}"
        )
    print(sep)
    print()

    # ── Decision summary ──────────────────────────────────────────────
    if not triggered_cats:
        print("  ┌─────────────────────────────────────────────────────────────────────────────┐")
        print("  │  ✓  F1 ≥ threshold across ALL categories — NO fine-tuning required.        │")
        print("  │                                                                             │")
        print("  │  Your supervisor is correct: prompting is sufficient.                      │")
        print("  │  The project contribution is the evaluation methodology itself:             │")
        print("  │    • The two-mode (RAG + Sentry) hybrid architecture                       │")
        print("  │    • The bias-controlled 200-prompt dataset                                │")
        print("  │    • This closed-loop benchmark → strategy framework                       │")
        print("  │                                                                             │")
        print("  │  This is a valid and publishable contribution without fine-tuning.         │")
        print("  └─────────────────────────────────────────────────────────────────────────────┘")
        print()
        return

    # Group triggered methods
    dpo_cats  = [m for m in triggered_cats if m.category.method == "DPO"]
    qlora_cats = [m for m in triggered_cats if m.category.method == "QLoRA"]
    grpo_cats  = [m for m in triggered_cats if m.category.method == "GRPO"]

    print("  ┌─────────────────────────────────────────────────────────────────────────────┐")
    print("  │  ⚠  FINE-TUNING REQUIRED in the following categories:                      │")
    print("  └─────────────────────────────────────────────────────────────────────────────┘")
    print()

    if dpo_cats:
        print("  ▶ DPO  (Direct Preference Optimisation)")
        for m in dpo_cats:
            delta = m.category.f1_threshold - m.f1
            print(f"    • Cat {m.category.short} — {m.category.name}:  "
                  f"F1 = {m.f1:.3f}  (threshold {m.category.f1_threshold:.2f}, gap {delta:.3f})")
        print(f"    Argument: {dpo_cats[0].category.justification}")
        print()

    if qlora_cats:
        print("  ▶ QLoRA  (Quantised Low-Rank Adaptation)")
        for m in qlora_cats:
            delta = m.category.f1_threshold - m.f1
            print(f"    • Cat {m.category.short} — {m.category.name}:  "
                  f"F1 = {m.f1:.3f}  (threshold {m.category.f1_threshold:.2f}, gap {delta:.3f})")
        print(f"    Argument: {qlora_cats[0].category.justification}")
        print()

    if grpo_cats:
        print("  ▶ GRPO  (Group Relative Policy Optimisation)")
        for m in grpo_cats:
            delta = m.category.f1_threshold - m.f1
            print(f"    • Cat {m.category.short} — {m.category.name}:  "
                  f"F1 = {m.f1:.3f}  (threshold {m.category.f1_threshold:.2f}, gap {delta:.3f})")
        print(f"    Argument: {grpo_cats[0].category.justification}")
        print()

    # ── Priority recommendation ────────────────────────────────────────
    print("  ─────────────────────────────────────────────────────────────────────────────")
    print("  PRIORITY ORDER  (highest delta first):")
    sorted_triggered = sorted(triggered_cats, key=lambda m: m.category.f1_threshold - m.f1, reverse=True)
    for rank, m in enumerate(sorted_triggered, 1):
        delta = m.category.f1_threshold - m.f1
        print(f"    {rank}. Cat {m.category.short} — {m.category.name:<28}  "
              f"method={m.category.method}  gap={delta:.3f}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# EXPORT HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def export_csv(metrics_list: List[CategoryMetrics], path: str) -> None:
    """Write per-category metrics to CSV for downstream analysis."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "category_short", "category_name", "n", "pass", "fail",
            "precision", "recall", "f1", "threshold", "triggered", "method"
        ])
        for m in metrics_list:
            writer.writerow([
                m.category.short, m.category.name, m.total,
                m.tp, m.fn,
                f"{m.precision:.4f}", f"{m.recall:.4f}", f"{m.f1:.4f}",
                m.category.f1_threshold, int(m.triggered), m.category.method,
            ])
    print(f"  Metrics exported to: {path}")


def print_template() -> None:
    """Print an empty CSV template for manual result entry."""
    print("prompt_id,result")
    for cat in CATEGORIES:
        for pid in cat.prompt_ids:
            print(f"{pid},")   # leave result blank for user to fill in


# ──────────────────────────────────────────────────────────────────────────────
# DEMO  (shows what the output looks like with realistic simulated results)
# ──────────────────────────────────────────────────────────────────────────────

def build_demo_results() -> Dict[str, int]:
    """
    Simulated benchmark results representing a reasonable baseline
    for Qwen3:8b on a RAG + sentry system WITHOUT fine-tuning.

    Pass rates mirror the 'Expected pre-training baseline' in the
    decision framework:
      A (Refusal)      ~60% pass  → F1 ≈ 0.75  (below 0.85 → DPO triggered)
      B (Fabrication)  ~72% pass  → F1 ≈ 0.84  (above 0.80 → ok)
      C (Routing)      ~88% pass  → F1 ≈ 0.94  (above 0.90 → ok)
      D (Format)       ~78% pass  → F1 ≈ 0.88  (above 0.80 → ok)
      E (French)       ~70% pass  → F1 ≈ 0.82  (below 0.85 → QLoRA triggered)
      F (Empty-result) ~60% pass  → F1 ≈ 0.75  (below 0.80 → DPO triggered)
      G (Token press.) ~65% pass  → F1 ≈ 0.79  (above 0.70 → ok)
      H (Complex par.) ~60% pass  → F1 ≈ 0.75  (above 0.70 → ok)
      I (Synthesis)    ~65% pass  → F1 ≈ 0.79  (above 0.75 → ok)
    """
    import random
    random.seed(42)

    pass_rates = {
        "A": 0.60, "B": 0.72, "C": 0.88, "D": 0.78,
        "E": 0.70, "F": 0.60, "G": 0.65, "H": 0.60, "I": 0.65,
    }
    results = {}
    for cat in CATEGORIES:
        rate = pass_rates[cat.short]
        for pid in cat.prompt_ids:
            results[pid] = 1 if random.random() < rate else 0
    return results


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute F1 / Precision / Recall from benchmark results and recommend fine-tuning strategy"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--demo",     action="store_true", help="Run with built-in demo results")
    group.add_argument("--csv",      metavar="FILE",      help="Load results from CSV (columns: prompt_id, result)")
    group.add_argument("--json",     metavar="FILE",      help="Load results from JSON")
    group.add_argument("--template", action="store_true", help="Print empty CSV template to stdout")
    parser.add_argument("--export",  metavar="FILE",      help="Export per-category metrics CSV to this path")
    args = parser.parse_args()

    if args.template:
        print_template()
        return

    if args.demo:
        print("\n  [DEMO MODE — simulated Qwen3:8b baseline results (no fine-tuning)]")
        raw = build_demo_results()
    elif args.csv:
        raw = parse_csv(args.csv)
        print(f"\n  Loaded {len(raw)} results from {args.csv}")
    elif args.json:
        raw = parse_json(args.json)
        print(f"\n  Loaded {len(raw)} results from {args.json}")
    else:
        parser.print_help()
        print("\n  Tip: run with --demo to see a worked example.")
        return

    metrics_list = assign_to_categories(raw)
    print_report(metrics_list, raw_total=len(raw))

    if args.export:
        export_csv(metrics_list, args.export)


if __name__ == "__main__":
    main()
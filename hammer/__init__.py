"""
Hammer — Evaluation & Training Data Pipeline  (v1.1 — Neo4j edition)
─────────────────────────────────────────────────────────────────────
Phase 3 of the inference plane. Runs asynchronously alongside production.
Never touches LangGraph / inference pipeline.

Components:
  evaluator.py      — RAGAS scorer + custom metrics → weighted_score
  validator.py      — Deterministic sentry answer checker (no LLM)
  dataset_builder.py — QLoRA / DPO / GRPO dataset exporter
  benchmark.py      — Adapter gate (compares new adapter vs baseline)
  run_hammer.py     — CLI orchestrator

Context retrieval backend: Neo4j hybrid_search()  (replaces Qdrant)
  Tier 1 — context_snippets stored in chat_history at inference time
  Tier 2 — Neo4j re-query fallback (vector + multi-hop graph traversal)

Scoring formula (v1.1):
  faithfulness        35%   RAGAS (rag/both) | validator_score (sentries)
  answer_relevance    35%   RAGAS (rag/both) | entity_match_relevance (sentries)
  temporal_consistency 10%  Custom, pure Python — all intents
  code_correctness    20%   Python AST — only when code blocks present

New in v1.1:
  graph_coverage      audit metric — fraction of sources from graph expansion
  entity_match_relevance — replaces hardcoded 0.50 for sentries
  per-intent gate in benchmark — sentries faithfulness >= 0.75 required
  token_f1 — benchmark-only F1 reference metric

Training signals written to chat_history.evaluation.training_signal:
  "qlora_positive"   weighted_score >= 0.80
  "dpo_rejected"     weighted_score <  0.50
  "neutral"          everything else
"""

__version__ = "1.1.0"

# AI Assistant — Hybrid Graph-RAG + Live-API Agent with a Self-Improving Eval Loop

A production-style, fully local enterprise assistant that answers engineering questions from **two sources at once**: a Neo4j knowledge graph built from Jira/Confluence/GitLab exports (vector + multi-hop graph retrieval), and the **live** GitLab/Jira/Confluence APIs. A LangGraph agent routes each question, a local LLM (Ollama) synthesizes the answer, and an evaluation pipeline ("Hammer") scores every response, mines training data, and gates fine-tuned adapter deployments.

Everything runs locally: Neo4j, MongoDB, Ollama, and QLoRA/DPO fine-tuning sized for a 6 GB consumer GPU.

## Architecture

```
                          ┌─────────────────┐
        user question ───►│ intent classifier│  rag | sentries | both
                          └────────┬────────┘
              ┌────────────────────┼────────────────────┐
              ▼                                         ▼
   ┌─────────────────────┐                 ┌─────────────────────────┐
   │  RAG (Neo4j)        │                 │  Sentries (live APIs)   │
   │  vector ANN          │                 │  GitLab · Jira ·        │
   │  + multi-hop graph   │                 │  Confluence, parallel   │
   │  + cross-encoder     │                 │  dispatch, fan-out caps │
   │    rerank            │                 └───────────┬─────────────┘
   └──────────┬───────────┘                             │
              └──────────────────┬──────────────────────┘
                                 ▼
                       ┌──────────────────┐
                       │  Weaver (Ollama) │  budgeted context, streaming
                       └────────┬─────────┘
                                ▼
                          final answer
                                │
                                ▼
                  ┌───────────────────────────┐
                  │  Hammer (async eval loop) │  RAGAS + deterministic
                  │  grade → dataset → gate   │  validator → QLoRA/DPO
                  └───────────────────────────┘
```

## Highlights

- **Hybrid retrieval**: embedding ANN over chunk vectors stored in Neo4j, expanded through typed Jira link edges (BLOCKS, RELATES_TO, CHILD_OF, …) and the Confluence page hierarchy, then reranked with a cross-encoder. Questions naming a ticket key bypass search entirely with a direct indexed graph fetch.
- **Deterministic routing**: a keyword/regex intent classifier (no LLM latency on the hot path) with an optional LLM fallback; ~30 routing rules map natural language to concrete API calls (JQL, CQL, GitLab REST) with conversation context carry-over and per-project fan-out caps.
- **Context-budgeted generation**: every prompt section (knowledge base, live data, history) has an enforced budget so the model window can never silently overflow.
- **Self-improving loop**: every answer is scored (RAGAS faithfulness/relevance where applicable, a zero-LLM validator for live-data answers), graded GOLD/SILVER/BRONZE/FAILED with orthogonal failure tags, exported to QLoRA/DPO/GRPO datasets, and fine-tuned adapters must pass a benchmark gate (score delta + per-intent faithfulness floors) before promotion via MLflow.
- **Bilingual**: English and French queries throughout the routing and prompting layers.

## Why it matters (business value)

Engineering teams lose hours per week re-checking Jira, Confluence and GitLab by hand because they don't trust assistant answers. This project attacks that trust problem end to end, with measured results rather than claims:

- **Answers users can act on.** Cross-encoder reranking, direct ticket lookup, and metadata-grounded citations (status, priority, assignee, URL) mean answers are verifiable at a glance — the difference between a demo and a tool people rely on.
- **No silent failures on high-stakes questions.** Multi-source prompts used to overflow the model window (~12.6k tokens vs a 10.2k budget), silently dropping the instructions — exactly on the complex questions where stakes are highest. Context is now budgeted by design (verified: same scenario fits at 5.7k tokens).
- **Costs that don't scale with company size.** Broad queries used to fan out one API call per repository; caps cut a verified 40→12 (GitLab) and 30→10 (Jira) calls, bounding both latency and rate-limit exposure as the org grows. Per-query overhead (GC/GPU flushes, redundant health checks, sync DB writes) was removed from the hot path.
- **Fine-tuning spend that actually pays off.** The evaluation layer was provably mislabeling correct answers as hallucinations, poisoning the GOLD/DPO training pools, while exported training prompts had drifted from production format — and the adapter-promotion path could never ship. All fixed and verified: every future GPU-hour trains on clean labels, in the production prompt format, with a working path to deployment behind a quality gate.

## Scale

Built and tested against a real enterprise corpus, not a toy dataset:

| Metric | Value |
|---|---|
| Embedded corpus | **~24 GB** of pre-embedded chunks (20 GB Jira + 3.7 GB Confluence JSONL) |
| Jira issues represented | ~424,000 across dozens of projects |
| Chunk nodes in Neo4j (768-dim vectors) | _run `python -m rag.neo4j_search stats` →_ `N` |
| Graph relationships (PART_OF, BLOCKS, RELATES_TO, CHILD_OF, …) | `N` |
| Typed Jira link relationship types modeled | 20+ (BLOCKS, CAUSES, DUPLICATES, IMPLEMENTS, SPLIT_FROM, …) |
| Production answers scored by the eval loop | `N` (`python -m hammer.run_hammer status`) |
| Fine-tuning hardware | single RTX 3050 (6 GB) — QLoRA 4-bit + gradient checkpointing |

## Screenshots

| Inference (CLI, per-node timing) | Evaluation loop |
|---|---|
| ![Inference](assets/inference_cli.png) | ![Hammer status](assets/hammer_status.png) |

| Knowledge graph | QLoRA training | MLflow runs |
|---|---|---|
| ![Graph](assets/graph_stats.png) | ![Training](assets/training_run.png) | ![MLflow](assets/mlflow_runs.png) |

## Engineering rigor

The full stack went through three documented optimization rounds with A/B verification before reaching this state — including fixes verified by automated checks: a prompt-overflow bug (12.6k tokens silently truncated against a 10.2k window → now budgeted), API fan-out cut 40→12 calls on broad queries, evaluation mislabeling that poisoned training data (dead query-echo exemptions, narrow key regexes), training prompts that had drifted from inference prompts, and a model-promotion loop that could never ship. The complete diagnoses live in:

- [`rag/DIAGNOSIS.md`](rag/DIAGNOSIS.md) — retrieval & generation layer
- [`AGENT_SENTRIES_DIAGNOSIS.md`](AGENT_SENTRIES_DIAGNOSIS.md) — agent, routing, weaver
- [`HAMMER_TRAINING_DIAGNOSIS.md`](HAMMER_TRAINING_DIAGNOSIS.md) — evaluation & training pipeline

## Stack

Python · Neo4j 5 (vector index + Cypher) · MongoDB · Ollama (Qwen3) · sentence-transformers (arctic-embed-m + cross-encoder rerank) · LangGraph · RAGAS · MLflow · TRL/PEFT (QLoRA, DPO) · FastAPI · React

## Repository layout

| Path | What it is |
|---|---|
| `rag/` | Neo4j import (graph schema + vector index), hybrid search engine, RAG generator |
| `agent/` | LangGraph intent agent, intent classifier, answer weaver |
| `sentries/` | Live-API layer: dispatcher, GitLab/Jira/Confluence clients, NL→API router |
| `hammer/` | Evaluator, deterministic validator, dataset builder, deployment benchmark |
| `training/` | QLoRA/DPO pipeline with MLflow registry promotion |
| `api/`, `frontend/` | FastAPI backend + React UI |
| `pipeline/` | Source-system extraction/embedding pipeline |

## Run with Docker (fastest)

```bash
cp .env.example .env            # set NEO4J_PASSWORD + your API tokens
ollama serve && ollama pull qwen3:8b   # Ollama runs on the host (GPU)
docker compose up -d --build
```

That starts **Neo4j 5 + APOC** (browser: `:7474`), **MongoDB**, the **FastAPI backend** (`:8000`), and the **React frontend** (`:3000`). On CPU-only machines, remove the `deploy:` GPU block from `docker-compose.yml`. Then load your data: `docker compose exec backend python -m rag.neo4j_import`.

## Quickstart (bare metal)

```bash
# 1. Infrastructure
docker compose up -d neo4j mongo
ollama serve && ollama pull qwen3:8b

# 2. Python env
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env            # fill in credentials

# 4. Build the knowledge graph (from your embedded chunk exports)
python -m rag.neo4j_import

# 5. Talk to it
python agent/intent_agent.py    # interactive CLI (/timing, /impl, /history)

# 6. Run the evaluation loop
python -m hammer.run_hammer status
python -m hammer.run_hammer score --limit 20 --dry-run
```

All tunables (retrieval depth, rerank pool, context budgets, fan-out caps, eval thresholds, training hyperparameters) are environment variables — see [`.env.example`](.env.example).

## Evaluation → training loop

```bash
python -m hammer.run_hammer score            # score new chat turns
python -m hammer.run_hammer dataset all      # export QLoRA / DPO / GRPO sets
python training/pipeline.py check-compat     # adapter ↔ serving model guard
python training/pipeline.py prepare && python training/pipeline.py train --method qlora
python -m hammer.run_hammer benchmark datasets/eval_set.jsonl --version my-lora-v1
python training/pipeline.py promote <mlflow_run_id>   # only if the gate passes
```

## Notes

- Example data, vector stores, model checkpoints, and generated reports are intentionally not part of the repository (see `.gitignore`); bring your own Jira/Confluence/GitLab exports.
- Project names and ticket keys appearing in code comments and docs are illustrative.

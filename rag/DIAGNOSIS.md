# RAG Optimization — Diagnosis & Changes

**Scope:** retrieval + generation, balanced for precision/recall, answer faithfulness, and latency.
**Method:** new `*_test.py` files alongside the originals. Nothing original was modified, so reverting is "delete the test files" (or just keep importing the originals).

## What the system is

Precomputed `snowflake-arctic-embed-m` (768‑dim) chunk embeddings live in Neo4j as `:Chunk` nodes wired to `:JiraTicket` / `:ConfluencePage` / `:GitLabDoc` with typed link edges. `neo4j_search.py` does vector ANN → multi‑hop graph expansion → lexical re‑rank. `rag_generator.py` builds context and generates with a local Ollama model (`qwen3`). The `*_flat.py` files are your earlier single‑hop versions; `neo4j_rag.py` is an alternate chat front‑end.

The active pair is **`rag_generator.py` → `neo4j_search.py`**. Those are what I optimized.

## Diagnosis

**Retrieval quality.** The final ranking was a hand‑tuned `0.7·cosine + 0.3·keyword` blend. The keyword term used whitespace splitting with substring matching (`term in text`), so `"is"` matches inside `"this"`, and there is no stopword removal — a weak, noisy signal. There was no true reranker, which is the largest available precision lever for this kind of corpus.

**Seed extraction was fragile (recall bug).** Graph expansion derived Jira/Confluence seed IDs by regex‑stripping `parent_document_id` / `chunk_id` strings. The code's own comments note `parent_document_id` is frequently `NULL`, so seeds were silently missed and the graph half of "hybrid" under‑fired.

**Metadata was lost for vector hits (faithfulness).** `vector_search` never returned `assignee` / `priority` / `issue_key` (those live on the `:JiraTicket` node, not the `:Chunk`). The normaliser read `hit.get("assignee")`, which was therefore **always empty** for vector results — so the model couldn't reliably answer "who's the assignee / what's the priority" without the ticket happening to arrive via graph expansion.

**Context was over‑truncated.** Every source was clipped to 700 chars regardless of `NUM_CTX=8192`, leaving most of the context window unused and cutting off answers.

**No ticket‑aware routing.** A question that literally names a ticket ("What is **TM‑18399** about…") still went through embed → ANN → graph and hoped the ticket surfaced, instead of fetching it directly by its unique key.

### Latency culprits (you asked me to flag these)

1. **3–4 separate Ollama `/api/tags` HTTP calls at startup** — `_check_ollama` + `_resolve_model_name` + `_check_model` (+ `_list_models` on failure), each with a 5 s timeout. Collapsed to **one** call.
2. **Over‑fetch‑then‑discard on filtered queries (the big one).** `rag_generator.retrieve()` over‑fetched `top_k*5` and **graph‑expanded all of them**, then filtered by project *in Python*. With `top_k=5` that's ~50 vector hits and up to 50 multi‑hop expansions, most thrown away. Fixed by pushing the project/source filter **into the Cypher vector query** and expanding only surviving seeds.
3. **Fragile seed parsing** caused missed expansions (above) — not latency directly, but it makes the graph step do work that returns nothing.
4. Minor: three Neo4j sessions per query (vector + jira_expand + conf_expand). Left as‑is; flagged for a future single‑session pass.

**Honest tradeoff:** the cross‑encoder reranker *adds* per‑query cost. I bounded it (fixed candidate pool, parent‑doc dedup shrinks the batch, single shared embedding) and offset it with the fixes above. In the **full** pipeline, LLM generation dominates wall‑clock, so rerank is a small fraction; in **retrieval‑only** mode it's the main new cost. Use `--retrieval-only` in the harness to see it isolated. Net retrieval latency on *filtered* queries should drop; on unfiltered queries it rises modestly but precision improves — measure with the harness and tune `RERANK_POOL` / `VEC_CANDIDATES` down if you want it cheaper.

## What changed

| Area | Original | Optimized (`*_test.py`) |
|---|---|---|
| Final ranking | `0.7·cosine + 0.3·keyword` substring | **Cross‑encoder rerank** (`RERANKER_MODEL`), lexical only as fallback |
| Seed extraction | regex on `parent_document_id` (often NULL) | **exact** `jt.issue_key` / `cp.page_id` via `OPTIONAL MATCH` |
| Vector metadata | no assignee/priority/key | enriched from parent ticket/page |
| Project filter | Python over‑fetch `top_k*5` + post‑filter | **pushed into Cypher**; expand only real seeds |
| Graph score | flat `0.6` | **distance‑decayed** prior + rerank |
| Diversity | none | **parent‑doc cap** (`PARENT_DOC_CAP`) |
| Named tickets | none | **direct indexed fetch** fast path |
| Context budget | hard 700 chars | **adaptive** from `NUM_CTX` |
| Ollama startup | 3–4 HTTP calls | **1** call |
| Observability | none | `last_timings` / `result["timings"]` per stage |

Files added (originals untouched):

- `rag/neo4j_search_test.py` — optimized retrieval engine (same `Neo4jSearch` API).
- `rag/rag_generator_test.py` — optimized generator (same `RAGGenerator` API), imports the engine above.
- `rag/rag_ab_test.py` — A/B harness.

## New `.env` keys (all optional; sensible defaults baked in)

```
RERANKER_MODEL      cross-encoder/ms-marco-MiniLM-L-6-v2   # ~80MB, fast, English-strong
RERANK_POOL         50      # max candidates sent to the reranker
VEC_CANDIDATES      40      # vector pool fetched before rerank
PARENT_DOC_CAP      3       # max chunks kept per parent document
GRAPH_MIN_SLOTS     0       # 0 = auto (top_k//4) reserved graph slots
GRAPH_MIN_RERANK    0.0     # drop reserved graph hits below this rerank score
RERANK_TEXT_CHARS   800     # passage length sent to reranker
GRAPH_DECAY         0.85    # per-hop prior decay
CONTEXT_CHARS_PER_SOURCE  0 # 0 = auto from NUM_CTX; else a fixed per-source cap
```

The reranker uses the `CrossEncoder` class already bundled with `sentence-transformers` — **no new pip package**, only a one‑time model download on first run. For higher quality at more cost, set `RERANKER_MODEL=BAAI/bge-reranker-base` (or `…-v2-m3` for multilingual).

## How to test

From the project root (the folder containing `rag/`), with Neo4j up (and Ollama up for full mode):

```bash
# Fast: compare retrieval quality + latency only, no LLM
python -m rag.rag_ab_test --retrieval-only

# Full pipeline, side-by-side answers + latency
python -m rag.rag_ab_test

# Tune knobs / supply your own queries (one per line)
python -m rag.rag_ab_test --top-k 6 --hops 2 --queries myqueries.txt
```

You can also drive the optimized pieces directly:

```bash
python -m rag.neo4j_search_test hybrid "what blocks the ExampleProduct release"
python -m rag.rag_generator_test "What is PROJ-18399 about and what do the docs say about CC errors?"
```

Read the harness output for: latency delta (per stage), source overlap, and the two answers. Expect **low source overlap** — that's the point; judge whether the *new* sources are more on‑topic.

## How to revert

Nothing to undo — the originals were never touched. To go back, just keep importing `rag_generator` / `neo4j_search`. To adopt the optimized path, point `intent_agent.py` at `rag_generator_test` (same class name and method signatures), or rename the files when you're satisfied. You can delete the three `*_test.py` files at any time.

## Also found (not changed — out of scope)

`neo4j_rag.py` (the alternate chat front‑end) has a **latent bug**: its `GRAPH_EXPAND` / `CONF_EXPAND` Cypher uses `*1..$hops` with `$hops` as a **parameter**. Neo4j does not allow parameters inside a variable‑length relationship pattern, so any call that exercises graph expansion there will throw at runtime. `neo4j_search.py` (and my `_test` version) correctly bake the hop count in as a literal. If you use `neo4j_rag.py`, say the word and I'll fix it the same way.

---
*Validation note:* I statically reviewed the new files (logic + f‑string/Cypher brace escaping verified by inspection), but the sandbox here wouldn't boot this session, so I could **not** run `py_compile` or the live pipeline (no access to your Neo4j, Ollama, or the model downloads either way). Before wiring anything in, run the one‑liner below; then use the A/B harness to confirm quality/latency on your machine.

```bash
python -m py_compile rag/neo4j_search_test.py rag/rag_generator_test.py rag/rag_ab_test.py && echo OK
```

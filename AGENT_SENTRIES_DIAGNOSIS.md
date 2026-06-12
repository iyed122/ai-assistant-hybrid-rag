# Agent + Sentries Optimization — Diagnosis & Changes

Companion to `rag/DIAGNOSIS.md` (which covers the retrieval/generation layer). Same method: every change lives in a new `*_test.py` file; **no original file was modified**. Revert = keep using the originals, or delete the test files.

## What this layer is

`agent/intent_agent.py` is a LangGraph pipeline: classify intent (`rag` / `sentries` / `both`) → run RAG retrieval and/or parallel live-API "sentries" (GitLab/Jira/Confluence via `sentries/`) → `gpu_clear` → `weaver_node` synthesizes the final answer with Ollama. `sentries/chatbot_sentries.py` holds the regex router (`route_query`), project registry, and count shortcut. The `*_1.py` files are inactive older versions; nothing imports them.

## Bugs found (the interesting ones)

**1. `weaver_node.py` defines `_build_prompt` and `_format_history` TWICE.** Python keeps whichever is defined last, so the first pair is dead code — including its careful 300-chars-per-turn history cap. The active `_format_history` injects up to ~800 raw chars per stored turn (×3 turns) into every prompt, and the two `_build_prompt`s disagree on the RAG cap (3,000 vs 6,000 — the 6,000 one is active). Latent landmine: editing the first copy does nothing.

**2. The sentries context has NO size budget.** The RAG side is capped (6,000 chars) but `_format_sentries_context` output is unbounded: a fan-out of 2+ calls × 30 items × 400-char descriptions, or one 4,000-char MR diff per item, easily exceeds `num_ctx=10240`. Ollama then silently drops the *oldest* tokens — i.e. the instruction header and knowledge-base block at the top of the prompt — which is precisely the "LLM never sees X" failure the file's own comments fight elsewhere. This is both a faithfulness and a latency problem (wasted prompt tokens are processed before generation starts).

**3. `intent_agent.rag_node` computes its score filter and never applies it.** `filtered = [s for s in sources if score >= 0.20]` is built, logged… and then `sources` (unfiltered) is what goes into the result. The threshold is a no-op; noise sources flow into the weaver prompt. Also, the intended source cap discussed in the comments was never re-applied — all 8 retrieved sources pass through.

**4. `gpu_clear_node` runs `gc.collect()` + `torch.cuda.empty_cache()` on EVERY query** — in the hot path between retrieval and generation. A full GC pass with a loaded embedding-model heap costs tens to hundreds of ms; `empty_cache()` forces a device sync and releases cached VRAM blocks that the next embed must re-allocate. Ollama runs in its own process, so this frees nothing it can use per-turn. This is pure unnecessary latency, every single turn.

**5. Uncapped fan-outs in `route_query`.** Broad queries ("show all merge requests", "jira tickets", branches with no project) emit **one API call per project** — `personal_projects()` and `all_jira_keys()` are uncapped (only the WIP rule caps Jira at 10). With N projects that's N HTTP calls through a 5-calls/sec rate limiter and a 6-thread pool, then a flood of items into the weaver context. Latency scales with your project count, not your question.

**6. Inconsistent Jira-key regexes.** Classifier `{2,5}`, agent `{2,8}`, sentries `{2,8}`. A long-prefix key like `LONGKEY-12355` (6 chars) gets no sentry boost and no "key+docs → both" override in the classifier, so the live ticket can be silently skipped. Unified at `{2,9}`.

**7. Smaller items.** `RateLimiter.wait()` is not thread-safe under the ThreadPool (bursts can exceed the API limit → 429 retries → latency spikes). Hardcoded demo project names (`auth-service|ecommerce|notification-service`) sit in the classifier's sentry signals; hardcoded personal namespaces (`legacy-group-a`, `legacy-group-b`, `primary-namespace`) in the registry; `_infer_project` hardcodes `gitlab.com` so it never matches self-hosted GitLab; the end-of-turn MongoDB `insert_one` is synchronous on the hot path.

## What changed, where

| File (new) | Fixes |
|---|---|
| `agent/weaver_node_test.py` | one canonical `_build_prompt`/`_format_history` (#1); history cap restored; **new sentries cap** (#2); env-tunable budgets |
| `agent/intent_agent_test.py` | score filter actually applied + source cap (#3); gpu_clear throttled, default off (#4); key regex `{2,9}` (#6); GitLab host from env (#7); Mongo persist in daemon thread (#7); auto-wires all `_test` modules with fallback to originals |
| `agent/intent_classifier_test.py` | demo project names removed, key regex `{2,9}` (#6, #7); LLM fallback trimmed (24 tokens / 10 s) |
| `sentries/chatbot_sentries_test.py` | thin overlay (routing logic untouched): fan-out caps (#5), env-driven namespace exclusions, **opt-in** thread-safe RateLimiter (#7) |
| `agent/agent_ab_test.py` | A/B harness — two offline modes + live mode |

The test agent prefers `rag_generator_test` (cross-encoder reranked retrieval + direct ticket fast-path from the previous round) and falls back to `rag_generator` if you delete it.

## New env keys (all optional)

```
# intent_agent_test
RAG_MIN_SCORE          0.20    # 0 disables the filter
RAG_MAX_SOURCES        6
RAG_TOP_K              8
GPU_CLEAR_EVERY_N      0       # 0 = never run gc/CUDA clear; N = every Nth turn
GITLAB_URL                     # used to sniff project URLs (self-hosted ok)

# weaver_node_test
WEAVER_RAG_CAP         6000    # same as the active original
WEAVER_SENTRIES_CAP    12000   # NEW — was unbounded
WEAVER_HISTORY_CHAR_CAP 300    # per stored turn
WEAVER_NUM_CTX         10240

# chatbot_sentries_test
SENTRY_FANOUT_MAX_GITLAB   12      # 0 disables cap
SENTRY_FANOUT_MAX_JIRA     10      # 0 disables cap
SENTRY_EXCLUDED_NAMESPACES legacy-group-a,legacy-group-b
AGENT_TEST_THREADSAFE_RL   0       # 1 = patch RateLimiter (affects whole process)
```

## How to test

Offline (free, instant — no Ollama/Neo4j/APIs needed):

```bash
python agent/agent_ab_test.py classifier   # routing diffs, incl. LONGKEY-key fix
python agent/agent_ab_test.py weaver       # prompt-size A/B: shows the overflow
```

Live (Neo4j + Ollama + API creds; loads both pipelines — needs RAM headroom):

```bash
python agent/agent_ab_test.py live "What is PROJ-18399 about and what do the docs say?"
```

Interactive: `python agent/intent_agent_test.py` — `/impl` shows which module variants are active, `/timing` shows per-node latency (watch `gpu_clear` drop to ~0 and `rag` include the reranker).

## How to revert / adopt

Originals untouched. To adopt: point your API layer at `agent.intent_agent_test` (same `ask()`/`get_graph()` API), or rename files once satisfied. To revert: delete the `*_test.py` files. Each test module also degrades gracefully — if you delete one, `intent_agent_test` falls back to the original for that piece.

## Verification (already run in an isolated sandbox)

All new files compile (`py_compile`), and the offline checks pass:

- **Classifier A/B** — `LONGKEY-12355` queries: original routed `rag` (live ticket silently skipped), test routes `both`/`sentries`. "ecommerce best practices": original mis-routed `both` via the hardcoded demo name, test routes `rag`. Bonus bug found while testing: the original's `merge.?request\b` **never matches the plural** "merge requests" (the trailing `\b` fails before the "s") — those queries only routed correctly when a hardcoded demo name happened to be present. Fixed with `merge.?requests?|pull.?requests?` in the test build.
- **Weaver prompt A/B** (synthetic both-mode turn: 8 RAG sources + 2×30-item sentry fan-out + 3-turn history): original prompt **44,163 chars ≈ 12,618 tokens → overflows num_ctx 10,240 and gets silently truncated** (instructions + KB block dropped first); optimized prompt 19,981 chars ≈ 5,708 tokens — fits.
- **Fan-out caps** through the real `route_query`: "show all merge requests" over 40 projects: 40 → **12** API calls; "how many jira tickets are open?" over 30 Jira projects: 30 → **10** calls.

Not verified here (needs your live services): full `live` mode with Neo4j + Ollama + API credentials — run `python agent/agent_ab_test.py live` yourself.

One caution: if results ever look stale while iterating, delete the `__pycache__` folders in `agent/` and `sentries/` — Python can serve old bytecode after files change.

# Hammer + Training Optimization — Diagnosis & Changes

Third round, same method as `rag/DIAGNOSIS.md` and `AGENT_SENTRIES_DIAGNOSIS.md`: every change lives in a new `*_test.py`; **no original file modified**. These layers don't serve user queries — they decide *what your model learns* and *whether an adapter ships*, so bugs here silently poison training data and the deployment gate.

## What these layers are

`hammer/` scores every `chat_history` turn (RAGAS + a deterministic validator), assigns a quality grade (GOLD/SILVER/BRONZE/FAILED) plus failure tags, exports training datasets (QLoRA/DPO/GRPO), and gates adapter deployment via a benchmark. `training/pipeline.py` prepares datasets from those grades, trains QLoRA/DPO adapters on an RTX 3050, and is supposed to promote them through the MLflow registry to the weaver.

## Bugs found

**H1 — The validator's query-echo exemptions are dead code (false hallucinations).** `validate_sentry_answer()` carefully exempts keys the user asked about ("the model citing PROJ-18399 when asked about PROJ-18399 is correct") — but `evaluator.py` **never passes `query=`** on either the `both` or `sentries` path, and `benchmark.py` doesn't either. So when sentries return empty and the model correctly addresses the asked ticket, it's penalised as `fabricated_keys_on_empty_sentries` → **hallucination tag → FAILED grade** → the answer is expelled from the GOLD pool and queued as a DPO "model failure". Confirmed offline: same doc scores 0.70 without query vs 0.85 with it.

**H2 — Pure-sentries docs never get the most precise check.** The sentries path calls the validator without `sentries_summary` (only `both` passes it), so the summary cross-check — "the most precise faithfulness check available" per its own docstring — never runs for `intent="sentries"` documents.

**H3 — `sentries_summary or []` destroys None semantics.** On the `both` path, `None` (old schema — payload simply not persisted) is coerced to `[]` ("sentries returned NOTHING"), arming the fabrication penalty against old-schema docs. Partially masked only when sources happen to contain Jira keys.

**H4 — The `{2,5}` Jira-key regex again** (`evaluator._JIRA_KEY_RE_TAGS`, `validator._JIRA_KEY_RE`). Long prefixes like `LONGKEY-12355` are invisible: an answer citing only such keys looks like it "cited NO keys" → **false `tool_misuse`** (critical → FAILED); `entity_match_relevance` misses the key-match path and underscores relevance. Same regex family bug fixed in the agent layer last round; widened to `{2,9}` here.

**H5 — Training prompts drifted from inference prompts.** `dataset_builder.py` claims to mirror `weaver_node.py` "exactly", but its `_BOTH_PROMPT` is an old copy ("combining the two data sources…") while the live weaver says "LIVE API DATA is the primary source of truth… do NOT say it is unavailable", with a different instruction list; the history label differs too. **Every exported QLoRA/DPO record teaches a prompt format the model never sees in production.**

**H6 — `GATE_ALLOW_EQUAL` does nothing.** In `benchmark.py` gate 3, the `if GATE_ALLOW_EQUAL:` and `else:` branches execute the *identical* comparison — the flag changes only the failure message.

**T1 — The promote loop is broken end-to-end.** Training logs the adapter as an artifact but **never calls `mlflow.register_model`**, so the "weaver-qwen" registry has no versions. Therefore `promote_model()` always raises (`search_model_versions` finds nothing) and `get_active_adapter()` always returns `None` — **the weaver can never pick up a trained adapter**. The train→promote→serve loop has never been completable.

**T2 — Adapter/serving model mismatch.** Training defaults to `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`; inference serves Ollama `qwen3` / `qwen3:8b`. A LoRA adapter only loads into the exact architecture+size it was trained on — **Qwen2.5-3B adapters can never serve on Qwen3-8B**, even after the weaver switches to HF inference. (The 6 GB VRAM budget explains the 3B choice; the plan needs either a 3B serving model or training on the serving model's architecture.)

**T3 — Tiny-dataset split bug.** `_split` gives the eval set priority: with 1 GOLD record, `train=[]`, `eval=[record]` — you'd fine-tune on nothing. Matters exactly when you're starting out with few GOLD docs.

**Minor / noted, not patched:** the adapter is `log_artifact`-ed twice per run (double upload); `hammer_eval_after_promote` hardcodes faithfulness=0.7 (a pseudo-metric); `run_evaluator` materialises the whole cursor into RAM.

**Integration notes for the optimized RAG stack (calibration, not code):** `sources[].score` from the reranked pipeline are cross-encoder sigmoids, not embedding cosines — recalibrate `HAMMER_RETRIEVAL_MISS_SRC` (default 0.65 was cosine-tuned; start near 0.30) and revisit `HAMMER_LATENCY_P95_RAG/_BOTH` since rerank shifts latency. Tier-2 re-retrieval can now follow production via `HAMMER_TIER2_ENGINE`.

## What changed, where

| File (new) | Fixes |
|---|---|
| `hammer/validator_test.py` | H4 at the source: patches `_JIRA_KEY_RE` to `{2,9}` inside `hammer.validator` (in-process; original file untouched) |
| `hammer/evaluator_test.py` | H1+H2 (query & summary passed on every path), H3 (None preserved), H4 (tag regex `{2,9}`), selectable Tier-2 engine (`HAMMER_TIER2_ENGINE=auto\|test\|original`); same CLI as the original |
| `hammer/benchmark_test.py` | H6 (allow-equal gate actually works, per-intent gate env-tunable), H1-in-benchmark (query/summary forwarded), scores via the fixed evaluator |
| `hammer/dataset_builder_test.py` | H5: imports prompt templates + history formatting **from the weaver module itself** (test build preferred) — drift is now structurally impossible |
| `training/pipeline_test.py` | T1 (promote registers the adapter on demand — old runs become promotable retroactively), T2 (`check-compat` command + loud warning, `TRAIN_COMPAT_ACK=1` to override), T3 (train-priority split) |
| `hammer/hammer_ab_test.py` | offline A/B for every fix: `echo`, `tags`, `prompts`, `gate`, `split` |

All overlays re-use the original modules for everything not being fixed, so behaviour stays identical elsewhere. Each runs with the same CLI as its original.

## How to test

Offline (free, instant):

```bash
python hammer/hammer_ab_test.py            # all five checks, originals measured first
```

Against your live Mongo (no writes with --dry-run):

```bash
python -m hammer.evaluator      --limit 20 --dry-run --force   # original scoring
python -m hammer.evaluator_test --limit 20 --dry-run --force   # fixed scoring
# → compare grades/failure_tags distribution; expect fewer false FAILED/hallucination
```

Re-score for real, then re-export training data with production-true prompts:

```bash
python -m hammer.evaluator_test --force
python -m hammer.dataset_builder_test qlora
python -m hammer.dataset_builder_test dpo
python training/pipeline_test.py check-compat   # see T2 before you train
```

## How to revert / adopt

Originals untouched — delete the `*_test.py` files to revert. To adopt: call the `_test` CLIs (same arguments), or rename when satisfied. Caveat: the overlays patch shared modules **in-process** (validator regex, builder templates), so don't import both original and test variants in one process when measuring the original — the harness orders its checks to avoid this; separate CLI invocations are always clean.

## Verification (already run in an isolated sandbox)

All six files compile, and all five offline A/B checks confirm the bugs and their fixes:

- **echo (H1):** answer echoing the asked ticket on empty sentries — original call style scores **0.700 with a false `fabricated_keys_on_empty_sentries` penalty** (→ hallucination tag → FAILED); with `query=` passed it scores **0.850, no penalties**.
- **prompts (H5):** `rag`/`sentries` templates match the weaver, **`both` has DRIFTED** (weaver: "LIVE API DATA is the primary source of truth…" vs builder: "combining the two data sources…"); after `dataset_builder_test`, templates match the weaver exactly.
- **tags (H4):** answer citing only `LONGKEY-*` keys that ARE in sources — original tags it **`tool_misuse` (critical → FAILED)** because `{2,5}` can't see the keys; fixed build emits **no tags**. `entity_match_relevance` on a long-prefix key: **1.00** (was token-overlap ≈0.4–0.7).
- **gate (H6):** truth-table shows the original's `GATE_ALLOW_EQUAL` branches are identical (Δ=0 and Δ=+0.003 rows differ only in the fixed build).
- **split (T3):** 1 GOLD record — original yields **train=0 / eval=1** (trains on nothing); fixed yields 1/0.

Not verifiable here (needs your services): live `--dry-run` evaluator comparison on real chat_history, MLflow promote (T1), and an actual training run.

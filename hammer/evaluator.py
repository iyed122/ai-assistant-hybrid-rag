#!/usr/bin/env python3
"""
Hammer Evaluator  —  Neo4j edition
════════════════════════════════════
Reads unscored chat_history documents, computes a weighted quality score,
and writes the evaluation subdoc back to MongoDB.

Scoring formula (v1.1):
  faithfulness         35%  — RAGAS (rag/both) | validator score (sentries)
  answer_relevance     35%  — RAGAS (rag/both) | entity_match_relevance (sentries)
  temporal_consistency 10%  — custom, pure Python, all intents
  code_correctness     20%  — Python AST, only when code blocks present;
                               weight redistributed to faithfulness otherwise

RAGAS LLM backend: Ollama (same model as Weaver — no extra API keys).

Context source — two-tier strategy:
  Tier 1 (fast): If chat_history stores context_snippets (written at inference
    time by intent_agent.py), use those directly — no re-retrieval needed.
  Tier 2 (fallback): Re-query Neo4j hybrid_search() at eval time.
    Neo4j hybrid_search provides richer context — vector hits plus
    context (graph-expanded chunks) at the cost of higher per-query latency
    multi-hop graph traversal. Latency is ~0.5–2 s. The ThreadPoolExecutor
    in run_evaluator() keeps total batch time manageable.

Graph coverage: eval_doc now records graph_coverage (fraction of stored
  sources that came from graph expansion) as an audit metric. This is not
  part of the weighted score formula but is stored for trend analysis.

Critical rule:
  Do NOT run RAGAS on sentries-only responses — no retrieved passage to
  check faithfulness against. Use the deterministic validator instead.
"""

from __future__ import annotations

import ast
import gc
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL",   "Snowflake/snowflake-arctic-embed-m")
OLLAMA_HOST       = os.getenv("OLLAMA_HOST",       "http://localhost:11434").strip().rstrip("/")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL",      "qwen3").strip()
ARCTIC_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Graph retrieval parameters — mirror neo4j_search.py defaults
GRAPH_HOPS        = int(os.getenv("GRAPH_HOPS",       "2"))
MAX_GRAPH_EXPAND  = int(os.getenv("MAX_GRAPH_EXPAND",  "20"))

# Scoring weights
# W_TEMPORAL cut from 0.20 → 0.10: temporal checks almost never fire for this
# assistant; the freed 0.10 moves to answer_relevance which is the more
# informative signal (especially after the entity-match fix for sentries).
W_FAITHFULNESS     = 0.35
W_ANSWER_RELEVANCE = 0.35   # was 0.25
W_TEMPORAL         = 0.10   # was 0.20
W_CODE             = 0.20

# Training signal thresholds
THRESHOLD_QLORA = 0.80
THRESHOLD_DPO   = 0.50

# ─── NaN guard ────────────────────────────────────────────────────────────────
import math as _math

def _safe_float(val, penalty: float = 0.30) -> float:
    """
    Convert val to float, replacing NaN/None with a conservative penalty.

    Fixes the silent-1.0 bug: min(1.0, NaN) == 1.0 in Python because
    NaN comparisons always return False.  Any timed-out RAGAS metric
    arrives here as NaN and is replaced with 0.30 before arithmetic.
    """
    if val is None:
        return penalty
    try:
        f = float(val)
        return penalty if _math.isnan(f) else f
    except (TypeError, ValueError):
        return penalty

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hammer.evaluator")

# ─── Optional dependency flags ────────────────────────────────────────────────
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
            "neo4j_search not importable — context re-retrieval disabled. "
            "pip install neo4j sentence-transformers"
        )

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not available — embedding metrics disabled")

try:
    import ragas
    RAGAS_AVAILABLE = True
    logger.info("RAGAS found: %s", ragas.__version__)
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("ragas not installed — using embedding-based approximations (pip install ragas)")

# ─── Lazy singletons ──────────────────────────────────────────────────────────
_mongo_client:        Optional[MongoClient] = None
_neo4j_search_inst    = None   # Neo4jSearch singleton
_embed_model          = None


def _get_db():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI)
    return _mongo_client[MONGO_DB]


def _get_neo4j_search():
    """
    Return a shared Neo4jSearch instance, or None if unavailable.

    Neo4jSearch is heavy to initialise (it loads the
    embedding model on first instantiation), so we keep a singleton and
    only create it once per evaluator process.
    """
    global _neo4j_search_inst
    if not NEO4J_SEARCH_AVAILABLE:
        return None
    if _neo4j_search_inst is None:
        try:
            _neo4j_search_inst = _Neo4jSearch()
            logger.info("Neo4j search instance ready for context re-retrieval")
        except Exception as e:
            logger.warning("Neo4jSearch init failed: %s", e)
    return _neo4j_search_inst


def _get_embed_model():
    """
    Return the sentence-transformer model.
    Shared with Neo4jSearch internally, but also used standalone for
    embedding-approx fallbacks (RAGAS failure path).
    """
    global _embed_model
    if not EMBEDDINGS_AVAILABLE:
        return None
    if _embed_model is None:
        try:
            _embed_model = SentenceTransformer(EMBEDDING_MODEL)
            _embed_model.max_seq_length = 512
        except Exception as e:
            logger.warning("Embedding model load failed: %s", e)
    return _embed_model


# ═══════════════════════════════════════════════════════════════════════════════
# Context retrieval — two-tier (snippets from MongoDB, then Neo4j fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve_context(
    query: str,
    top_k: int = 5,
    doc: Optional[Dict] = None,
) -> Tuple[List[str], str]:
    """
    Retrieve passage text for RAGAS faithfulness scoring.

    Two-tier strategy:
      Tier 1 — Stored context_snippets (fast, no re-retrieval):
        If the chat_history document includes context_snippets (written by
        intent_agent.py at inference time), use those directly.  These are
        the actual passages the LLM saw when generating the answer, making
        them the ground truth for faithfulness scoring.

      Tier 2 — Neo4j hybrid_search() re-retrieval (fallback):
        If no snippets are stored, re-query Neo4j using the original query.
        Hybrid search (vector + multi-hop graph) is used so the retrieved
        contexts match the production retrieval path as closely as possible.

    Args:
        query:  Original user query.
        top_k:  Number of passages to retrieve (Tier 2 only).
        doc:    The full chat_history document (used for Tier 1 lookup).

    Returns:
        Tuple of (passages: List[str], tier: str) where tier is one of:
          "stored_snippets"      — Tier 1, ground-truth inference context
          "neo4j_tier2_fallback" — Tier 2, reconstructed context (not what
                                   the model saw; faithfulness scores against
                                   this are approximate)
          "empty"                — neither tier returned anything
    """
    # ── Tier 1: use stored snippets if available ──────────────────────────────
    if doc is not None:
        snippets = doc.get("context_snippets") or []
        if snippets:
            texts = [s.get("text", "") if isinstance(s, dict) else str(s) for s in snippets]
            texts = [t for t in texts if t and len(t) > 15]
            if texts:
                logger.info("retrieve_context: Tier 1 — %d stored snippets (no Neo4j call)", len(texts))
                return texts, "stored_snippets"

    # ── Tier 2: Neo4j hybrid_search() re-retrieval ───────────────────────────
    searcher = _get_neo4j_search()
    if searcher is None:
        logger.debug("retrieve_context: Neo4j unavailable — returning empty context")
        return [], "empty"
    try:
        hits = searcher.hybrid_search(
            query=query,
            top_k=top_k,
            hops=GRAPH_HOPS,
            max_expand=MAX_GRAPH_EXPAND,
        )
        texts = [h.get("text", "") for h in hits if h.get("text") and len(h.get("text", "")) > 15]
        logger.debug(
            "retrieve_context: Tier 2 — %d Neo4j hits → %d texts (hops=%d)",
            len(hits), len(texts), GRAPH_HOPS,
        )
        return texts, "neo4j_tier2_fallback"
    except Exception as e:
        logger.warning("Neo4j context re-retrieval failed: %s", e)
        return [], "empty"


# ═══════════════════════════════════════════════════════════════════════════════
# RAGAS setup — Ollama as LLM backend
# ═══════════════════════════════════════════════════════════════════════════════

_ragas_llm_inst = None
_ragas_emb_inst = None


def _build_ragas_llm():
    global _ragas_llm_inst
    if _ragas_llm_inst is not None:
        return _ragas_llm_inst
    """
    Build a RAGAS-compatible LLM wrapper backed by Ollama.
    Tries langchain_community first, then langchain (older API).
    Returns None if any import fails.
    """
    for module_path in (
        "langchain_community.llms",
        "langchain.llms",
    ):
        try:
            import importlib
            llms_mod = importlib.import_module(module_path)
            OllamaLLM = getattr(llms_mod, "Ollama")
            from ragas.llms import LangchainLLMWrapper
            llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0.0)
            _ragas_llm_inst = LangchainLLMWrapper(llm)
            return _ragas_llm_inst
        except Exception:
            continue
    logger.warning("Could not build RAGAS LLM wrapper — falling back to embedding approximations")
    return None


def _build_ragas_embeddings():
    """
    Build RAGAS-compatible embeddings wrapper using the local HuggingFace model.
    Cached as a singleton — HuggingFaceEmbeddings reloads the model on every
    instantiation (causing the repeated 'Loading SentenceTransformer' log lines
    and 5–15 s of HF network chatter per document).
    """
    global _ragas_emb_inst
    if _ragas_emb_inst is not None:
        return _ragas_emb_inst
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        hf = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _ragas_emb_inst = LangchainEmbeddingsWrapper(hf)
        return _ragas_emb_inst
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 1 — RAGAS faithfulness + answer relevance
# ═══════════════════════════════════════════════════════════════════════════════

def _cosine_sim(a, b) -> float:
    """Cosine similarity between two numpy arrays."""
    import numpy as np
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _approx_faithfulness(answer: str, contexts: List[str]) -> float:
    """
    Embedding-based faithfulness approximation.

    Algorithm:
      1. Split answer into sentences (>15 chars).
      2. For each sentence compute max cosine-sim against 200-char context windows.
      3. Faithfulness = fraction of sentences with sim > 0.45.

    Used when RAGAS/LLM is unavailable.
    """
    model = _get_embed_model()
    if model is None or not contexts:
        return 0.5

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 15]
    if not sentences:
        return 0.5

    ctx_text   = " ".join(contexts)
    ctx_chunks = [ctx_text[i : i + 200] for i in range(0, len(ctx_text), 160)][:30]

    scores = []
    for sent in sentences:
        s_emb    = model.encode(sent, normalize_embeddings=True)
        max_sim  = max(
            (_cosine_sim(s_emb, model.encode(c, normalize_embeddings=True)) for c in ctx_chunks if len(c) > 15),
            default=0.0,
        )
        scores.append(max_sim)

    supported = sum(1 for s in scores if s > 0.45)
    return round(supported / len(scores), 4) if scores else 0.5


def _approx_answer_relevance(query: str, answer: str) -> float:
    """Embedding cosine-sim between query and answer (truncated to 800 chars)."""
    model = _get_embed_model()
    if model is None:
        return 0.5
    try:
        q_emb = model.encode(query, normalize_embeddings=True)
        a_emb = model.encode(answer[:800], normalize_embeddings=True)
        return round(max(0.0, min(1.0, _cosine_sim(q_emb, a_emb))), 4)
    except Exception:
        return 0.5


def score_ragas(
    query:    str,
    answer:   str,
    contexts: List[str],
) -> Dict[str, Any]:
    """
    Run RAGAS faithfulness + answer_relevance.

    Tries RAGAS 0.2+ API first, then 0.1.x API, then embedding approximations.
    Always returns a dict with keys: faithfulness, answer_relevance, method.
    """
    if not contexts:
        return {
            "faithfulness":     0.30,   # penalise: answer without grounding
            "answer_relevance": _approx_answer_relevance(query, answer),
            "method":           "no_context_penalty",
        }

    if RAGAS_AVAILABLE:
        ragas_llm = _build_ragas_llm()
        ragas_emb = _build_ragas_embeddings()

        if ragas_llm is not None:
            # ── Try RAGAS 0.2+ (dataset_schema API) ──────────────────────────
            try:
                from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
                from ragas.metrics import Faithfulness, AnswerRelevancy
                from ragas import evaluate as ragas_eval

                metrics = [Faithfulness(), AnswerRelevancy()]
                for m in metrics:
                    m.llm = ragas_llm
                    if hasattr(m, "embeddings") and ragas_emb:
                        m.embeddings = ragas_emb

                sample  = SingleTurnSample(
                    user_input=query,
                    response=answer,
                    retrieved_contexts=contexts,
                )
                dataset = EvaluationDataset(samples=[sample])
                result  = ragas_eval(dataset=dataset, metrics=metrics)
                row     = result.to_pandas().iloc[0].to_dict()

                faith = float(row.get("faithfulness", float("nan")))
                ar    = float(row.get("answer_relevancy", row.get("answer_relevance", float("nan"))))
                # NaN means RAGAS timed out internally — fall through to 0.1 API / embedding fallback
                if _math.isnan(faith) or _math.isnan(ar):
                    raise ValueError(
                        f"RAGAS 0.2 returned NaN (LLM timeout?): "
                        f"faithfulness={faith}  answer_relevance={ar}"
                    )
                return {
                    "faithfulness":     faith,
                    "answer_relevance": ar,
                    "method":           "ragas_0_2",
                }
            except (ImportError, AttributeError, Exception) as e:
                logger.debug("RAGAS 0.2 API failed (%s), trying 0.1 API", e)

            # ── Try RAGAS 0.1.x (HuggingFace Dataset API) ───────────────────
            try:
                from datasets import Dataset
                from ragas import evaluate as ragas_eval
                from ragas.metrics import faithfulness as f_metric, answer_relevancy as ar_metric

                f_metric.llm  = ragas_llm
                ar_metric.llm = ragas_llm
                if ragas_emb:
                    ar_metric.embeddings = ragas_emb

                # Raise the per-job timeout from the default 180 s to 600 s so
                # slow cloud-proxied LLMs (150–170 s round-trip) don't hit the
                # watchdog and silently return NaN.
                try:
                    from ragas.run_config import RunConfig
                    _run_cfg = RunConfig(timeout=600, max_retries=2, max_wait=60)
                except Exception:
                    _run_cfg = None

                ds     = Dataset.from_dict({"question": [query], "answer": [answer], "contexts": [contexts]})
                result = ragas_eval(
                    ds,
                    metrics=[f_metric, ar_metric],
                    **( {"run_config": _run_cfg} if _run_cfg is not None else {} ),
                )

                faith = float(result["faithfulness"])
                ar    = float(result["answer_relevancy"])
                # NaN means RAGAS timed out — fall through to embedding fallback
                if _math.isnan(faith) or _math.isnan(ar):
                    raise ValueError(
                        f"RAGAS 0.1 returned NaN (LLM timeout?): "
                        f"faithfulness={faith}  answer_relevance={ar}"
                    )
                return {
                    "faithfulness":     faith,
                    "answer_relevance": ar,
                    "method":           "ragas_0_1",
                }
            except Exception as e:
                logger.warning("RAGAS 0.1 API also failed (%s) — using embedding approximation", e)

    # ── Fallback: embedding approximations ────────────────────────────────────
    return {
        "faithfulness":     _approx_faithfulness(answer, contexts),
        "answer_relevance": _approx_answer_relevance(query, answer),
        "method":           "embedding_approx",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 2 — Temporal consistency (custom, pure Python)
# ═══════════════════════════════════════════════════════════════════════════════

_YEAR_RE     = re.compile(r"\b(20\d{2}|19\d{2})\b")
_BEFORE_YEAR = re.compile(r"\bbefore\s+(20\d{2}|19\d{2})\b", re.I)
_AFTER_YEAR  = re.compile(r"\bafter\s+(20\d{2}|19\d{2})\b",  re.I)
_SEQUENCE_RE = re.compile(
    r"\b(first|then|next|afterward|finally|step\s+\d+|subsequently|before|previously)\b",
    re.I,
)


def score_temporal_consistency(answer: str) -> float:
    """
    Checks:
      1. No future years cited (> current_year + 1).
      2. "before YYYY" year < "after YYYY" year — no backward timeline.
      3. "finally" appears after "first" in the text.
      4. Step numbers in ascending order (step 1 → step 2 → step 3).

    Returns 1.0 if no inconsistencies detected, lower otherwise.
    """
    if not answer or len(answer) < 40:
        return 1.0

    issues      = 0
    total_checks = 0
    now_year    = datetime.now().year

    # Check 1: unreasonable future years
    years = [int(y) for y in _YEAR_RE.findall(answer)]
    if years:
        total_checks += 1
        if any(y > now_year + 1 for y in years):
            issues += 1

    # Check 2: before/after ordering
    # Valid:       "after 2020 ... before 2025"  → min_before=2025 > min_after=2020  → ok
    # Contradiction: "before 2020 ... after 2025" → min_before=2020 < min_after=2025  → issue
    before_years = [int(y) for y in _BEFORE_YEAR.findall(answer)]
    after_years  = [int(y) for y in  _AFTER_YEAR.findall(answer)]
    if before_years and after_years:
        total_checks += 1
        if min(before_years) > min(after_years):   # "after 2020 ... before 2025" is valid
            pass
        else:
            issues += 1   # "before 2020 ... after 2025" is a timeline contradiction

    # Check 3: finally before first
    text_lower  = answer.lower()
    finally_pos = text_lower.find("finally")
    first_pos   = text_lower.find("first")
    if finally_pos != -1 and first_pos != -1:
        total_checks += 1
        if finally_pos < first_pos:
            issues += 1

    # Check 4: numbered steps in order
    step_numbers = [int(m) for m in re.findall(r"\bstep\s+(\d+)\b", answer, re.I)]
    if len(step_numbers) >= 3:
        total_checks += 1
        if step_numbers != sorted(step_numbers):
            issues += 1

    if total_checks == 0:
        return 1.0

    return round(max(0.0, 1.0 - issues / total_checks), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Metric 3 — Code correctness (Python AST)
# ═══════════════════════════════════════════════════════════════════════════════

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.I)
_PYTHON_SIGNAL = re.compile(r"\b(def |import |class |print\(|from \w+ import)\b")


def score_code_correctness(answer: str) -> Optional[float]:
    """
    Parses Python code blocks in the answer with ast.parse.

    Returns:
      None   — no code blocks found (weight redistributed to faithfulness)
      0.0–1.0 — fraction of Python-looking code blocks with valid syntax

    Non-Python blocks (JS, bash, YAML) are ignored (assumed correct).
    """
    blocks = _CODE_BLOCK_RE.findall(answer)
    if not blocks:
        return None

    total_python = 0
    valid_python = 0

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if not _PYTHON_SIGNAL.search(block):
            continue   # not Python — skip
        total_python += 1
        try:
            ast.parse(block)
            valid_python += 1
        except SyntaxError:
            pass   # genuine Python with syntax error → counts as invalid

    if total_python == 0:
        return None   # only non-Python code found — treat as no-code

    return round(valid_python / total_python, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Weighted score aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_weighted_score(
    faithfulness:    float,
    answer_relevance: float,
    temporal:        float,
    code:            Optional[float],
) -> float:
    """
    Aggregate the four metrics into a single weighted score.

    When code is None (no Python blocks), the W_CODE weight shifts to
    faithfulness so scores still sum to 1.0:
      faithfulness     → 35% + 20% = 55%
      answer_relevance → 35%
      temporal         → 10%

    All inputs are passed through _safe_float() first — NaN values
    (e.g. from a timed-out RAGAS call) are replaced with 0.30 so they
    never silently resolve to 1.0 via min(1.0, NaN).
    """
    # NaN guard — fixes: min(1.0, NaN) == 1.0 silent corruption bug
    faithfulness     = _safe_float(faithfulness)
    answer_relevance = _safe_float(answer_relevance)
    temporal         = _safe_float(temporal)
    if code is not None:
        code = _safe_float(code)

    if code is not None:
        score = (
            faithfulness     * W_FAITHFULNESS    +
            answer_relevance * W_ANSWER_RELEVANCE +
            temporal         * W_TEMPORAL         +
            code             * W_CODE
        )
    else:
        w_faith_adjusted = W_FAITHFULNESS + W_CODE
        score = (
            faithfulness     * w_faith_adjusted  +
            answer_relevance * W_ANSWER_RELEVANCE +
            temporal         * W_TEMPORAL
        )

    return round(max(0.0, min(1.0, score)), 4)


def _training_signal(score: float) -> str:
    """DEPRECATED — legacy score-only signal. Kept for reference only.
    Do not use for new docs; use _grade_to_signal() instead, which respects
    failure tags. Computing the signal from score alone leaks tagged failures
    (e.g. a 0.89-scoring tool_misuse doc) into the qlora_positive pool."""
    if score >= THRESHOLD_QLORA:
        return "qlora_positive"
    if score < THRESHOLD_DPO:
        return "dpo_rejected"
    return "neutral"


def _grade_to_signal(grade: str, failure_tags: Optional[List[str]] = None,
                     score: float = 1.0) -> str:
    """Map the 2D classification to the legacy single-axis training_signal.

    This is the source of truth for training_signal. It respects both the
    grade and the failure tags so the label is genuinely actionable:

        GOLD                              → qlora_positive
        FAILED + hallucination/tool_misuse → dpo_rejected   (model-behaviour
                                              failure DPO can correct)
        FAILED + score < 0.5              → dpo_rejected   (clearly bad answer)
        FAILED (other, e.g. retrieval_miss) → neutral       (NOT a model problem
                                              — excluded from DPO so we don't
                                              train the model to compensate for
                                              bad retrieval)
        SILVER / BRONZE                   → neutral

    Why hallucination/tool_misuse → dpo_rejected but retrieval_miss → neutral:
    DPO corrects model behaviour by teaching chosen-over-rejected. A fabrication
    or ignored-retrieval is a behaviour the model controls and can be corrected.
    A retrieval_miss is the retriever's fault — using it as a DPO rejection would
    teach the model the wrong lesson (answer well despite bad context).
    """
    tags = failure_tags or []
    if grade == "GOLD":
        return "qlora_positive"
    if grade == "FAILED":
        dpo_actionable = (
            "hallucination" in tags
            or "tool_misuse" in tags
            or score < THRESHOLD_DPO
        )
        return "dpo_rejected" if dpo_actionable else "neutral"
    # SILVER / BRONZE
    return "neutral"


# ═══════════════════════════════════════════════════════════════════════════════
# Enterprise-grade 2D classification: Quality Grade + Failure Tags
# ═══════════════════════════════════════════════════════════════════════════════
#
# Replaces the single-axis `training_signal` with two orthogonal axes:
#   Axis 1  — quality_grade : GOLD / SILVER / BRONZE / FAILED  (mutually exclusive)
#   Axis 2  — failure_tags  : zero or more tags from a fixed vocabulary
#
# This mirrors the model used by Datadog / Sentry / OpenTelemetry GenAI:
# a single status tier plus multi-valued tags. Critical tags override the
# score-band rule (hallucination at any score → FAILED).
#
# Tag vocabulary (in severity order):
#   hallucination   — CRITICAL  (validator caught fabrication patterns)
#   tool_misuse     — CRITICAL  (sources retrieved but not exploited)
#   retrieval_miss  — CRITICAL  (model wasn't given good context — not its fault)
#   incomplete      — high      (404 speculation / verbose empty)
#   format_violation— medium    (non-fabrication validator penalty)
#   latency_breach  — low       (intent-level p95 exceeded)
# ═══════════════════════════════════════════════════════════════════════════════

# Critical tags that force quality_grade=FAILED regardless of weighted_score
CRITICAL_TAGS = {"hallucination", "tool_misuse", "retrieval_miss"}

# Validator notes that map to the `hallucination` tag.
# Currently we only fire on deterministically-caught fabrication patterns —
# per user direction, we ignore the rare "key hallucinated entirely" case
# because the Neo4j key-mapping makes it structurally near-impossible.
_HALLUCINATION_NOTES = {
    "fabricated_keys_on_empty_sentries",
    "jira_prefix_mismatch",
}

# Pre-compiled Jira-key regex (matches AUTH-12, PROJ-18208, LONGKEY-12355, etc.)
# FLATTENED FIX: was {2,5} — long-prefix keys were invisible to failure-tag
# derivation, causing false tool_misuse on answers citing only such keys.
_JIRA_KEY_RE_TAGS = re.compile(r"\b[A-Z]{2,9}-\d+\b")

# Retrieval miss threshold for mean source similarity score
# 0.65 is the default for the Snowflake/snowflake-arctic-embed-m embedding model
# below which retrieved chunks are usually weakly related to the query.
RETRIEVAL_MISS_FAITHFULNESS_THRESHOLD = float(
    os.getenv("HAMMER_RETRIEVAL_MISS_FAITH", "0.50")
)
RETRIEVAL_MISS_SOURCE_SCORE_THRESHOLD = float(
    os.getenv("HAMMER_RETRIEVAL_MISS_SRC", "0.65")
)

# Incomplete-answer thresholds (404 speculation pattern)
INCOMPLETE_MIN_LEN = int(os.getenv("HAMMER_INCOMPLETE_MIN_LEN", "200"))

# Latency p95 caps per intent — used by `latency_breach` tag.
# Tuned conservatively from the 286-doc observed distribution:
#   sentries: median ~0.5 s, p95 ~3.5 s
#   rag     : median ~1.6 s, p95 ~5.0 s
#   both    : median ~2.0 s, p95 ~7.0 s
_LATENCY_P95_PER_INTENT = {
    "sentries": float(os.getenv("HAMMER_LATENCY_P95_SENTRIES", "3.5")),
    "rag":      float(os.getenv("HAMMER_LATENCY_P95_RAG",      "5.0")),
    "both":     float(os.getenv("HAMMER_LATENCY_P95_BOTH",     "7.0")),
}

# Honest-empty regex (matches validator's pattern — kept here for tag derivation
# without a hard import on validator.py, which may not be available in all
# deployments)
_HONEST_EMPTY_TAG_RE = re.compile(
    r"\b(no\s+(results?|data|tickets?|issues?|matches?|pages?|matching)\s+(found|available|returned)|"
    r"could\s+not\s+find|"
    r"aucun(e)?\s+(résultat|donnée|ticket|page)|"
    r"empty\s+result|"
    r"i\s+(don[’']?t|do\s+not)\s+have\s+(any\s+)?data)\b",
    re.IGNORECASE,
)


def _extract_jira_keys_from_sources(sources: List[Dict[str, Any]]) -> set:
    """
    Pull every Jira key that the retrievers actually returned for this query.
    Looks at issue_key, url, and title fields — sources are dicts, not free text.
    """
    keys: set = set()
    for s in sources or []:
        if s.get("source") != "jira":
            continue
        ik = s.get("issue_key", "")
        if ik:
            keys.add(ik.upper())
        url = s.get("url", "") or ""
        keys.update(k.upper() for k in _JIRA_KEY_RE_TAGS.findall(url))
        title = s.get("title", "") or ""
        keys.update(k.upper() for k in _JIRA_KEY_RE_TAGS.findall(title))
    return keys


def _mean_source_score(sources: List[Dict[str, Any]]) -> float:
    """Average similarity score across retrieved sources (0 if none)."""
    if not sources:
        return 0.0
    scores = [s.get("score", 0) for s in sources if isinstance(s.get("score"), (int, float))]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compute_failure_tags(
    eval_doc:  Dict[str, Any],
    doc:       Dict[str, Any],
) -> List[str]:
    """
    Derive failure tags from eval_doc + the original chat_history doc.

    Tags are multi-valued — a single document can carry several. Each tag has
    one deterministic trigger; no tag is derived from another tag (no chaining).

    Args:
        eval_doc:  The partially-built evaluation subdoc (must have faithfulness,
                   weighted_score, intent, optionally validator_details).
        doc:       The original chat_history document (sources, answer, timing).

    Returns:
        Sorted list of unique tag strings (may be empty).
    """
    tags: set = set()

    intent          = eval_doc.get("intent", doc.get("intent", "both"))
    answer          = doc.get("answer", "") or ""
    sources         = doc.get("sources", []) or []
    sentries_summary = doc.get("sentries_summary")   # may be None or []
    total_time_s    = doc.get("total_time_s", 0) or 0
    faith           = eval_doc.get("faithfulness", 0.5)
    vd              = eval_doc.get("validator_details") or {}

    # ── Tag 1: hallucination (CRITICAL) ──────────────────────────────────────
    # Fired when the validator deterministically caught fabrication.
    for ch in vd.get("checks", []):
        if ch.get("penalty", 0) <= 0:
            continue
        note_cat = (ch.get("note", "") or "").split(":")[0].strip()
        if note_cat in _HALLUCINATION_NOTES:
            tags.add("hallucination")
            break

    # ── Tag 2: tool_misuse (CRITICAL) ────────────────────────────────────────
    # The dispatcher returned retrievable data but the answer ignored it.
    #
    # Graph-awareness: sources include graph-expanded nodes (from_graph=True)
    # that are *related* to the directly retrieved items via CHILD_OF, RELATES_TO,
    # BLOCKS, etc. When the answer cites a key that is reachable via graph
    # traversal from a directly-retrieved key, that is legitimate context use —
    # NOT tool misuse. We build an expanded key set that includes both direct
    # and graph-neighbour keys before comparing against the answer.
    #
    # Applies to: sentries, both, AND rag — the rag path can also receive
    # graph-expanded Jira/Confluence nodes via Neo4j hybrid_search().
    if intent in ("sentries", "both", "rag") and sources:
        # Direct (non-graph) source keys
        direct_keys = _extract_jira_keys_from_sources(
            [s for s in sources if not s.get("from_graph", False)]
        )
        # Graph-neighbour keys — reachable via graph traversal.
        # Treat these as legitimate context regardless of relationship type
        # or distance (max_distance is already capped by GRAPH_HOPS at retrieval).
        graph_keys = _extract_jira_keys_from_sources(
            [s for s in sources if s.get("from_graph", False)]
        )
        # Full reachable key set — anything the model legitimately had access to
        all_reachable_keys = direct_keys | graph_keys

        answer_upper = answer.upper()
        cited_keys   = set(_JIRA_KEY_RE_TAGS.findall(answer_upper))

        # Only flag tool_misuse when sentries/both had live data to work with.
        # For rag intent, only flag if there are Jira sources (not pure Confluence/GitLab).
        has_jira_sources = bool(all_reachable_keys)

        # Keys the user named in the query are always legitimate to echo back —
        # they are not fabrications and not tool misuse, even when the retriever
        # did not return them (e.g. live API unavailable, wrong-project RAG hits).
        query_keys = set(_JIRA_KEY_RE_TAGS.findall(doc.get("query", "").upper()))

        if has_jira_sources:
            # 2a — model cited ONLY unreachable, non-echo keys.
            #
            # Graph-awareness: all_reachable_keys already includes graph-expanded
            # neighbours (from_graph=True), so a graph-cited key counts as reachable
            # and does NOT trip this rule. Query-echoed keys are likewise exempt.
            #
            # The rule fires only when, after removing graph-reachable keys and
            # query-echoed keys, the answer cited keys AND none of them were
            # reachable — i.e. the model invented every key it cited. A MIX of
            # reachable (direct or graph) and one stray key is legitimate
            # synthesis, not misuse, so it must not fire.
            non_echo_cited = cited_keys - query_keys
            if non_echo_cited and not (all_reachable_keys & non_echo_cited):
                tags.add("tool_misuse")
            # 2b — reachable keys exist, answer is substantive, but cites NO keys
            # and doesn't acknowledge emptiness — model wrote generic prose
            # while ignoring all retrieved entities.
            # ONLY for sentries/both: rag answers legitimately describe concepts
            # from Confluence docs without citing every Jira key in sources.
            # EXEMPT analytical queries: cross-entity questions (compare, trace,
            # find where X appears as Y, which subsystem, most referenced) produce
            # correct narrative answers without explicit key citations. Flagging
            # these as tool_misuse is a false positive.
            elif intent in ("sentries", "both"):
                _ANALYTICAL_RE = re.compile(
                    r"\b(where|which|compare|trace|find\s+all|most\s+referenced|"
                    r"cross.?reference|appear|conflict|risk|subsystem|summarize|"
                    r"summarise)\b", re.I,
                )
                is_analytical = bool(_ANALYTICAL_RE.search(doc.get("query", "")))
                if (all_reachable_keys and not cited_keys
                        and len(answer) > 100
                        and not is_analytical
                        and not _HONEST_EMPTY_TAG_RE.search(answer)):
                    tags.add("tool_misuse")

        # 2c — claimed empty when sources are non-empty (applies to all intents)
        if sources and _HONEST_EMPTY_TAG_RE.search(answer):
            tags.add("tool_misuse")

    # ── Tag 3: retrieval_miss (CRITICAL) ─────────────────────────────────────
    # Low faithfulness combined with weakly-relevant sources = retriever's fault,
    # not the model's. We exclude these from training data entirely.
    if faith < RETRIEVAL_MISS_FAITHFULNESS_THRESHOLD:
        mss = _mean_source_score(sources)
        if mss > 0 and mss < RETRIEVAL_MISS_SOURCE_SCORE_THRESHOLD:
            tags.add("retrieval_miss")

    # ── Tag 4: incomplete (high) ─────────────────────────────────────────────
    # 404-speculation pattern: model writes a long substantive answer when no
    # sentries data is available, instead of saying "not found".
    # Triggers ONLY when ALL of the following hold:
    #   - intent is sentries or both
    #   - sentries_summary is explicitly empty/None
    #   - sources are also empty (no Jira/Confluence/GitLab hits to ground in)
    #   - answer is long
    #   - answer doesn't contain honest-empty phrasing
    #   - answer doesn't cite any Jira keys (if it does, the answer is grounded
    #     in real entities and is not 404 speculation, even if the summary field
    #     is missing — this is a common case in older chat_history docs where
    #     sentries_summary was never written but sentries did return data)
    if intent in ("sentries", "both"):
        sentries_empty = (
            sentries_summary is None
            or sentries_summary == []
            or (isinstance(sentries_summary, list) and len(sentries_summary) == 0)
        )
        sources_empty = not sources
        answer_cites_keys = bool(_JIRA_KEY_RE_TAGS.search(answer))

        if (sentries_empty
                and sources_empty
                and not answer_cites_keys
                and len(answer) > INCOMPLETE_MIN_LEN
                and not _HONEST_EMPTY_TAG_RE.search(answer)):
            tags.add("incomplete")

    # ── Tag 5: format_violation (medium) ─────────────────────────────────────
    # Any non-fabrication validator penalty fired.
    for ch in vd.get("checks", []):
        if ch.get("penalty", 0) <= 0:
            continue
        note_cat = (ch.get("note", "") or "").split(":")[0].strip()
        if note_cat not in _HALLUCINATION_NOTES:
            # Don't double-count: if hallucination already fired on the same doc,
            # we still tag format_violation since a doc can have both kinds of issues.
            tags.add("format_violation")
            break

    # ── Tag 6: latency_breach (low) ──────────────────────────────────────────
    p95 = _LATENCY_P95_PER_INTENT.get(intent, 10.0)
    if total_time_s > p95:
        tags.add("latency_breach")

    return sorted(tags)


def compute_quality_grade(
    eval_doc: Dict[str, Any],
    tags:     List[str],
) -> str:
    """
    Compute the quality grade — a single mutually-exclusive tier per doc.

    Rules (applied in order, first match wins):
        1. Any CRITICAL tag fired → FAILED
        2. weighted_score < 0.50  → FAILED
        3. weighted_score ≥ 0.85 AND faithfulness ≥ 0.70 AND no tags → GOLD
        4. weighted_score ≥ 0.70  → SILVER
        5. weighted_score ≥ 0.50  → BRONZE
        6. fallback               → FAILED

    The asymmetry between GOLD (requires no tags) and SILVER/BRONZE (tolerate
    non-critical tags) is intentional: GOLD is the training-data pool, and
    we want it pristine.
    """
    if any(t in CRITICAL_TAGS for t in tags):
        return "FAILED"

    score = eval_doc.get("weighted_score", 0.0)
    faith = eval_doc.get("faithfulness",   0.0)

    if score < 0.50:
        return "FAILED"
    if score >= 0.85 and faith >= 0.70 and not tags:
        return "GOLD"
    if score >= 0.70:
        return "SILVER"
    if score >= 0.50:
        return "BRONZE"
    return "FAILED"


# ═══════════════════════════════════════════════════════════════════════════════
# Document evaluator
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a single chat_history document.

    Returns an evaluation subdoc ready to be $set under doc["evaluation"].
    Never raises — errors are recorded in eval_doc["error"].
    """
    query  = doc.get("query",  "")
    answer = doc.get("answer", "")
    intent = doc.get("intent", "both")

    eval_doc: Dict[str, Any] = {
        "intent":       intent,
        "evaluated_at": datetime.utcnow(),
    }

    if not query or not answer:
        eval_doc.update({
            "weighted_score":   0.0,
            "training_signal":  "dpo_rejected",
            "quality_grade":    "FAILED",
            "failure_tags":     ["incomplete"],
            "error":            "missing query or answer",
        })
        return eval_doc

    # ── Temporal consistency (all intents) ────────────────────────────────────
    temporal = score_temporal_consistency(answer)
    eval_doc["temporal_consistency"] = temporal

    # ── Code correctness (all intents, only when Python blocks present) ───────
    code = score_code_correctness(answer)
    eval_doc["code_correctness"] = code   # may be None

    # ── Intent-specific faithfulness + answer_relevance ───────────────────────
    if intent == "rag":
        # Pure RAG: answer is fully grounded in retrieved passages → full RAGAS.
        contexts, ctx_tier = retrieve_context(query, top_k=5, doc=doc)
        eval_doc["num_contexts"]          = len(contexts)
        eval_doc["scoring_context_source"] = ctx_tier
        # Persist a compact snapshot of what RAGAS scored against (first 3 passages,
        # truncated to 300 chars each) so the score is fully auditable.
        eval_doc["scoring_context_snippets"] = [p[:300] for p in contexts[:3]]

        rag_scores = score_ragas(query, answer, contexts)
        eval_doc["faithfulness"]     = _safe_float(rag_scores["faithfulness"])
        eval_doc["answer_relevance"] = _safe_float(rag_scores["answer_relevance"])
        eval_doc["scoring_method"]   = rag_scores["method"]

        sources = doc.get("sources", [])
        if sources:
            graph_count = sum(1 for s in sources if s.get("from_graph"))
            eval_doc["graph_coverage"] = round(graph_count / len(sources), 4)
        else:
            eval_doc["graph_coverage"] = None

    elif intent == "both":
        # HYBRID: answer draws from the live Sentries API *and* RAG context.
        #
        # The core problem: RAGAS faithfulness only checks whether answer claims
        # are grounded in retrieved passages.  When Sentries actually returned
        # data (timing["sentries"] > 0.1 s, or sentries_summary is non-empty),
        # the API-sourced facts are not in those passages → RAGAS assigns ~0.07
        # faithfulness and marks the response DPO_REJECTED, even though the
        # model gave a correct, grounded answer.  This is a systematic false
        # reject that poisons training data.
        #
        # Fix — three-path logic keyed on whether Sentries contributed:
        #
        #   Path A  sentries_summary non-empty (new schema ≥ v1.1):
        #     Most precise.  Validator cross-checks the answer against the
        #     actual Sentries payload.  RAGAS answer_relevance still runs
        #     (it is query↔answer cosine sim, not context-dependent).
        #
        #   Path B  sentries_summary absent but timing["sentries"] > 0.1 s
        #     (old schema): Sentries was invoked and probably returned data
        #     but the raw payload wasn't persisted.  Use validator (sources
        #     array only) + RAGAS AR, then take max(v_faith, ragas_faith).
        #
        #   Path C  sentries_summary == [] and timing["sentries"] ≈ 0:
        #     Sentries returned nothing → treat as pure RAG, full RAGAS.
        #
        # faithfulness = max(validator_faith, ragas_faith)
        #   Taking the maximum means: if either scoring path says the answer
        #   is faithful, we trust it.  This is the correct prior for a system
        #   that has two independent ground-truth sources.

        sentries_summary = doc.get("sentries_summary")          # list|None
        sentries_timing  = doc.get("timing", {}).get("sentries", 0)

        # Live (non-graph) Jira sources = sentries fetched this data directly
        live_jira_sources = [
            s for s in doc.get("sources", [])
            if s.get("source") == "jira" and not s.get("from_graph", False)
        ]

        # Expanded sentries contribution detection (was missing timing > 0 and
        # live Jira source signal — caused many correct hybrid answers to fall
        # to Path C and receive RAGAS-only faithfulness of 0)
        sentries_contributed = (
            bool(sentries_summary)                                    # non-empty list (most reliable)
            or (sentries_summary is None and sentries_timing > 0.1)  # old schema timing proxy
            or sentries_timing > 0                                    # any completed sentries call
            or bool(live_jira_sources)                                # live Jira data in sources
        )

        contexts, ctx_tier = retrieve_context(query, top_k=5, doc=doc)
        eval_doc["num_contexts"]             = len(contexts)
        eval_doc["scoring_context_source"]   = ctx_tier
        eval_doc["scoring_context_snippets"] = [p[:300] for p in contexts[:3]]
        rag_scores = score_ragas(query, answer, contexts)
        ragas_faith = _safe_float(rag_scores["faithfulness"])
        ragas_ar    = _safe_float(rag_scores["answer_relevance"])

        if sentries_contributed:
            # Path A / B — hybrid scoring
            # FLATTENED FIXES: (1) query= is now passed so the validator's
            # query-echo exemptions actually run (they were dead code — echoed
            # keys were penalised as fabrications → false hallucination tags);
            # (2) None summary is PRESERVED (was `or []`, which turned "payload
            # not persisted" into "sentries returned nothing" and armed the
            # fabricated_keys_on_empty_sentries penalty on old-schema docs).
            try:
                from hammer.validator import validate_sentry_answer
                v_score, v_details = validate_sentry_answer(
                    answer=answer,
                    sources=doc.get("sources", []),
                    sentries_summary=sentries_summary,
                    query=query,
                )
                eval_doc["validator_details"] = v_details
            except ImportError:
                v_score = 0.50
                eval_doc["validator_details"] = None

            # Take max: if either ground-truth source endorses the answer,
            # it is faithful.  Store both for auditability.
            eval_doc["faithfulness"]     = max(v_score, ragas_faith)
            eval_doc["answer_relevance"] = ragas_ar
            eval_doc["scoring_method"]   = f"hybrid_{rag_scores['method']}"
            eval_doc["validator_faith"]  = v_score      # audit
            eval_doc["ragas_faith"]      = ragas_faith  # audit
        else:
            # Path C — Sentries returned nothing → pure RAG
            # Faithfulness floor: intent="both" means the intent classifier
            # expected multi-source data.  When RAGAS returns near-zero faithfulness
            # but answer_relevance is high, this is almost certainly a context-miss
            # (API facts not in RAG context) rather than hallucination.  Apply a
            # conservative floor to prevent false DPO rejection of correct answers.
            # Only fires when ragas_faith < 0.15 (clear RAGAS context-miss) AND
            # ragas_ar > 0.60 (answer is demonstrably relevant to the query).
            faithfulness_floor_applied = False
            if ragas_faith < 0.15 and ragas_ar > 0.60:
                ragas_faith = max(ragas_faith, 0.30)
                faithfulness_floor_applied = True

            eval_doc["faithfulness"]              = ragas_faith
            eval_doc["answer_relevance"]          = ragas_ar
            eval_doc["scoring_method"]            = rag_scores["method"]
            eval_doc["faithfulness_floor_applied"] = faithfulness_floor_applied  # audit

        # Graph coverage audit metric (same for all both-paths)
        sources = doc.get("sources", [])
        if sources:
            graph_count = sum(1 for s in sources if s.get("from_graph"))
            eval_doc["graph_coverage"] = round(graph_count / len(sources), 4)
        else:
            eval_doc["graph_coverage"] = None

    elif intent == "sentries":
        # Deterministic validator replaces RAGAS for sentries-only responses.
        # Per the report: "Do not run RAGAS on sentries-only responses."
        try:
            from hammer.validator import validate_sentry_answer, entity_match_relevance
            # FLATTENED FIXES: query= (echo exemptions live) and
            # sentries_summary= (the most precise check — summary coverage —
            # previously NEVER ran for pure-sentries docs).
            v_score, v_details = validate_sentry_answer(
                answer=answer,
                sources=doc.get("sources", []),
                sentries_summary=doc.get("sentries_summary"),
                query=query,
            )
            eval_doc["faithfulness"]      = v_score   # validator score → faithfulness slot
            # entity_match_relevance replaces the broken 0.50 hardcode:
            # checks whether the query's key entities (Jira keys, project names,
            # usernames) appear in the answer — a real relevance signal.
            eval_doc["answer_relevance"]  = entity_match_relevance(query, answer)
            eval_doc["scoring_method"]    = "validator"
            eval_doc["validator_details"] = v_details
        except ImportError:
            eval_doc["faithfulness"]     = 0.50
            eval_doc["answer_relevance"] = 0.50
            eval_doc["scoring_method"]   = "validator_unavailable"

    else:
        # Unknown intent — conservative neutral
        eval_doc["faithfulness"]     = 0.50
        eval_doc["answer_relevance"] = 0.50
        eval_doc["scoring_method"]   = "unknown_intent"

    # ── Aggregate ─────────────────────────────────────────────────────────────
    weighted = compute_weighted_score(
        faithfulness    = eval_doc.get("faithfulness",    0.50),
        answer_relevance= eval_doc.get("answer_relevance",0.50),
        temporal        = temporal,
        code            = code,
    )
    eval_doc["weighted_score"]  = weighted

    # ── 2D classification: failure tags + quality grade ──────────────────────
    # Tags are derived first (they may include validator-driven and
    # source-driven signals); the grade then consumes both score and tags.
    failure_tags = compute_failure_tags(eval_doc, doc)
    eval_doc["failure_tags"]  = failure_tags
    eval_doc["quality_grade"] = compute_quality_grade(eval_doc, failure_tags)

    # Backward-compatible single-axis label — now DERIVED FROM quality_grade,
    # not from the raw score.  Previously this was computed from `weighted`
    # alone, which produced contradictions: a doc with score 0.895 but a
    # critical failure tag (e.g. tool_misuse) was graded FAILED yet labelled
    # qlora_positive, leaking failures into the training pool.  Deriving from
    # the grade keeps the two axes consistent.
    eval_doc["training_signal"] = _grade_to_signal(
        eval_doc["quality_grade"],
        failure_tags=failure_tags,
        score=eval_doc.get("weighted_score", 1.0),
    )

    return eval_doc


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluator(
    batch_size:    int  = 20,
    force_rescore: bool = False,
    limit:         Optional[int] = None,
    dry_run:       bool = False,
    workers:       int  = 4,
) -> Dict[str, Any]:
    """
    Main evaluation loop.

    Args:
        batch_size:    Log progress every N documents.
        force_rescore: Re-score documents that already have an evaluation.
        limit:         Max documents to process (None = all unscored).
        dry_run:       Evaluate but do not write back to MongoDB.
        workers:       ThreadPoolExecutor workers (default 4). The evaluator is
                       I/O-bound (MongoDB reads + Neo4j hybrid queries) so
                       parallelism gives near-linear speedup up to ~8 workers
                       without GPU contention.

    Returns:
        Summary dict: {processed, failed, avg_score, qlora_count, dpo_count}.
    """
    import concurrent.futures

    db         = _get_db()
    collection = db["chat_history"]

    query_filter: Dict[str, Any] = {
        "answer": {"$exists": True},
        "error":  None,
    }
    if not force_rescore:
        query_filter["evaluation"] = {"$exists": False}

    total = collection.count_documents(query_filter)
    if limit:
        total = min(total, limit)

    logger.info("=" * 60)
    logger.info("Hammer Evaluator  v1.2")
    logger.info("=" * 60)
    logger.info("Documents to evaluate : %d", total)
    logger.info("Force rescore         : %s", force_rescore)
    logger.info("Dry run               : %s", dry_run)
    if limit:
        logger.info("Limit                 : %d", limit)
    logger.info("")

    if total == 0:
        logger.info("Nothing to evaluate. All documents already scored.")
        return {"processed": 0, "failed": 0, "avg_score": None}

    cursor = collection.find(query_filter)
    if limit:
        cursor = cursor.limit(limit)

    docs = list(cursor)   # materialise so we can fan out

    processed = 0
    failed    = 0
    scores:   List[float] = []
    signals:  Dict[str, int] = {"qlora_positive": 0, "dpo_rejected": 0, "neutral": 0}
    grades:   Dict[str, int] = {"GOLD": 0, "SILVER": 0, "BRONZE": 0, "FAILED": 0}
    tag_counts: Dict[str, int] = {}

    def _process_doc(doc):
        """Worker: evaluate one doc; returns (eval_doc, doc_id) or raises."""
        return evaluate_document(doc), doc["_id"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_doc, doc): doc for doc in docs}

        for future in concurrent.futures.as_completed(futures):
            try:
                eval_doc, doc_id = future.result()
                weighted = eval_doc["weighted_score"]
                signal   = eval_doc["training_signal"]
                grade    = eval_doc.get("quality_grade", "FAILED")
                tags     = eval_doc.get("failure_tags", [])

                if not dry_run:
                    collection.update_one(
                        {"_id": doc_id},
                        {"$set": {"evaluation": eval_doc}},
                    )

                scores.append(weighted)
                signals[signal] = signals.get(signal, 0) + 1
                grades[grade]   = grades.get(grade, 0) + 1
                for t in tags:
                    tag_counts[t] = tag_counts.get(t, 0) + 1
                processed += 1

                if processed % batch_size == 0:
                    recent_avg = sum(scores[-batch_size:]) / min(batch_size, len(scores))
                    logger.info(
                        "Progress: %d/%d  |  score=%.3f  |  grade=%s  |  tags=%s  |  method=%s",
                        processed, total, recent_avg, grade,
                        ",".join(tags) if tags else "—",
                        eval_doc.get("scoring_method", "?"),
                    )

            except Exception as e:
                doc = futures[future]
                logger.error("Failed doc %s: %s", doc.get("_id"), e, exc_info=True)
                failed += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluation Complete")
    logger.info("=" * 60)
    logger.info("Processed    : %d  |  Failed: %d", processed, failed)
    logger.info("Avg score    : %.4f", avg_score)
    logger.info("")
    logger.info("Quality grade distribution:")
    for g in ("GOLD", "SILVER", "BRONZE", "FAILED"):
        n = grades.get(g, 0)
        pct = 100 * n / processed if processed else 0
        logger.info("  %-7s : %4d  (%.1f%%)", g, n, pct)
    if tag_counts:
        logger.info("")
        logger.info("Failure tag counts (non-zero):")
        for t, n in sorted(tag_counts.items(), key=lambda x: -x[1]):
            crit = " [CRITICAL]" if t in CRITICAL_TAGS else ""
            logger.info("  %-18s : %4d%s", t, n, crit)
    logger.info("")
    logger.info("Legacy training_signal (for backward compat):")
    logger.info("  QLoRA +      : %d  (≥ %.2f)", signals.get("qlora_positive", 0), THRESHOLD_QLORA)
    logger.info("  DPO rejects  : %d  (< %.2f)", signals.get("dpo_rejected",  0), THRESHOLD_DPO)
    logger.info("  Neutral      : %d", signals.get("neutral", 0))

    # ── Record baseline in model_versions on first run ────────────────────────
    if not dry_run and scores:
        versions = db["model_versions"]
        if versions.count_documents({"version": "baseline"}) == 0:
            versions.insert_one({
                "version":          "baseline",
                "training_method":  "none",
                "base_model":       OLLAMA_MODEL,
                "benchmark": {
                    "weighted_score": avg_score,
                    "num_samples":    len(scores),
                    "qlora_positive": signals.get("qlora_positive", 0),
                    "dpo_rejected":   signals.get("dpo_rejected",   0),
                },
                "deployed":    True,
                "deployed_at": datetime.utcnow(),
                "created_at":  datetime.utcnow(),
            })
            logger.info("✓ Baseline recorded in model_versions: %.4f", avg_score)
        else:
            # Update the rolling window score
            versions.update_one(
                {"version": "baseline"},
                {"$set": {
                    "benchmark.last_rolling_score": avg_score,
                    "benchmark.last_eval_at":       datetime.utcnow(),
                }},
            )

    summary = {
        "processed":      processed,
        "failed":         failed,
        "avg_score":      avg_score,
        # New 2D classification counts
        "grades":         dict(grades),
        "failure_tags":   dict(tag_counts),
        # Backward-compat
        "qlora_count":    signals.get("qlora_positive", 0),
        "dpo_count":      signals.get("dpo_rejected",   0),
        "neutral_count":  signals.get("neutral", 0),
    }
    return summary


# ─── Rolling window quality monitor (for continuous Hammer runs) ──────────────

def rolling_quality_check(window: int = 50, alert_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Compute rolling weighted_score over last `window` responses.
    Compares to baseline and raises an alert dict if regression > alert_threshold.

    Called by run_hammer every 6 hours (or on demand).
    """
    db         = _get_db()
    collection = db["chat_history"]

    recent = list(
        collection.find(
            {"evaluation.weighted_score": {"$exists": True}},
            {"evaluation.weighted_score": 1, "_id": 0},
        ).sort("timestamp", -1).limit(window)
    )

    if not recent:
        return {"status": "no_data"}

    scores = [r["evaluation"]["weighted_score"] for r in recent]
    rolling_avg = round(sum(scores) / len(scores), 4)

    # Compare to baseline
    baseline_doc = db["model_versions"].find_one({"version": "baseline"})
    baseline_score = (
        baseline_doc["benchmark"]["weighted_score"]
        if baseline_doc
        else None
    )

    regression = None
    should_alert = False
    if baseline_score is not None:
        regression   = round(baseline_score - rolling_avg, 4)
        should_alert = regression > alert_threshold

    result = {
        "window":        window,
        "rolling_avg":   rolling_avg,
        "baseline":      baseline_score,
        "regression":    regression,
        "should_alert":  should_alert,
        "checked_at":    datetime.utcnow().isoformat(),
    }

    if should_alert:
        logger.warning(
            "⚠ QUALITY REGRESSION DETECTED: rolling=%.4f  baseline=%.4f  drop=%.4f",
            rolling_avg, baseline_score, regression,
        )
        # Optionally write to openclaw_alerts collection
        try:
            db["openclaw_alerts"].insert_one({
                **result,
                "alert_type": "quality_regression",
                "created_at": datetime.utcnow(),
            })
        except Exception:
            pass

    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hammer Evaluator")
    parser.add_argument("--limit",  type=int,  default=None,  help="Max docs to evaluate")
    parser.add_argument("--batch",  type=int,  default=20,    help="Batch size for logging")
    parser.add_argument("--force",  action="store_true",      help="Re-score already-scored docs")
    parser.add_argument("--dry-run",action="store_true",      help="Evaluate but do not write to MongoDB")
    parser.add_argument("--check",  action="store_true",      help="Rolling quality check only (no scoring)")
    args = parser.parse_args()

    if args.check:
        result = rolling_quality_check()
        import json
        print(json.dumps(result, indent=2, default=str))
    else:
        run_evaluator(
            batch_size=args.batch,
            force_rescore=args.force,
            limit=args.limit,
            dry_run=args.dry_run,
        )

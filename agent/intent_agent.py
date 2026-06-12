#!/usr/bin/env python3
"""
intent_agent_test.py  —  OPTIMIZED Intent Agent  (A/B test variant)
===================================================================
Drop-in for intent_agent.py — same graph, same public API (ask, get_graph,
build_graph, main). Wires in the *_test modules when present and falls back
to the originals, so it runs even if you delete some test files.

WHAT CHANGED vs intent_agent.py  (full write-up: AGENT_SENTRIES_DIAGNOSIS.md)
──────────────────────────────────────────────────────────────────────────────
1. NO-OP SCORE FILTER FIXED (bug).  rag_node computed
       filtered = [s for s in sources if score >= RAG_MIN_SCORE]
   …and then never used `filtered` — the threshold did nothing, and ALL
   retrieved sources (top_k=8, noise included) flowed into the weaver prompt.
   Now the filter is applied, plus a RAG_MAX_SOURCES cap (default 6).
   Both env-tunable; RAG_MIN_SCORE=0 disables.

2. gc/CUDA CLEAR THROTTLED (latency).  gpu_clear_node ran gc.collect() —
   with a loaded embedding model heap that's tens to hundreds of ms — plus
   torch.cuda.empty_cache() (device sync, and the freed VRAM cache must be
   re-allocated on the next embed) on EVERY query, in the hot path between
   retrieval and generation. Now controlled by GPU_CLEAR_EVERY_N
   (default 0 = off; N>0 runs the clear every N-th query).

3. MONGO PERSIST OFF THE HOT PATH (latency).  chat_history insert_one was
   synchronous at end-of-turn; on a slow/remote Mongo it visibly delays the
   prompt returning. Now fire-and-forget in a daemon thread.

4. CONSISTENT JIRA-KEY REGEX.  {2,8} here vs {2,5} in the classifier vs
   {2,8} in sentries — unified at {2,9} (also catches LONGKEY-12355-style keys).

5. _infer_project GitLab host from env.  The URL regex hardcoded gitlab.com —
   useless on self-hosted GitLab. Host now derived from GITLAB_URL.

6. Uses rag_generator_test (cross-encoder reranked retrieval + direct
   ticket fast-path), weaver_node_test (single prompt builder + sentries
   context cap), intent_classifier_test, chatbot_sentries_test (fan-out
   caps) — each with automatic fallback to the original module.
"""

from __future__ import annotations

import gc
import os
import re
import sys
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# ── Path fix: ensure agent/ and sibling packages are always importable ─────────
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_AGENT_DIR)

for _p in [
    _AGENT_DIR,
    _ROOT_DIR,
    os.path.join(_ROOT_DIR, "sentries"),
    os.path.join(_ROOT_DIR, "rag"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── LangGraph ──────────────────────────────────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    from typing import TypedDict  # type: ignore[assignment]

CHECKPOINTER_AVAILABLE = False

# ── Internal modules — prefer *_test variants, fall back to originals ──────────
# SWAPPED-IN BUILD: canonical names hold optimized code; *_test = parked originals.
try:
    from intent_classifier import classify_intent
    CLASSIFIER_AVAILABLE = True
    _CLASSIFIER_IMPL = "intent_classifier (optimized, swapped-in)"
except ImportError:
    try:
        from intent_classifier_test import classify_intent
        CLASSIFIER_AVAILABLE = True
        _CLASSIFIER_IMPL = "intent_classifier_test (parked original)"
    except ImportError:
        CLASSIFIER_AVAILABLE = False
        _CLASSIFIER_IMPL = "none"

try:
    from weaver_node import weave, weave_stream
    WEAVER_AVAILABLE = True
    _WEAVER_IMPL = "weaver_node (optimized, swapped-in)"
except ImportError:
    try:
        from weaver_node_test import weave, weave_stream
        WEAVER_AVAILABLE = True
        _WEAVER_IMPL = "weaver_node_test (parked original)"
    except ImportError:
        WEAVER_AVAILABLE = False
        _WEAVER_IMPL = "none"

try:
    from rag_generator import RAGGenerator
    RAG_AVAILABLE = True
    _RAG_IMPL = "rag_generator (optimized, swapped-in)"
except ImportError:
    try:
        from rag_generator_test import RAGGenerator
        RAG_AVAILABLE = True
        _RAG_IMPL = "rag_generator_test (parked original)"
    except ImportError:
        RAG_AVAILABLE = False
        _RAG_IMPL = "none"

try:
    from sentry_dispatcher import SentryDispatcher
    try:
        from chatbot_sentries import (
            ProjectRegistry,
            Context as SentryContext,
            route_query,
            is_pure_count,
            count_results,
        )
        _SENTRIES_IMPL = "chatbot_sentries (optimized overlay, swapped-in)"
    except ImportError:
        from chatbot_sentries_test import (
            ProjectRegistry,
            Context as SentryContext,
            route_query,
            is_pure_count,
            count_results,
        )
        _SENTRIES_IMPL = "chatbot_sentries_test (parked original)"
    SENTRIES_AVAILABLE = True
except ImportError:
    SENTRIES_AVAILABLE = False
    _SENTRIES_IMPL = "none"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("intent_agent_test")
logger.info(
    "impl: classifier=%s weaver=%s rag=%s sentries=%s",
    _CLASSIFIER_IMPL, _WEAVER_IMPL, _RAG_IMPL, _SENTRIES_IMPL,
)

# ── Tunables (new) ─────────────────────────────────────────────────────────────
RAG_MIN_SCORE    = float(os.getenv("RAG_MIN_SCORE",    "0.20"))  # 0 disables
RAG_MAX_SOURCES  = int(os.getenv("RAG_MAX_SOURCES",    "6"))
RAG_TOP_K        = int(os.getenv("RAG_TOP_K",          "8"))
GPU_CLEAR_EVERY_N = int(os.getenv("GPU_CLEAR_EVERY_N", "0"))     # 0 = off

# Unified Jira issue-key pattern ({2,9} prefix — matches classifier + sentries)
_JIRA_KEY_RE = re.compile(r'\b[A-Z][A-Z0-9]{1,8}-\d+\b')

# ── Conversation history (in-memory) ───────────────────────────────────────────
_conversation_history: List[Dict[str, Any]] = []
HISTORY_MAX_TURNS = 3

# ── MongoDB — lazy, fail-silent ────────────────────────────────────────────────
try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

_mongo_collection = None
_mongo_client     = None

def _get_mongo_client():
    global _mongo_client
    if _mongo_client is not None:
        return _mongo_client
    if not MONGO_AVAILABLE:
        return None
    try:
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        _mongo_client.server_info()
    except Exception as exc:
        logger.warning("MongoDB unavailable: %s", exc)
        _mongo_client = None
    return _mongo_client


def _get_mongo_collection():
    global _mongo_collection
    if _mongo_collection is not None:
        return _mongo_collection
    client = _get_mongo_client()
    if client is None:
        return None
    try:
        db = os.getenv("MONGO_DB", "knowledge_base")
        _mongo_collection = client[db]["chat_history"]
        logger.info("MongoDB chat_history collection ready (db=%s)", db)
    except Exception as exc:
        logger.warning("MongoDB chat_history unavailable: %s", exc)
        _mongo_collection = None
    return _mongo_collection


class AgentState(TypedDict, total=False):
    query:                str
    intent:               str
    rag_result:           Optional[Dict[str, Any]]
    sentries_queries:     Optional[List[Dict[str, Any]]]
    sentries_results:     Optional[List[Any]]
    final_answer:         str
    gpu_cleared:          bool
    error:                Optional[str]
    timing:               Dict[str, float]
    conversation_history: Optional[List[Dict[str, Any]]]
    context_snippets:     Optional[List[str]]
    sentry_fallback:      bool


# ══════════════════════════════════════════════════════════════════════════════
# Node: intent_node
# ══════════════════════════════════════════════════════════════════════════════

def intent_node(state: AgentState) -> AgentState:
    import time
    t0 = time.monotonic()

    query = state.get("query", "").strip()
    if not query:
        logger.warning("intent_node received empty query")
        state["intent"] = "both"
        return state

    if CLASSIFIER_AVAILABLE:
        decision = classify_intent(query, use_llm_fallback=False)
    else:
        logger.warning("intent_classifier not available — defaulting to 'both'")
        decision = "both"

    logger.info("intent_node: %r → %s", query[:60], decision)

    timing = state.get("timing") or {}
    timing["intent"] = time.monotonic() - t0
    state["timing"] = timing
    state["intent"] = decision
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Node: rag_node
# ══════════════════════════════════════════════════════════════════════════════

_rag_instance: Optional[Any] = None

def _get_rag() -> Optional[Any]:
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance
    if not RAG_AVAILABLE:
        return None
    try:
        _rag_instance = RAGGenerator()
        return _rag_instance
    except Exception as exc:
        logger.error("RAGGenerator init failed: %s", exc)
        return None


# Monkey-patch: force think:False on any Ollama call (fallback path only)
import contextlib
import requests as _requests

@contextlib.contextmanager
def _no_think_ollama():
    _original_post = _requests.post
    OLLAMA_HOST_PREFIX = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip().rstrip("/")

    def _patched_post(url, *args, **kwargs):
        if isinstance(url, str) and url.startswith(OLLAMA_HOST_PREFIX):
            body = kwargs.get("json")
            if isinstance(body, dict):
                body.setdefault("think", False)
                kwargs["json"] = body
        return _original_post(url, *args, **kwargs)

    _requests.post = _patched_post
    try:
        yield
    finally:
        _requests.post = _original_post


def rag_node(state: AgentState) -> AgentState:
    """
    Retrieve relevant sources — no internal LLM generation (weaver synthesises).

    FIXES vs original:
      • RAG_MIN_SCORE filter is now actually applied (was computed and dropped)
      • RAG_MAX_SOURCES cap protects the weaver context budget
    """
    import time
    t0 = time.monotonic()

    query = state.get("query", "")
    logger.info("rag_node: retrieving for %r", query[:60])

    rag = _get_rag()
    if rag is None:
        state["rag_result"] = None
        state["error"] = "RAG system not available (check dependencies)"
        timing = state.get("timing") or {}
        timing["rag"] = time.monotonic() - t0
        state["timing"] = timing
        return state

    try:
        if hasattr(rag, "retrieve"):
            # In "both" mode the live ticket comes from Sentries; strip keys so
            # RAG focuses on the docs side. ({2,9} — was {2,8})
            intent = state.get("intent", "")
            if intent == "both":
                rag_query = _JIRA_KEY_RE.sub("", query).strip() or query
            else:
                rag_query = query

            sources = rag.retrieve(query=rag_query, top_k=RAG_TOP_K)

            # Deduplicate by title first, URL as fallback
            seen, unique = set(), []
            for s in sources:
                key = s.get("title") or s.get("url", "")
                if key and key not in seen:
                    seen.add(key)
                    unique.append(s)
            sources = unique

            # FIX: threshold is now enforced (original computed it, then
            # passed the UNFILTERED list to the weaver).
            if RAG_MIN_SCORE > 0:
                filtered = [s for s in sources if s.get("score", 0) >= RAG_MIN_SCORE]
                if not filtered:
                    logger.info(
                        "rag_node: all %d sources below threshold %.2f — no usable context",
                        len(sources), RAG_MIN_SCORE,
                    )
                sources = filtered

            # FIX: cap source count so weaver context stays inside its budget.
            if RAG_MAX_SOURCES > 0:
                sources = sources[:RAG_MAX_SOURCES]

            context_snippets = [s.get("text", "") for s in sources if s.get("text")]
            state["context_snippets"] = context_snippets

            result = {
                "answer":           "",
                "sources":          sources,
                "query":            query,
                "num_sources":      len(sources),
                "context_snippets": context_snippets,
            }
            logger.info("rag_node (retrieve): %d sources after dedup+filter+cap", len(sources))

        else:
            logger.info("rag_node: no retrieve() — using answer_question() with think:False")
            with _no_think_ollama():
                result = rag.answer_question(query=query, top_k=5)
            result["answer"] = ""
            context_snippets = [s.get("text", "") for s in result.get("sources", []) if s.get("text")]
            result["context_snippets"] = context_snippets
            state["context_snippets"] = context_snippets

        state["rag_result"] = result

    except Exception as exc:
        logger.error("rag_node failed: %s", exc)
        state["rag_result"] = None
        state["error"] = f"RAG error: {exc}"

    timing = state.get("timing") or {}
    timing["rag"] = time.monotonic() - t0
    state["timing"] = timing
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Node: sentries_node  — parallel dispatch via ThreadPoolExecutor
# ══════════════════════════════════════════════════════════════════════════════

def _gitlab_host_pattern() -> str:
    """Derive the GitLab host for URL sniffing from env (was hardcoded gitlab.com)."""
    raw = os.getenv("GITLAB_URL", "https://gitlab.com")
    host = urlparse(raw if "://" in raw else f"https://{raw}").netloc or "gitlab.com"
    return re.escape(host)


def _infer_project(query: str, history: List[Dict[str, Any]]) -> Optional[str]:
    """Infer GitLab 'owner/project' from URLs in history or query + env owner."""
    url_re     = re.compile(_gitlab_host_pattern() + r'/([\w.-]+)/([\w.-]+)', re.I)
    project_re = re.compile(r'\bin\s+([\w-]+)\b', re.I)

    _STOPWORDS = {
        "the", "a", "an", "our", "this", "that", "my", "your", "its",
        "mr", "pr", "all", "any", "no", "not", "is", "it",
    }

    for turn in reversed(history):
        m = url_re.search(turn.get("answer", ""))
        if m:
            return f"{m.group(1)}/{m.group(2)}"

    pm = project_re.search(query)
    if pm:
        candidate = pm.group(1).lower()
        if candidate not in _STOPWORDS:
            owner = os.getenv("GITLAB_OWNER", "")
            if owner:
                return f"{owner}/{pm.group(1)}"

    return None


_dispatcher: Optional[Any] = None
_registry:   Optional[Any] = None
_sentry_ctx: Optional[Any] = None


def _get_dispatcher():
    global _dispatcher, _registry, _sentry_ctx
    if _dispatcher is not None:
        return _dispatcher, _registry, _sentry_ctx
    if not SENTRIES_AVAILABLE:
        return None, None, None
    try:
        _dispatcher  = SentryDispatcher(verbose=False)
        _registry    = ProjectRegistry()
        _registry.load(_dispatcher)
        _sentry_ctx  = SentryContext()
        logger.info("Sentry dispatcher ready (%s). Projects: %d",
                    _SENTRIES_IMPL, len(_registry.all_paths()))
        return _dispatcher, _registry, _sentry_ctx
    except Exception as exc:
        logger.error("Sentry dispatcher init failed: %s", exc)
        return None, None, None


def sentries_node(state: AgentState) -> AgentState:
    """Parallel sentry dispatch (unchanged), fan-outs capped via registry overlay."""
    import time
    t0 = time.monotonic()

    query = state.get("query", "")
    logger.info("sentries_node: routing %r", query[:60])

    dispatcher, registry, ctx = _get_dispatcher()
    if dispatcher is None:
        state["sentries_queries"] = []
        state["sentries_results"] = []
        if not state.get("error"):
            state["error"] = "Sentry dispatcher not available (check credentials)"
        timing = state.get("timing") or {}
        timing["sentries"] = time.monotonic() - t0
        state["timing"] = timing
        return state

    try:
        queries = list(route_query(query, registry, ctx.as_dict()) or [])

        # Jira ticket priority injection ({2,9} — was {2,8})
        jira_match = _JIRA_KEY_RE.search(query.upper())
        if jira_match:
            issue_key = jira_match.group(0)
            if not any(q.get("params", {}).get("issue_key") == issue_key for q in queries):
                queries.append({
                    "source": "jira",
                    "operation": "get_issue",
                    "params": {"issue_key": issue_key},
                })
                logger.info("sentries_node: injected priority get_issue for %s", issue_key)

        # MR diff injection
        mr_match = re.search(r'\bMR\s*#(\d+)\b', query, re.IGNORECASE)
        if mr_match:
            mr_iid  = int(mr_match.group(1))
            history = state.get("conversation_history") or []
            project = _infer_project(query, history)
            if project:
                diff_q = {
                    "source":    "gitlab",
                    "operation": "get_mr_diff",
                    "params":    {"project": project, "mr_iid": mr_iid},
                }
                already = any(
                    q.get("operation") == "get_mr_diff" and
                    q.get("params", {}).get("mr_iid") == mr_iid and
                    q.get("params", {}).get("project") == project
                    for q in queries
                )
                if not already:
                    queries.append(diff_q)
                    logger.info("sentries_node: injected get_mr_diff for %s MR #%d", project, mr_iid)

        if not queries:
            logger.info("sentries_node: no API calls routed for this query")
            state["sentries_queries"] = []
            state["sentries_results"] = []
        else:
            max_workers = min(len(queries), 6)
            results = [None] * len(queries)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(dispatcher.dispatch, q): i
                    for i, q in enumerate(queries)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        logger.error("sentries_node: dispatch failed for query[%d]: %s", idx, exc)
                        results[idx] = None

            results = [r for r in results if r is not None]
            ctx.update(queries)
            total_items = sum(len(r.data) for r in results if r.success)
            logger.info("sentries_node: %d calls → %d items (parallel)", len(queries), total_items)

            state["sentries_queries"] = queries
            state["sentries_results"] = results

    except Exception as exc:
        logger.error("sentries_node failed: %s", exc)
        state["sentries_queries"] = []
        state["sentries_results"] = []
        state["error"] = f"Sentries error: {exc}"

    timing = state.get("timing") or {}
    timing["sentries"] = time.monotonic() - t0
    state["timing"] = timing
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Node: rag_fallback_node
# ══════════════════════════════════════════════════════════════════════════════

def _sentries_have_data(state: AgentState) -> bool:
    results = state.get("sentries_results") or []
    return any(
        getattr(r, "success", False) and getattr(r, "data", None)
        for r in results
    )


def rag_fallback_node(state: AgentState) -> AgentState:
    logger.info(
        "rag_fallback_node: sentries returned no data for %r — running RAG",
        state.get("query", "")[:60],
    )
    state["sentry_fallback"] = True
    state = rag_node(state)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Node: gpu_clear_node — THROTTLED (was: full gc + CUDA cache clear EVERY query)
# ══════════════════════════════════════════════════════════════════════════════

_gpu_clear_counter = 0

def gpu_clear_node(state: AgentState) -> AgentState:
    import time
    t0 = time.monotonic()
    global _gpu_clear_counter
    _gpu_clear_counter += 1

    if GPU_CLEAR_EVERY_N > 0 and _gpu_clear_counter % GPU_CLEAR_EVERY_N == 0:
        logger.debug("gpu_clear_node: running gc (turn %d)", _gpu_clear_counter)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("gpu_clear_node: CUDA cache cleared")
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("gpu_clear_node: CUDA clear failed (%s)", exc)
        state["gpu_cleared"] = True
    else:
        # FIX: skipped by default. gc.collect() with a loaded embedding model
        # heap + cuda.empty_cache() device sync cost real ms on EVERY query and
        # freed nothing that mattered (model tensors stay referenced; Ollama
        # runs in its own process). Set GPU_CLEAR_EVERY_N>0 to re-enable.
        state["gpu_cleared"] = False

    timing = state.get("timing") or {}
    timing["gpu_clear"] = time.monotonic() - t0
    state["timing"] = timing
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Node: weaver_node
# ══════════════════════════════════════════════════════════════════════════════

def weaver_node(state: AgentState) -> AgentState:
    """Synthesise a final answer, streaming tokens live to the terminal."""
    import time
    t0 = time.monotonic()

    query  = state.get("query", "")
    intent = state.get("intent", "both")

    if not WEAVER_AVAILABLE:
        msg = "ERROR: weaver module not available. Check weaver_node(.py/_test.py) is present."
        print(f"\n  ✗ {msg}\n")
        state["final_answer"] = msg
        return state

    # COUNT short-circuit: skip LLM for pure aggregation queries
    if SENTRIES_AVAILABLE and is_pure_count(query):
        sentries_results = state.get("sentries_results") or []
        if sentries_results:
            count_answer = count_results(query, sentries_results)
            print(f"\n📊 [COUNT]\n{chr(9472) * 70}\n{count_answer}")
            timing = state.get("timing") or {}
            timing["weaver"] = time.monotonic() - t0
            state["timing"]       = timing
            state["final_answer"] = count_answer
            return state

    icon_map = {"rag": "📚", "sentries": "📡", "both": "🔀", "rag_fallback": "🔄"}
    icon     = icon_map.get(intent, "🤖")

    if state.get("sentry_fallback"):
        print(f"\n{icon} [SENTRY FALLBACK → RAG]\n{chr(9472) * 70}")
        print("  ⚠  No live API data found — answering from knowledge base.\n")
    else:
        print(f"\n{icon} [{intent.upper()}]\n{chr(9472) * 70}")

    chunks: List[str]    = []
    showing_thinking     = False
    first_real_char      = True

    try:
        for chunk in weave_stream(
            query                = query,
            intent               = intent,
            rag_result           = state.get("rag_result"),
            sentries_results     = state.get("sentries_results"),
            sentries_queries     = state.get("sentries_queries"),
            conversation_history = state.get("conversation_history"),
        ):
            if chunk == "\x00THINKING":
                print("  ⟳ Thinking...", end="", flush=True)
                showing_thinking = True
                continue

            if chunk == "\x00DONE_THINKING":
                if showing_thinking:
                    print("\r" + " " * 20 + "\r", end="", flush=True)
                    showing_thinking = False
                continue

            if first_real_char:
                print("  ", end="", flush=True)
                first_real_char = False

            formatted = chunk.replace("\n", "\n  ")
            print(formatted, end="", flush=True)
            chunks.append(chunk)

    except KeyboardInterrupt:
        print(" [interrupted]", flush=True)

    if showing_thinking:
        print("\r" + " " * 20 + "\r", end="", flush=True)

    print()

    full_answer = "".join(chunks)
    state["final_answer"] = full_answer
    logger.info("weaver_node: streamed %d chars", len(full_answer))

    timing = state.get("timing") or {}
    timing["weaver"] = time.monotonic() - t0
    state["timing"] = timing
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Conditional routers (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def route_after_intent(state: AgentState) -> str:
    return state.get("intent", "both")

def route_after_rag(state: AgentState) -> str:
    if state.get("intent") == "both":
        return "sentries"
    return "gpu_clear"

def route_after_sentries(state: AgentState) -> str:
    if state.get("intent") != "sentries":
        return "gpu_clear"
    if not _sentries_have_data(state):
        logger.info(
            "route_after_sentries: no sentry data → rag_fallback (queries=%d)",
            len(state.get("sentries_queries") or []),
        )
        return "rag_fallback"
    return "gpu_clear"


# ══════════════════════════════════════════════════════════════════════════════
# Graph builder (unchanged topology)
# ══════════════════════════════════════════════════════════════════════════════

def build_graph():
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "langgraph is not installed. Run:\n"
            "  pip install langgraph langchain langchain-core"
        )

    graph = StateGraph(AgentState)
    graph.add_node("intent",       intent_node)
    graph.add_node("rag",          rag_node)
    graph.add_node("sentries",     sentries_node)
    graph.add_node("rag_fallback", rag_fallback_node)
    graph.add_node("gpu_clear",    gpu_clear_node)
    graph.add_node("weaver",       weaver_node)

    graph.set_entry_point("intent")
    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {"rag": "rag", "sentries": "sentries", "both": "rag"},
    )
    graph.add_conditional_edges(
        "rag",
        route_after_rag,
        {"sentries": "sentries", "gpu_clear": "gpu_clear"},
    )
    graph.add_conditional_edges(
        "sentries",
        route_after_sentries,
        {"gpu_clear": "gpu_clear", "rag_fallback": "rag_fallback"},
    )
    graph.add_edge("rag_fallback", "gpu_clear")
    graph.add_edge("gpu_clear",    "weaver")
    graph.add_edge("weaver",       END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# Fallback runner (no LangGraph)
# ══════════════════════════════════════════════════════════════════════════════

def run_without_langgraph(query: str) -> AgentState:
    logger.warning("LangGraph not available — running sequential fallback")
    state: AgentState = {
        "query": query, "intent": "both", "timing": {},
        "gpu_cleared": False, "final_answer": "", "error": None,
        "rag_result": None, "sentries_queries": None, "sentries_results": None,
        "sentry_fallback": False,
        "conversation_history": list(_conversation_history[-HISTORY_MAX_TURNS:]),
    }
    state = intent_node(state)
    intent = state.get("intent", "both")
    if intent in ("rag", "both"):
        state = rag_node(state)
    if intent in ("sentries", "both"):
        state = sentries_node(state)
        if intent == "sentries" and not _sentries_have_data(state):
            logger.info("run_without_langgraph: sentries empty → rag_fallback")
            state = rag_fallback_node(state)
    state = gpu_clear_node(state)
    state = weaver_node(state)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Background Mongo persist (FIX: was synchronous on the hot path)
# ══════════════════════════════════════════════════════════════════════════════

def _persist_turn_async(doc: Dict[str, Any]):
    def _insert():
        collection = _get_mongo_collection()
        if collection is None:
            return
        try:
            collection.insert_one(doc)
        except Exception as exc:
            logger.warning("Failed to persist turn to MongoDB: %s", exc)
    threading.Thread(target=_insert, daemon=True, name="mongo-persist").start()


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None and LANGGRAPH_AVAILABLE:
        _compiled_graph = build_graph()
        logger.info("Graph compiled (checkpointing removed)")
    return _compiled_graph


def ask(query: str) -> Dict[str, Any]:
    """Run the full pipeline; the answer streams to stdout inside weaver_node."""
    if not query or not query.strip():
        return {
            "answer": "Please provide a question.",
            "intent": "none", "sources": [], "timing": {}, "error": "empty query",
        }

    if LANGGRAPH_AVAILABLE:
        graph = get_graph()
        initial: AgentState = {
            "query": query.strip(), "timing": {}, "gpu_cleared": False,
            "final_answer": "", "error": None, "rag_result": None,
            "sentries_queries": None, "sentries_results": None,
            "conversation_history": list(_conversation_history[-HISTORY_MAX_TURNS:]),
        }
        final_state: AgentState = graph.invoke(initial)
    else:
        final_state = run_without_langgraph(query.strip())

    answer  = final_state.get("final_answer", "")
    intent  = final_state.get("intent", "unknown")
    timing  = final_state.get("timing", {})
    rag     = final_state.get("rag_result") or {}
    sources = rag.get("sources", [])

    if answer and not answer.startswith("ERROR"):
        turn = {
            "query":  query.strip(),
            "answer": answer[:800],
            "intent": intent,
        }
        _conversation_history.append(turn)

        # Build the persistence doc on the main thread (cheap), insert in
        # a daemon thread (FIX: insert_one was synchronous end-of-turn).
        sentries_results = final_state.get("sentries_results") or []
        sentries_summary = []
        for r in sentries_results:
            if getattr(r, "success", False) and getattr(r, "data", None):
                sentries_summary.append({
                    "source":    getattr(r, "source", ""),
                    "operation": getattr(r, "operation", ""),
                    "count":     len(r.data),
                    "titles": [
                        item.get("title") or item.get("key") or item.get("id", "")
                        for item in r.data[:5]
                    ],
                })
        doc = {
            "query":   query.strip(),
            "answer":  answer,
            "intent":  intent,
            "timing":  timing,
            "sources": [
                {
                    "source":         s.get("source", ""),
                    "title":          s.get("title", ""),
                    "url":            s.get("url", ""),
                    "project":        s.get("project", ""),
                    "issue_key":      s.get("issue_key", ""),
                    "content":        s.get("content", ""),
                    "from_graph":     s.get("from_graph", False),
                    "relationship":   s.get("relationship", ""),
                    "graph_distance": s.get("graph_distance", 0),
                    "score":          s.get("score", 0.0),
                    "status":         s.get("status", ""),
                    "priority":       s.get("priority", ""),
                    "assignee":       s.get("assignee", ""),
                }
                for s in sources
            ],
            "context_snippets": [
                {
                    "title":        s.get("title", ""),
                    "source":       s.get("source", ""),
                    "text":         (s.get("text") or "")[:500],
                    "from_graph":   s.get("from_graph", False),
                    "relationship": s.get("relationship", ""),
                }
                for s in sources[:5]
            ],
            "sentries_summary": sentries_summary,
            "error":            final_state.get("error"),
            "timestamp":        datetime.utcnow(),
            "total_time_s":     round(sum(timing.values()), 3),
            "pipeline":         "intent_agent_test",
        }
        _persist_turn_async(doc)

    return {
        "answer":  answer,
        "intent":  intent,
        "sources": sources,
        "timing":  timing,
        "error":   final_state.get("error"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

HEADER = """
╔══════════════════════════════════════════════════════════════════════════╗
║  🤖  UNIFIED AI ASSISTANT — Intent Agent + RAG + Sentries  [TEST BUILD]  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Routing:  🧠 Intent Classifier  →  📚 RAG  |  📡 Sentries  |  🔀 Both ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Commands:  /help  /timing  /impl  /clear  quit                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


def _print_footer(result: Dict[str, Any], show_timing: bool):
    sources = result.get("sources", [])
    error   = result.get("error")

    if sources:
        print(f"\n  📚 Sources ({len(sources)}):")
        for i, s in enumerate(sources[:3], 1):
            score = s.get("score", 0)
            title = s.get("title", "?")[:55]
            src   = s.get("source", "?").upper()
            print(f"    {i}. [{src}] {title}  (score {score:.3f})")

    if error:
        print(f"\n  ⚠  Warning: {error}")

    if show_timing:
        timing = result.get("timing", {})
        total  = sum(timing.values())
        parts  = " | ".join(f"{k}: {v:.2f}s" for k, v in timing.items())
        print(f"\n  ⏱  {parts}  |  total: {total:.2f}s")

    print("\n" + "═" * 70 + "\n")


def main():
    print(HEADER)
    print(f"  impl: classifier={_CLASSIFIER_IMPL}  weaver={_WEAVER_IMPL}")
    print(f"        rag={_RAG_IMPL}  sentries={_SENTRIES_IMPL}")
    print(f"  rag_min_score={RAG_MIN_SCORE}  rag_max_sources={RAG_MAX_SOURCES}  "
          f"gpu_clear_every_n={GPU_CLEAR_EVERY_N}\n")

    if not LANGGRAPH_AVAILABLE:
        print("  ⚠  LangGraph not installed — running in sequential fallback mode.")
        print("     Install: pip install langgraph langchain langchain-core\n")

    print("  Warming up...")
    if LANGGRAPH_AVAILABLE:
        get_graph()
        print("  ✓ Graph compiled")

    if RAG_AVAILABLE:
        print("  Loading embedding model...", end="", flush=True)
        _get_rag()
        print(" ✓")
    print()

    show_timing = False
    count       = 0

    while True:
        try:
            query = input("You › ").strip()
            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                print(f"\n👋 Goodbye! ({count} questions answered)\n")
                break

            if query.lower() == "/help":
                print("""
  /timing  Toggle timing display on/off
  /impl    Show which module implementations are active
  /history Show recent conversation history
  /clear   Clear screen and reset history
  quit     Exit
""")
                continue

            if query.lower() == "/impl":
                print(f"\n  classifier={_CLASSIFIER_IMPL}  weaver={_WEAVER_IMPL}")
                print(f"  rag={_RAG_IMPL}  sentries={_SENTRIES_IMPL}\n")
                continue

            if query.lower() == "/timing":
                show_timing = not show_timing
                print(f"  Timing display: {'ON ✓' if show_timing else 'OFF'}\n")
                continue

            if query.lower() == "/history":
                if not _conversation_history:
                    print("  (no history yet)\n")
                else:
                    for i, turn in enumerate(_conversation_history, 1):
                        print(f"  [{i}] You: {turn['query'][:80]}")
                        print(f"       Bot: {turn['answer'][:100]}...\n")
                continue

            if query.lower() == "/clear":
                _conversation_history.clear()
                print("\033[2J\033[H")
                print(HEADER)
                continue

            count += 1
            result = ask(query)
            _print_footer(result, show_timing)

        except KeyboardInterrupt:
            print(f"\n\n👋 Goodbye! ({count} questions answered)\n")
            break
        except Exception as exc:
            print(f"\n  ✗ Unexpected error: {exc}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
weaver_node_test.py  —  FIXED weaver  (A/B test variant)
========================================================
Drop-in for weaver_node.py — same public API: weave(...), weave_stream(...).

WHAT CHANGED vs weaver_node.py
───────────────────────────────
1. DUPLICATE DEFINITIONS REMOVED (bug).  The original defines _format_history
   AND _build_prompt TWICE; Python silently keeps only the second of each:
     • the careful 300-char-per-turn history cap in the first _format_history
       was DEAD CODE — the active version injected up to ~800 chars per stored
       turn × 3 turns of raw history into every prompt;
     • the first _build_prompt's RAG_CAP=3000 was dead; the active one used
       6000. Whichever was intended, two competing definitions is a landmine.
   This file has exactly ONE canonical definition of each.

2. SENTRIES CONTEXT CAP (new — faithfulness + latency).  RAG context was
   capped (6000 chars) but the sentries context was UNBOUNDED. A fan-out
   (30 items × many fields, diffs up to 4000 chars each) can exceed
   num_ctx=10240 tokens; Ollama then silently drops the OLDEST tokens —
   i.e. the instructions and the knowledge-base block at the top of the
   prompt — which is exactly the failure the original's own comments
   describe. New WEAVER_SENTRIES_CAP (default 12000 chars ≈ 3000 tokens)
   guarantees the prompt fits. Fewer wasted tokens also = faster generation.

3. ENV-TUNABLE BUDGETS.  WEAVER_RAG_CAP (default 6000 — same as the active
   original), WEAVER_SENTRIES_CAP (12000), WEAVER_HISTORY_CHAR_CAP (300/turn),
   WEAVER_NUM_CTX (10240). Defaults preserve current behaviour except where
   the fix is the point.

Streaming, <think> filtering, payload shape, stop tokens, prompts, and the
context formatters are otherwise IDENTICAL to the original.
"""

from __future__ import annotations

import json
import os
import re
import logging
import requests
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger("weaver")

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b").strip()
MAX_TOKENS   = int(os.getenv("WEAVER_MAX_TOKENS", "2010"))
TEMPERATURE  = float(os.getenv("WEAVER_TEMPERATURE", "0.2"))

# Context budgets (chars). ~3.5 chars/token rule of thumb.
RAG_CAP          = int(os.getenv("WEAVER_RAG_CAP",          "6000"))
SENTRIES_CAP     = int(os.getenv("WEAVER_SENTRIES_CAP",     "12000"))
HISTORY_CHAR_CAP = int(os.getenv("WEAVER_HISTORY_CHAR_CAP", "300"))
NUM_CTX          = int(os.getenv("WEAVER_NUM_CTX",          "10240"))

# ── MLflow adapter lookup (unchanged) ────────────────────────────────────────
_active_adapter_path: Optional[str] = None
_adapter_checked: bool = False

def _check_mlflow_adapter() -> Optional[str]:
    global _active_adapter_path, _adapter_checked
    if _adapter_checked:
        return _active_adapter_path
    _adapter_checked = True
    try:
        from training.pipeline import get_active_adapter
        _active_adapter_path = get_active_adapter()
        if _active_adapter_path:
            logger.info("MLflow Production adapter: %s", _active_adapter_path)
        else:
            logger.info("No MLflow Production adapter — using base Ollama model")
    except (ImportError, Exception) as e:
        logger.debug("MLflow adapter lookup skipped: %s", e)
    return _active_adapter_path

# Fields always suppressed in formatted context (unchanged).
_SKIP_FIELDS = {
    "text", "message",
    "snippet", "comments", "changelog", "embedding",
}

_RAG_ONLY_PROMPT = """\
/no_think
You are a helpful AI assistant for a software engineering team.
Answer the user's question by prioritizing the knowledge-base context below.
If the context doesn't contain enough information, say so clearly.
Be concise and cite which source the information came from.
Respond in english or french, depending on the language of the question.
{history_block}
Knowledge-Base Context:
{rag_context}

User Question: {query}

Answer:"""

_SENTRIES_ONLY_PROMPT = """\
/no_think
You are a helpful AI assistant for a software engineering team.
Answer the user's question by prioritizing the live API data retrieved below.
Include IDs, titles, states, authors, dates and URLs where available.
If the data is empty or an error occurred, say so clearly.
Use numbered lists for multiple items. Be concise.
Respond in english or french, depending on the language of the question.
{history_block}
Live API Data:
{sentries_context}

User Question: {query}

Answer:"""

_BOTH_PROMPT = """\
/no_think
You are a helpful AI assistant for a software engineering team.
Respond in english or french, depending on the language of the question.
{history_block}
--- KNOWLEDGE BASE (static docs, guides, historical context) ---
{rag_context}

--- LIVE API DATA (Jira / GitLab / Confluence — real-time, authoritative) ---
{sentries_context}

Instructions:
- The LIVE API DATA is the primary source of truth for tickets, issues, MRs, and status.
- If a ticket ID is mentioned and appears in the Live API Data, summarize it from there — do NOT say it is unavailable.
- Use the Knowledge Base for background, documentation, and historical context only.
- If the Live API Data is empty or shows an error for a specific item, say so explicitly.
- Include IDs, URLs, states, authors from the live data where available.
- Be concise and direct.

User Question: {query}

Answer:"""


# ---------------------------------------------------------------------------
# <think> filtering (unchanged)
# ---------------------------------------------------------------------------

class _ThinkFilter:
    """Strips <think>…</think> blocks from a stream of text chunks."""

    _OPEN  = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf: str           = ""
        self._in_think: bool     = False
        self.thinking_started: bool = False
        self.thinking_ended:   bool = False

    @property
    def _max_partial(self) -> int:
        return max(len(self._OPEN), len(self._CLOSE)) - 1  # 7

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        output: List[str] = []

        while True:
            if self._in_think:
                idx = self._buf.find(self._CLOSE)
                if idx == -1:
                    keep = self._max_partial
                    self._buf = self._buf[-keep:] if len(self._buf) > keep else self._buf
                    break
                self._buf           = self._buf[idx + len(self._CLOSE):]
                self._in_think      = False
                self.thinking_ended = True
            else:
                idx = self._buf.find(self._OPEN)
                if idx == -1:
                    keep     = self._max_partial
                    safe_end = len(self._buf) - keep
                    if safe_end > 0:
                        output.append(self._buf[:safe_end])
                        self._buf = self._buf[safe_end:]
                    break
                output.append(self._buf[:idx])
                self._buf             = self._buf[idx + len(self._OPEN):]
                self._in_think        = True
                self.thinking_started = True

        return "".join(output)

    def flush(self) -> str:
        if self._in_think:
            self._buf = ""
            return ""
        result    = self._buf
        self._buf = ""
        return result


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Ollama I/O (num_ctx now env-driven; otherwise unchanged)
# ---------------------------------------------------------------------------

def _build_payload(prompt: str, stream: bool, intent: str = "both") -> dict:
    if intent == "sentries":
        num_predict = 1200
    elif intent == "both":
        num_predict = 1000
    else:
        num_predict = 800
    return {
        "model":   OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  stream,
        "think":   False,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx":     NUM_CTX,
            "num_predict": num_predict,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        },
    }


def _ollama_generate(prompt: str, intent: str = "both") -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=_build_payload(prompt, stream=False, intent=intent),
            timeout=180,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        return _strip_think(raw)
    except requests.exceptions.ConnectionError:
        return "ERROR: Ollama is not reachable. Please run `ollama serve`."
    except requests.exceptions.Timeout:
        return "ERROR: Ollama took too long to respond. Try a shorter query."
    except Exception as exc:
        logger.error("Ollama generate failed: %s", exc)
        return f"ERROR: {exc}"


def _ollama_stream(prompt: str, intent: str = "both") -> Generator[str, None, None]:
    try:
        with requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=_build_payload(prompt, stream=True, intent=intent),
            stream=True,
            timeout=180,
        ) as resp:
            resp.raise_for_status()
            filt                   = _ThinkFilter()
            emitted_thinking_sig   = False
            erased_thinking_sig    = False

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    logger.warning("Unparseable stream line: %r", raw_line)
                    continue

                token = data.get("response", "")
                if token:
                    clean = filt.feed(token)

                    if filt.thinking_started and not emitted_thinking_sig:
                        emitted_thinking_sig = True
                        yield "\x00THINKING"

                    if filt.thinking_ended and not erased_thinking_sig:
                        erased_thinking_sig = True
                        yield "\x00DONE_THINKING"

                    if clean:
                        yield clean

                if data.get("done"):
                    tail = filt.flush()
                    if tail:
                        yield tail
                    break

    except requests.exceptions.ConnectionError:
        yield "ERROR: Ollama is not reachable. Please run `ollama serve`."
    except requests.exceptions.Timeout:
        yield "ERROR: Ollama took too long to respond. Try a shorter query."
    except Exception as exc:
        logger.error("Ollama stream failed: %s", exc)
        yield f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# Context formatters (unchanged logic)
# ---------------------------------------------------------------------------

def _get(obj: Any, key: str, default=None) -> Any:
    """Unified accessor for dict-based and dataclass-based sentry results."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _format_rag_context(rag_result: Optional[Dict[str, Any]]) -> str:
    if not rag_result:
        return "(no knowledge-base results)"
    sources = rag_result.get("sources", [])
    if not sources:
        return "(knowledge base returned no relevant documents)"
    parts = []
    for i, src in enumerate(sources, 1):
        meta_items = []
        if src.get("issue_key"):    meta_items.append(f"Key: {src['issue_key']}")
        if src.get("status"):       meta_items.append(f"Status: {src['status']}")
        if src.get("priority"):     meta_items.append(f"Priority: {src['priority']}")
        if src.get("assignee"):     meta_items.append(f"Assignee: {src['assignee']}")
        if src.get("project"):      meta_items.append(f"Project: {src['project']}")
        if src.get("url"):          meta_items.append(f"URL: {src['url']}")
        meta_str = "  |  ".join(meta_items)

        parts.append(
            f"[{i}] [{src.get('source','?').upper()}] {src.get('source_type','')} — {src.get('title','Untitled')}\n"
            f"    Score: {src.get('score',0):.3f}\n"
            + (f"    {meta_str}\n" if meta_str else "")
            + f"    {src.get('text','')[:600]}"
        )
    return "\n\n".join(parts)


def _format_sentries_context(
    sentries_results: Optional[List[Any]],
    sentries_queries: Optional[List[Dict]],
    is_fanout: bool = False,
) -> str:
    if not sentries_results:
        return "(no live API results)"
    parts = []
    item_limit = 3 if is_fanout else 30

    for result, _ in zip(sentries_results, sentries_queries or [{}] * len(sentries_results)):
        source    = _get(result, "source", "?")
        operation = _get(result, "operation", "?")
        success   = _get(result, "success", False)
        data      = _get(result, "data", []) or []
        error     = _get(result, "error", None)

        header = f"[{source.upper()} → {operation}]"

        if not success:
            if not is_fanout:
                parts.append(f"{header}\nError: {error}")
            continue
        if not data:
            if not is_fanout:
                parts.append(f"{header}\nNo results found.")
            continue

        is_file_op = (operation == "get_file")

        rows = []
        for item in data[:item_limit]:
            lines = []
            for k, v in item.items():
                if v is None or v == "" or v == []:
                    continue
                if k in _SKIP_FIELDS:
                    continue
                if k == "content" and not is_file_op:
                    continue
                val = str(v)
                if k == "diff":
                    cap = 4000
                elif k == "content":
                    cap = 2000
                    if len(val) > cap:
                        val = val[:cap] + "…"
                    val = "\n".join(
                        f"{i+1:>4}  {line}"
                        for i, line in enumerate(val.splitlines())
                    )
                    lines.append(f"  {k}:\n{val}")
                    continue
                elif k in ("description", "body"):
                    cap = 120 if is_fanout else 400
                else:
                    cap = 120 if is_fanout else 300
                if len(val) > cap:
                    val = val[:cap] + "…"
                lines.append(f"  {k}: {val}")
            if lines:
                rows.append("\n".join(lines))
        if rows:
            parts.append(f"{header}\n" + "\n---\n".join(rows))
            remaining = len(data) - item_limit
            if remaining > 0:
                parts.append(f"  … and {remaining} more items")

    return "\n\n".join(parts) if parts else "(live APIs returned no usable data)"


# ---------------------------------------------------------------------------
# History + prompt builder — ONE canonical definition of each (the fix)
# ---------------------------------------------------------------------------

def _format_history(conversation_history: Optional[List[Dict[str, Any]]]) -> str:
    """Format last N turns, each previous answer capped to avoid context blowup."""
    if not conversation_history:
        return ""
    lines = ["\nRecent conversation (for context on follow-up questions):"]
    for turn in conversation_history:
        lines.append(f"User: {turn['query']}")
        ans = turn.get("answer", "")
        if HISTORY_CHAR_CAP > 0 and len(ans) > HISTORY_CHAR_CAP:
            ans = ans[:HISTORY_CHAR_CAP] + "… (truncated)"
        lines.append(f"Assistant: {ans}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _cap(text: str, cap: int, label: str) -> str:
    if cap > 0 and len(text) > cap:
        return text[:cap] + f"\n… ({label} truncated for context budget)"
    return text


def _build_prompt(
    query: str,
    intent: str,
    rag_result: Optional[Dict[str, Any]],
    sentries_results: Optional[List[Any]],
    sentries_queries: Optional[List[Dict]],
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    is_fanout     = bool(sentries_results) and len(sentries_results) > 3
    rag_ctx       = _format_rag_context(rag_result)
    sentries_ctx  = _format_sentries_context(sentries_results, sentries_queries, is_fanout=is_fanout)
    history_block = _format_history(conversation_history)

    # NEW: sentries context is now budgeted too (was unbounded — could push the
    # whole instruction header out of num_ctx on fan-out queries).
    sentries_ctx = _cap(sentries_ctx, SENTRIES_CAP, "live API data")

    if intent == "both":
        rag_ctx = _cap(rag_ctx, RAG_CAP, "knowledge base")
        return _BOTH_PROMPT.format(
            rag_context=rag_ctx,
            sentries_context=sentries_ctx,
            query=query,
            history_block=history_block,
        )

    if intent == "rag":
        return _RAG_ONLY_PROMPT.format(rag_context=rag_ctx, query=query, history_block=history_block)

    # sentries
    return _SENTRIES_ONLY_PROMPT.format(sentries_context=sentries_ctx, query=query, history_block=history_block)


# ---------------------------------------------------------------------------
# Public API (unchanged signatures)
# ---------------------------------------------------------------------------

def weave(
    query: str,
    intent: str,
    rag_result: Optional[Dict[str, Any]] = None,
    sentries_results: Optional[List[Any]] = None,
    sentries_queries: Optional[List[Dict]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Blocking – returns the complete answer as a string."""
    prompt = _build_prompt(query, intent, rag_result, sentries_results, sentries_queries, conversation_history)
    logger.debug("Weaver invoking Ollama (intent=%s, prompt_len=%d)", intent, len(prompt))
    answer = _ollama_generate(prompt, intent=intent)
    logger.debug("Weaver answer length: %d chars", len(answer))
    return answer


def weave_stream(
    query: str,
    intent: str,
    rag_result: Optional[Dict[str, Any]] = None,
    sentries_results: Optional[List[Any]] = None,
    sentries_queries: Optional[List[Dict]] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> Generator[str, None, None]:
    """Streaming variant – yields text chunks as they arrive from Ollama."""
    prompt = _build_prompt(query, intent, rag_result, sentries_results, sentries_queries, conversation_history)
    logger.debug("Weaver streaming Ollama (intent=%s, prompt_len=%d)", intent, len(prompt))
    yield from _ollama_stream(prompt, intent=intent)


# ---------------------------------------------------------------------------
# Interactive streaming CLI  –  python weaver_node_test.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    BANNER = (
        "\n+==================================================+"
        "\n|        Weaver (OPTIMIZED test) - CLI             |"
        f"\n|  model : {OLLAMA_MODEL:<38} |"
        f"\n|  host  : {OLLAMA_HOST:<38} |"
        "\n|  type 'exit' or Ctrl-C to quit                   |"
        "\n+==================================================+"
    )
    print(BANNER)

    try:
        pong = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        pong.raise_for_status()
        available = [m["name"] for m in pong.json().get("models", [])]
        if OLLAMA_MODEL not in available:
            print(f"\n  !  '{OLLAMA_MODEL}' not found (available: {', '.join(available) or 'none'}).")
        else:
            print(f"\n  OK Ollama reachable — model '{OLLAMA_MODEL}' ready.\n")
    except Exception as e:
        print(f"\n  !  Cannot reach Ollama at {OLLAMA_HOST}: {e}\n")

    while True:
        try:
            query = input(" You › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            sys.exit(0)

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            sys.exit(0)

        print("\n Weaver › ", end="", flush=True)
        try:
            for chunk in weave_stream(query, intent="rag"):
                print(chunk, end="", flush=True)
        except KeyboardInterrupt:
            pass
        print("\n")

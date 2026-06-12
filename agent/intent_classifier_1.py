#!/usr/bin/env python3
"""
Intent Classifier
─────────────────
Uses a fast keyword heuristic first, then optionally falls back to
Ollama LLM for ambiguous queries.

Returns one of three routing decisions:
  - "rag"      → query needs knowledge-base / historical / doc context
  - "sentries" → query needs live API data (GitLab, Jira, Confluence)
  - "both"     → query needs both sources

Design:
  Keyword heuristic is deterministic and fast (~0 ms).
  LLM fallback is only called when heuristic returns "both" AND the query
  is short enough that either could apply.
"""

import re
import os
import json
import logging
import requests
from typing import Literal

logger = logging.getLogger("intent.classifier")

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3").strip()

RouteDecision = Literal["rag", "sentries", "both"]

# ─── Keyword signals ──────────────────────────────────────────────────────────

# Strong signals → SENTRIES (live/real-time data)
_SENTRY_STRONG = re.compile(
    r"\b("
    r"issue|issues|bug|bugs|ticket|tickets|task|tasks|"
    r"merge.?request|pull.?request|\bmr\b|\bpr\b|"
    r"pipeline|pipelines|ci|cd|build|builds|deploy|"
    r"commit|commits|changelog|"
    r"sprint|board|backlog|"
    r"milestone|milestones|"
    r"open|closed|in.?progress|wip|"
    r"gitlab|jira|confluence|"
    r"branch|repo|repository|"
    r"real.?time|live|current|latest|recent|"
    r"auth-service|ecommerce|notification.?service|"
    r"who.?assigned|assignee|reporter|"
    r"status|priority|label|"
    # ── French terms (accents optional — engineers often omit them) ───────
    r"fichier|fichiers|"                          # file / files
    r"branche|branches|"                          # branch / branches
    r"depot|dep[oô]t|"                           # repo / dépôt
    r"probl[eè]me|probleme|"                      # bug / issue
    r"t[aâ]che|tache|"                            # task / tâche
    r"bogue|"                                     # bug (FR)
    r"montre.?moi|affiche.?moi|affiche|montre|"  # show me
    r"liste.?moi|"                                # list me
    r"voir|"                                      # see / check
    r"statut|[eé]tat|"                            # status / état
    r"priorit[eé]|"                               # priorité
    r"assign[eé]|"                                # assigné
    r"ouvert|ferm[eé]|en.?cours|"                # open / closed / in-progress
    r"\b[A-Z]{2,5}-\d+\b"                        # Jira issue keys e.g. AUTH-12
    r")\b",
    re.I,
)

# Strong signals → RAG (knowledge-base / docs / historical)
_RAG_STRONG = re.compile(
    r"\b("
    r"how.?to|how.?do.?i|explain|what.?is|what.?are|analyze|analyse|"
    r"guide|guides|documentation|docs|tutorial|"
    r"architecture|design|pattern|best.?practices?|"
    r"authenticate|authentication|oauth|jwt|"
    r"implement|setup|configure|configuration|"
    r"api.?reference|schema|"
    r"traceback|solution|"
    r"meaning|definition|describe|summarize|"
    r"historical|"
    r"Summarize|"
    r"according.?to|based.?on|from.?docs|"
    # ── French terms (accents optional — engineers often skip them) ───────
    r"explique|expliques|expliquer|"              # explain / explique
    r"comment.?ca.?marche|comment.?fonctionne|"  # how does it work
    r"comment.?faire|"                            # how to do
    r"qu.?est.?ce.?que|c.?est.?quoi|"            # what is
    r"tutoriel|"                                  # tutorial
    r"configurer|"                                # configure
    r"impl[eé]menter|"                            # implement
    r"d[eé]finition|signification|"              # definition
    r"r[eé]soudre|"                               # resolve
    r"meilleures?.?pratiques?"                    # best practices
    r")\b",
    re.I,
)

# A named source-code file in the query (e.g. main.py, handler.ts) is a strong
# sentry signal — the user wants *live* file content, not cached docs.
# Detected separately so it boosts sentry_hits even when "explain" is also
# present, which forces intent → "both" instead of pure "rag".
# This fixes: "explain what main.py does" → both (not rag-only → stale cache).
_FILE_IN_QUERY_RE = re.compile(
    r"\b[\w_\-]+\.(py|js|ts|go|java|rb|rs|cpp|c|cs|sh|yaml|yml|toml|json|md)\b",
    re.I,
)

# Jira issue key pattern — if present, heavily favours sentries
_JIRA_ISSUE_KEY_RE = re.compile(r"\b[A-Z]{2,5}-\d+\b")

# Phrases that suggest BOTH sources are needed
_BOTH_SIGNALS = re.compile(
    r"\b("
    r"all.{0,20}(issue|bug|ticket|mr|commit)|"
    r"(issue|bug|ticket|mr|commit).{0,20}all|"
    r"summary|overview|report|dashboard|"
    r"related.?to|context.?for|background.?on|"
    r"tell.?me.?everything|give.?me.?all|"
    r"what.?do.?we.?know|what.?happened"
    r")\b",
    re.I,
)

# ─── LLM fallback prompt ──────────────────────────────────────────────────────

_CLASSIFY_PROMPT = """\
/no_think
You are a routing classifier for an AI assistant that has two data sources:
  1. RAG (knowledge base): static docs, guides, tutorials, code explanations, historical info
  2. SENTRIES (live APIs): real-time GitLab issues/MRs/pipelines/commits, Jira tickets, Confluence pages

Classify the following user query into exactly ONE category:
  - "rag"      → only needs knowledge-base / documentation
  - "sentries" → only needs live API data
  - "both"     → needs both

Respond with ONLY a JSON object: {{"route": "<category>"}}

User query: {query}

JSON response:"""


def _llm_classify(query: str) -> RouteDecision:
    """Ask Ollama to classify intent. Returns 'both' on failure."""
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": _CLASSIFY_PROMPT.format(query=query),
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 32, "num_ctx": 2560},
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return "both"
        raw = resp.json().get("response", "").strip()
        # Strip think tags if present
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Extract JSON
        match = re.search(r'\{.*?"route"\s*:\s*"(\w+)".*?\}', raw)
        if match:
            decision = match.group(1).lower()
            if decision in ("rag", "sentries", "both"):
                return decision  # type: ignore[return-value]
    except Exception as exc:
        logger.debug("LLM classify failed: %s", exc)
    return "both"


def classify_intent(query: str, use_llm_fallback: bool = True) -> RouteDecision:
    """
    Classify a user query into a routing decision.

    Algorithm:
      1. Count sentry-strong and rag-strong keyword matches.
      2. If one side clearly dominates → return that category.
      3. If both sides match → check for explicit "both" signals.
      4. If still ambiguous and LLM fallback enabled → ask Ollama.
      5. Default: "both" (conservative — never miss data).

    Args:
        query:            The raw user question.
        use_llm_fallback: Set False in tests / offline mode.

    Returns:
        "rag" | "sentries" | "both"
    """
    sentry_hits = len(_SENTRY_STRONG.findall(query))
    rag_hits    = len(_RAG_STRONG.findall(query))
    both_hits   = len(_BOTH_SIGNALS.findall(query))

    # A bare Jira issue key (e.g. AUTH-12) is an unambiguous sentry signal —
    # override weak RAG preambles like "What is the status of AUTH-12?"
    if _JIRA_ISSUE_KEY_RE.search(query):
        sentry_hits += 2  # boost so it dominates a single "what.?is" hit

    # EXCEPTION: if both a Jira key AND documentation/guide signals are present,
    # the user wants the live ticket AND the related docs simultaneously.
    # Force "both" so RAG always runs — the +2 key boost would otherwise push
    # sentry_hits past rag_hits and silently drop the documentation retrieval.
    # Example: "What is PROJ-18462 about and what does the SRT documentation say?"
    #           sentry_hits=4, rag_hits=1 → would return "sentries" → docs missed.
    if _JIRA_ISSUE_KEY_RE.search(query) and rag_hits > 0:
        logger.debug("classify: jira_key + rag_signal → forcing both")
        return "both"

    # A named source-code file (e.g. main.py, handler.ts) means the user wants
    # live file content from GitLab, not cached docs.  Boost sentry so that
    # "explain what main.py does" → both (fresh file + RAG context) and not
    # pure "rag" → stale knowledge-base hit.
    if _FILE_IN_QUERY_RE.search(query):
        sentry_hits += 2

    logger.debug(
        "classify: sentry=%d rag=%d both=%d query=%r",
        sentry_hits, rag_hits, both_hits, query[:80],
    )

    # Clear winner
    if sentry_hits > 0 and rag_hits == 0:
        return "sentries"
    if rag_hits > 0 and sentry_hits == 0:
        return "rag"

    # Both signals present
    if sentry_hits > 0 and rag_hits > 0:
        if both_hits > 0:
            return "both"
        # Whichever side has more matches wins; tie → both
        if sentry_hits > rag_hits + 1:
            return "sentries"
        if rag_hits > sentry_hits + 1:
            return "rag"
        # Close race — LLM decides
        if use_llm_fallback:
            return _llm_classify(query)
        return "both"

    # No strong signals — try LLM
    if use_llm_fallback:
        return _llm_classify(query)

    return "both"


# ─── CLI smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("How do I authenticate with OAuth?",                   "rag"),
        ("Show open issues in auth-service",                     "sentries"),
        ("What are the recent commits in ecommerce-backend?",    "sentries"),
        ("List merge requests for notification-service",         "sentries"),
        ("What's the AUTH-12 ticket about and how do I fix it?", "both"),
        ("Explain the API rate limiting and show current bugs",   "both"),
        ("What is our database schema?",                          "rag"),
        ("Show pipelines for ecommerce-backend",                  "sentries"),
    ]
    print(f"\n{'Query':<55} {'Expected':<10} {'Got':<10} {'✓?'}")
    print("─" * 85)
    ok = 0
    for q, expected in tests:
        result = classify_intent(q, use_llm_fallback=False)
        match  = "✓" if result == expected else "✗"
        if result == expected:
            ok += 1
        print(f"{q[:54]:<55} {expected:<10} {result:<10} {match}")
    print(f"\nKeyword-only accuracy: {ok}/{len(tests)}")
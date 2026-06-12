#!/usr/bin/env python3
"""
intent_classifier_test.py  —  OPTIMIZED classifier  (A/B test variant)
======================================================================
Drop-in for intent_classifier.py — same classify_intent(query, use_llm_fallback)
API and same return values ("rag" | "sentries" | "both").

WHAT CHANGED vs intent_classifier.py
─────────────────────────────────────
1. Removed hardcoded demo project names ("auth-service|ecommerce|
   notification.?service") from _SENTRY_STRONG. They are leftovers from a
   demo org: they never match your real projects (ExampleProduct / TM-*), and CAN
   mis-fire on unrelated words ("ecommerce best practices" → sentries).
   Project resolution belongs to ProjectRegistry at the sentries layer.

2. Jira issue-key regex unified to [A-Z]{2,9}-\\d+ (was {2,5}).
   The old pattern MISSED keys with 6-9 char prefixes (e.g. LONGKEY-12355),
   so "What is LONGKEY-12355 about and what do the docs say?" did not get the
   sentry boost or the jira_key+rag→both override — it could route "rag" and
   skip the live ticket entirely. {2,9} matches intent_agent and sentries.

3. LLM fallback prompt: num_predict 32→24, timeout 15→10s (it only ever needs
   to emit a tiny JSON object; shorter timeout caps worst-case added latency).

Everything else (signal tables, scoring algorithm, thresholds) is unchanged
so A/B differences are attributable to the two fixes above.
"""

import re
import os
import logging
import requests
from typing import Literal

logger = logging.getLogger("intent.classifier")

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3").strip()

RouteDecision = Literal["rag", "sentries", "both"]

# ─── Keyword signals ──────────────────────────────────────────────────────────

# Strong signals → SENTRIES (live/real-time data)
# CHANGE: hardcoded demo project names removed; jira key regex widened to {2,9}.
_SENTRY_STRONG = re.compile(
    r"\b("
    r"issue|issues|bug|bugs|ticket|tickets|task|tasks|"
    # CHANGE: requests? — the trailing \b made the singular form fail on
    # "merge requests"/"pull requests" (plural was previously only caught by
    # the hardcoded demo project names this build removes).
    r"merge.?requests?|pull.?requests?|\bmr\b|\bpr\b|"
    r"pipeline|pipelines|ci|cd|build|builds|deploy|"
    r"commit|commits|changelog|"
    r"sprint|board|backlog|"
    r"milestone|milestones|"
    r"open|closed|in.?progress|wip|"
    r"gitlab|jira|confluence|"
    r"branch|repo|repository|"
    r"real.?time|live|current|latest|recent|"
    r"who.?assigned|assignee|reporter|"
    r"status|priority|label|"
    r"diff|diffs|"                            # MR diff queries
    r"!\d{1,6}\b|"                            # GitLab MR refs  !123
    r"#\d{1,6}\b|"                            # GitLab issue refs  #456
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
    r"\b[A-Z]{2,9}-\d+\b"                        # Jira issue keys e.g. AUTH-12, LONGKEY-12355
    r")\b",
    re.I,
)

# Strong signals → RAG (knowledge-base / docs / historical) — unchanged
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
    # ── French terms ──────────────────────────────────────────────────────
    r"explique|expliques|expliquer|"
    r"comment.?ca.?marche|comment.?fonctionne|"
    r"comment.?faire|"
    r"qu.?est.?ce.?que|c.?est.?quoi|"
    r"tutoriel|"
    r"configurer|"
    r"impl[eé]menter|"
    r"d[eé]finition|signification|"
    r"r[eé]soudre|"
    r"meilleures?.?pratiques?"
    r")\b",
    re.I,
)

# Named source-code file → live file content wanted (unchanged)
_FILE_IN_QUERY_RE = re.compile(
    r"\b[\w_\-]+\.(py|js|ts|go|java|rb|rs|cpp|c|cs|sh|yaml|yml|toml|json|md)\b",
    re.I,
)

# CHANGE: {2,5} → {2,9} so longer project prefixes (LONGKEY-…) are detected.
_JIRA_ISSUE_KEY_RE = re.compile(r"\b[A-Z]{2,9}-\d+\b")

# Phrases that suggest BOTH sources are needed — unchanged
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
                # CHANGE: 32→24 tokens, 15→10s timeout — the reply is a tiny JSON
                "options": {"temperature": 0.0, "num_predict": 24, "num_ctx": 2560},
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return "both"
        raw = resp.json().get("response", "").strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
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
    Same algorithm as the original — see module docstring for the two fixes.
    """
    sentry_hits = len(_SENTRY_STRONG.findall(query))
    rag_hits    = len(_RAG_STRONG.findall(query))
    both_hits   = len(_BOTH_SIGNALS.findall(query))

    # A bare Jira issue key is an unambiguous sentry signal.
    if _JIRA_ISSUE_KEY_RE.search(query):
        sentry_hits += 2

    # Jira key + doc/guide signals → user wants the live ticket AND the docs.
    if _JIRA_ISSUE_KEY_RE.search(query) and rag_hits > 0:
        logger.debug("classify: jira_key + rag_signal → forcing both")
        return "both"

    # Named source-code file → live file content from GitLab.
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
        if sentry_hits > rag_hits + 1:
            return "sentries"
        if rag_hits > sentry_hits + 1:
            return "rag"
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
        ("How do I authenticate with OAuth?",                    "rag"),
        ("Show open issues in auth-service",                     "sentries"),
        ("What are the recent commits in ecommerce-backend?",    "sentries"),
        ("List merge requests for notification-service",         "sentries"),
        ("What's the AUTH-12 ticket about and how do I fix it?", "both"),
        ("Explain the API rate limiting and show current bugs",  "both"),
        ("What is our database schema?",                         "rag"),
        ("Show pipelines for ecommerce-backend",                 "sentries"),
        # New: long-prefix Jira keys (missed by the old {2,5} pattern)
        ("What is LONGKEY-12355 about and what do the docs say?", "both"),
        ("Summarize LONGKEY-12355",                               "sentries"),
    ]
    print(f"\n{'Query':<55} {'Expected':<10} {'Got':<10} {'OK?'}")
    print("-" * 85)
    ok = 0
    for q, expected in tests:
        result = classify_intent(q, use_llm_fallback=False)
        match  = "Y" if result == expected else "N"
        if result == expected:
            ok += 1
        print(f"{q[:54]:<55} {expected:<10} {result:<10} {match}")
    print(f"\nKeyword-only accuracy: {ok}/{len(tests)}")

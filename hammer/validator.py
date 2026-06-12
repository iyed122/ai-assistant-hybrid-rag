#!/usr/bin/env python3
"""
Hammer Validator
════════════════
Deterministic, zero-LLM validator for sentries-intent responses.

Per the evaluation report:
  "For sentries-intent responses: checks cited IDs/URLs exist in the
   sentry result, no fabricated keys — deterministic, no LLM needed."

Since chat_history does not persist raw sentry results (only source
metadata), this validator works in two modes:

  MODE A — With sources (from chat_history["sources"])
    Checks that cited IDs/URLs in the answer are plausibly real by
    cross-referencing with the stored source metadata.

  MODE B — Text-only (no sources stored)
    Applies format + consistency heuristics to the answer text alone.

Returns:
  (score: float 0.0–1.0, details: dict)
  score feeds directly into the faithfulness slot of the weighted score.

Checks (v1.1):
  api_refusal              — model claimed it can't use the API (hard DPO signal)
  jira_key_format          — cited keys have plausible format/prefix vs sources
  url_validity             — cited URLs parse and hit trusted domains
  honest_empty             — bonus when model correctly says "no results"
  round_stat_fabrication   — suspicious "exactly N tickets" patterns
  source_coverage          — cited entities traceable to stored source metadata
  sentries_summary_coverage — NEW: cross-checks cited keys against the
                              persisted Sentries payload (sentries_summary field).
                              Skipped gracefully for old-schema docs that lack it.

Enhancement note (already landed in v1.1 schema):
  sentries_summary is now written to chat_history at inference time by
  intent_agent.py.  Old docs (sentries_summary absent) skip the new check
  automatically — no back-fill required.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("hammer.validator")


# ─── Patterns ─────────────────────────────────────────────────────────────────

# Jira issue key:  AUTH-12, PROJ-1234, LONGKEY-12355 (2–9 uppercase + hyphen + digits)
# FLATTENED FIX: was {2,5}-\d{1,6} — long project prefixes (LONGKEY-12355) were
# invisible to every fabrication/coverage check and to entity_match_relevance.
_JIRA_KEY_RE   = re.compile(r"(?<![A-Z])(?<![a-z])(?<!-)\b([A-Z]{2,9}-\d{1,7})\b")

# GitLab MR / issue references:  !123, #456
_GITLAB_REF_RE = re.compile(r"[!#](\d{1,6})\b")

# URLs (http/https)
_URL_RE = re.compile(r"https?://[^\s\)\"\'<>]+")

# Known trusted domains for each source
_TRUSTED_DOMAINS: Dict[str, List[str]] = {
    "jira":       ["atlassian.net", "atlassian.com", "jira."],
    "gitlab":     ["gitlab.com", "gitlab."],
    "confluence": ["atlassian.net", "atlassian.com", "confluence."],
}

# Numeric-only patterns that look suspiciously sequential / fabricated
# e.g., citing IDs 1, 2, 3, 4, 5 in a row is a hallucination heuristic
_SEQUENTIAL_IDS_RE = re.compile(r"\b([1-9])\b.{0,20}\b([1-9])\b.{0,20}\b([1-9])\b", re.S)

# "I don't have" / "no data" / explicit failure phrases — score these higher
# because the model correctly reported no data rather than fabricating
_HONEST_EMPTY_RE = re.compile(
    r"\b(no\s+results?|no\s+data|not\s+found|could\s+not\s+find|"
    r"no\s+issues?|no\s+open|empty|unavailable|"
    r"pas\s+de\s+r[eé]sultat|rien\s+trouv[eé]|aucun[e]?)\b",
    re.I,
)

# Hard negative: model claims it can't use the API / doesn't have access
# (the specific failure mode QLoRA will fix)
_API_REFUSAL_RE = re.compile(
    r"\b(i\s+don.t\s+have\s+access\s+to\s+the\s+(api|gitlab|jira|confluence)|"
    r"i\s+can.t\s+(access|call|query|retrieve)|"
    r"no\s+api\s+access|cannot\s+access\s+the\s+(live|real.?time))\b",
    re.I,
)

# Fabrication heuristic: perfectly round fake stats ("exactly 5 issues", "total of 10 bugs")
_ROUND_STAT_RE = re.compile(
    r"\b(exactly|precisely|total\s+of|there\s+are)\s+(\d+)\s+"
    r"(issues?|bugs?|tickets?|mr|merge\s+requests?|pipelines?)\b",
    re.I,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Check functions (each returns a (penalty: float, note: str) tuple)
# ═══════════════════════════════════════════════════════════════════════════════

def _check_api_refusal(answer: str) -> Tuple[float, str]:
    """Hard penalty: model claims it lacks API access. This is the primary DPO failure mode."""
    if _API_REFUSAL_RE.search(answer):
        return 0.30, "model_claimed_no_api_access"
    return 0.0, "ok"


def _check_jira_key_format(answer: str, sources: List[Dict], query: str = "") -> Tuple[float, str]:
    """
    Validate Jira issue keys mentioned in the answer.
    Penalises if keys appear with implausible project prefixes or if
    the count far exceeds what sources suggest was returned.

    Query-echo exemption: keys that appear verbatim in the user query are
    always expected in the answer (the model is addressing the asked ticket)
    and must never trigger a prefix-mismatch penalty — they are echoes, not
    fabrications.
    """
    cited_keys = _JIRA_KEY_RE.findall(answer)
    if not cited_keys:
        return 0.0, "no_jira_keys"

    # Keys that came directly from the user query — never penalise these
    query_keys = set(k.upper() for k in _JIRA_KEY_RE.findall(query))

    # Extract project prefixes of cited keys, excluding query-echoed keys
    non_echo_cited = [k for k in cited_keys if k.upper() not in query_keys]
    if not non_echo_cited:
        # All cited keys are echoes of the query — nothing to penalise
        return 0.0, f"jira_keys_ok: {cited_keys[:5]}"

    prefixes = {k.split("-")[0] for k in non_echo_cited}

    # If sources contain Jira data, check consistency
    jira_sources = [s for s in sources if s.get("source") == "jira"]
    if jira_sources:
        # Urls or titles may contain key hints
        combined_source_text = " ".join(
            s.get("title", "") + " " + s.get("url", "") for s in jira_sources
        ).upper()
        source_prefixes = set(_JIRA_KEY_RE.findall(combined_source_text))
        source_prefixes = {p.split("-")[0] for p in source_prefixes}

        # Fabrication signal: non-echo cited prefix not in any source
        if source_prefixes and not prefixes.intersection(source_prefixes):
            return 0.25, f"jira_prefix_mismatch: cited={prefixes} sources={source_prefixes}"

    # Format check: all cited keys must have numeric part (already guaranteed by regex)
    return 0.0, f"jira_keys_ok: {cited_keys[:5]}"


def _check_url_validity(answer: str) -> Tuple[float, str]:
    """
    Validate URLs mentioned in the answer:
      - Must parse correctly (no mangled URLs)
      - Domain must match one of the known trusted domains
        (or be a project-specific subdomain)
    """
    urls = _URL_RE.findall(answer)
    if not urls:
        return 0.0, "no_urls"

    bad_urls  = []
    good_urls = []

    for url in urls[:20]:  # cap at 20 for speed
        try:
            parsed = urlparse(url)
        except Exception:
            bad_urls.append(url)
            continue

        netloc = parsed.netloc.lower()
        if not netloc:
            bad_urls.append(url)
            continue

        trusted = any(
            trusted_domain in netloc
            for domains in _TRUSTED_DOMAINS.values()
            for trusted_domain in domains
        )
        # Also allow project-specific hostnames (e.g., company.gitlab.com)
        looks_real = "." in netloc and len(netloc) > 4

        if trusted or looks_real:
            good_urls.append(url)
        else:
            bad_urls.append(url)

    if bad_urls:
        ratio = len(bad_urls) / len(urls)
        penalty = min(0.30, ratio * 0.30)
        return penalty, f"suspicious_urls: {bad_urls[:3]}"

    return 0.0, f"urls_ok: {len(good_urls)}"


def _check_honest_empty(
    answer: str,
    sentries_summary: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[float, str]:
    """
    Reward: if the sentry returned no data and the model correctly said
    "no results found" instead of fabricating, give a bonus.

    Gating fix: only fires when sentries_summary is explicitly empty (== [])
    OR is None (old schema, no payload to verify). For non-empty summaries
    the bonus is suppressed — otherwise any answer text containing phrases
    like "not found" would score above 0.85 even when sentries returned
    a full payload and the model just happened to mention one missing item.
    """
    # Suppress bonus when we know sentries returned actual data
    if sentries_summary:  # non-empty list
        return 0.0, "summary_non_empty_bonus_suppressed"
    if _HONEST_EMPTY_RE.search(answer):
        return -0.10, "correctly_reported_empty"  # negative = bonus
    return 0.0, "ok"


def _check_round_stat_fabrication(answer: str) -> Tuple[float, str]:
    """
    Heuristic: "there are exactly 10 issues" is suspicious when the
    real sentry results rarely return perfectly round numbers at scale.

    Threshold raised to 3 matches (from 2) and penalty halved to 0.05
    to reduce false positives. Still serves as a low-signal fabrication
    hint but no longer swamps legitimate answers.
    """
    matches = _ROUND_STAT_RE.findall(answer)
    if len(matches) >= 3:
        return 0.05, f"suspicious_round_stats: {matches[:3]}"
    return 0.0, "ok"


def _check_source_coverage(answer: str, sources: List[Dict], query: str = "") -> Tuple[float, str]:
    """
    Cross-reference: entity references in the answer should be traceable
    to the returned sources.

    Matches three citation styles:
      - Double-quoted strings:  "AUTH-12: Login fails"
      - Markdown bold:          **AUTH-12: Login fails**
      - Backtick spans:         `AUTH-12`

    Also cross-checks Jira keys explicitly: every key cited in the answer
    that does NOT appear in any source title/URL is a fabrication signal.

    Query-echo exemption: keys that appear verbatim in the user query are
    always expected in the answer — the model is addressing the asked ticket.
    Penalising these as fabrications is always a false positive.
    """
    if not sources:
        return 0.0, "no_sources_to_check"

    source_corpus = " ".join(
        s.get("title", "") + " " + s.get("url", "") for s in sources
    ).upper()

    # Keys the user explicitly asked about — never fabrications
    query_keys = set(k.upper() for k in _JIRA_KEY_RE.findall(query))

    # Collect all cited references across styles
    cited_titles: List[str] = []
    cited_titles += re.findall(r'"([^"]{5,80})"', answer)
    cited_titles += re.findall(r'\*\*([^*]{5,80})\*\*', answer)
    cited_titles += re.findall(r'`([^`]{5,80})`', answer)

    # Jira key cross-check (most reliable fabrication signal)
    # Exempt query-echoed keys: the model citing PROJ-18399 when asked about
    # PROJ-18399 is correct behavior, not hallucination, even if the RAG
    # retriever didn't return that ticket (e.g. due to no live API access).
    cited_keys  = set(_JIRA_KEY_RE.findall(answer.upper()))
    source_keys = set(_JIRA_KEY_RE.findall(source_corpus))
    if cited_keys and source_keys:
        orphan_keys = (cited_keys - source_keys) - query_keys
        if len(orphan_keys) > len(cited_keys - query_keys) // 2 and (cited_keys - query_keys):
            return 0.20, f"jira_keys_not_in_sources: {list(orphan_keys)[:3]}"

    if not cited_titles:
        return 0.0, "no_cited_titles_found"

    # Filter to genuine source references only — strip out the model's own prose
    # headers and structural labels (e.g. "Answer:", "Status:", "Definition:",
    # "CC Errors in Redundancy Mode vs. CI Pipeline Failures").
    # A title counts as a source reference only if it contains at least one of:
    #   • A Jira key  (PROJ-123, DVR-456 …)
    #   • An opening bracket [ — ticket/page title convention in this system
    #   • A URL fragment (http)
    # Everything else is model-generated structure, not a claimed citation.
    _IS_SOURCE_REF = re.compile(r'(\b[A-Z]{2,5}-\d+\b|\[|https?://)', re.I)
    source_ref_titles = [t for t in cited_titles if _IS_SOURCE_REF.search(t)]

    if not source_ref_titles:
        # No genuine source-reference titles — nothing to penalise
        return 0.0, "no_source_ref_titles_found"

    # Further exempt titles whose only Jira key is a query-echo — the model
    # is expected to reference the ticket it was asked about, even if the
    # RAG retriever didn't return it (e.g. no live API at eval time).
    def _title_is_query_echo(title: str) -> bool:
        title_keys = set(k.upper() for k in _JIRA_KEY_RE.findall(title))
        return bool(title_keys) and title_keys.issubset(query_keys)

    non_echo_ref_titles = [t for t in source_ref_titles if not _title_is_query_echo(t)]

    if not non_echo_ref_titles:
        return 0.0, "no_source_ref_titles_found"

    unsupported = [
        t for t in non_echo_ref_titles
        if t.upper()[:20] not in source_corpus
    ]

    if len(unsupported) > len(non_echo_ref_titles) // 2:
        ratio = len(unsupported) / len(non_echo_ref_titles)
        return round(min(0.25, ratio * 0.25), 3), f"unsupported_titles: {unsupported[:3]}"

    return 0.0, "source_coverage_ok"


# ═══════════════════════════════════════════════════════════════════════════════
# Answer relevance for sentries (replaces hardcoded 0.50)
# ═══════════════════════════════════════════════════════════════════════════════


# Count-query patterns: "how many", "give me a number", "approximately", etc.
_COUNT_QUERY_RE = re.compile(
    r"\b(how\s+many|count|total\s+(number\s+of|tickets?|issues?)|"
    r"number\s+of|approximately|give\s+me\s+a\s+number|just\s+a\s+number|"
    r"combien|nombre\s+de)\b",
    re.I,
)

# Numeric-only answer: optional prefix like "Answer:" then one or more digits
# Handles: "15", "Answer: 15", "answer : 42", "~15", "about 15"
_NUMERIC_ANSWER_RE = re.compile(
    r"^\s*(?:answer\s*:\s*|~|about\s+|environ\s+|approximately\s+)?(\d{1,6})\s*$",
    re.I,
)


def entity_match_relevance(query: str, answer: str) -> float:
    """
    Deterministic answer_relevance proxy for sentries-intent responses.

    Replaces the hardcoded 0.50 neutral value that capped every perfect
    sentries answer at a weighted_score ceiling of ~0.79.

    Strategy (in priority order):
      0. Numeric count fast path — count query + pure-number answer → 0.85.
         Fixes the systematic underscoring of short numeric answers like
         "Answer: 15" which have zero token overlap with the query and
         previously floored at 0.30.
      1. Jira key match  — if query contains PROJ-123, answer must too.   → 1.0 / 0.2
      2. GitLab ref match — if query has !123 or #456, answer must too.   → 1.0 / 0.3
      3. Project/repo name token overlap (stop-words stripped).           → 0.3–1.0
      4. Default neutral                                                   → 0.5

    Returns float 0.0–1.0.
    """
    if not query or not answer:
        return 0.5

    query_up  = query.upper()
    answer_up = answer.upper()

    # 0. Numeric count fast path
    #    A count/quantity question answered with a bare number is maximally
    #    relevant — the model correctly followed a "negative constraint"
    #    (don't list names, just give a number).  Token overlap is zero by
    #    design, so the general token path would floor at 0.30 — wrong.
    if _COUNT_QUERY_RE.search(query) and _NUMERIC_ANSWER_RE.match(answer.strip()):
        return 0.85

    # 1. Jira key match
    query_jira_keys  = set(_JIRA_KEY_RE.findall(query_up))
    answer_jira_keys = set(_JIRA_KEY_RE.findall(answer_up))
    if query_jira_keys:
        matched = query_jira_keys & answer_jira_keys
        return round(1.0 if matched else 0.2, 4)

    # 2. GitLab ref match
    query_refs  = set(_GITLAB_REF_RE.findall(query))
    answer_refs = set(_GITLAB_REF_RE.findall(answer))
    if query_refs:
        matched = query_refs & answer_refs
        return round(1.0 if matched else 0.3, 4)

    # 3. Project/repo name token overlap
    _STOP = {"show", "list", "get", "open", "closed", "issues", "for", "in",
              "the", "a", "an", "of", "and", "or", "with", "recent", "latest",
              "pipelines", "mrs", "merge", "requests", "bugs", "tickets"}

    def _tokens(text: str) -> set:
        return {
            t.lower() for t in re.findall(r"\b[a-zA-Z][\w\-]{2,}\b", text)
            if t.lower() not in _STOP
        }

    query_tokens  = _tokens(query)
    answer_tokens = _tokens(answer)

    if not query_tokens:
        return 0.5

    overlap = query_tokens & answer_tokens
    ratio   = len(overlap) / len(query_tokens)
    # Map 0→0.3, 1→1.0 — even a single matching token is meaningfully relevant
    return round(min(1.0, 0.3 + ratio * 0.7), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Main validator
# ═══════════════════════════════════════════════════════════════════════════════

def _check_sentries_summary_coverage(
    answer: str,
    sentries_summary: List[Dict[str, Any]],
    sources: Optional[List[Dict[str, Any]]] = None,
    query: str = "",
) -> Tuple[float, str]:
    """
    Cross-check the answer against the persisted sentries_summary payload.

    This is the most precise faithfulness check available for Sentries data
    because it compares directly against the raw API results the model saw
    at inference time.

    Logic:
      - Extract Jira keys and numeric counts from the summary.
      - If the answer cites a key that's not in the summary → fabrication penalty.
      - If the summary is non-empty but the answer claims "no results" → penalty.
      - If the summary is empty and the answer reports empty → bonus.
      - If the summary is None (old schema, not persisted) → skip (0 penalty).

    Only active when sentries_summary is a non-None list (≥ v1.1 schema).
    """
    if sentries_summary is None:
        return 0.0, "sentries_summary_not_persisted"

    if not sentries_summary:
        # Summary says Sentries returned nothing.
        # However: if the sources array contains non-graph Jira entries with real
        # issue_keys, then sentries DID return data but sentries_summary was not
        # populated correctly (a known persistence gap in schema < v1.2).
        # In that case, treat it as "summary not persisted" -- skip the penalty
        # rather than falsely flagging keys that are provably in the source list.
        # This prevents false hallucination tags on docs where PROJ-911 / similar
        # keys are in sources (from_graph=False) but sentries_summary = [].
        source_jira_keys = set()
        for s in sources:
            if s.get("source") == "jira" and not s.get("from_graph", False):
                k = s.get("issue_key", "")
                if k: source_jira_keys.add(k.upper())
        if source_jira_keys:
            # Sources contain real Jira keys -- sentries_summary is a persistence
            # artifact, not evidence of empty results. Skip the fabrication check.
            return 0.0, "sentries_summary_empty_but_sources_present_skip"

        if _HONEST_EMPTY_RE.search(answer):
            return -0.05, "correctly_reported_empty_from_summary"
        # Model may have fabricated results when Sentries returned nothing.
        # BUT exempt two cases that are never fabrications:
        #   1. Key appears in a source URL (Fix A — already applied earlier)
        #   2. Key came from the user query — the model is addressing what
        #      it was asked about, not inventing data (Fix B — query-echo).
        query_keys    = set(k.upper() for k in _JIRA_KEY_RE.findall(query))
        source_url_keys = set(
            k.upper()
            for s in (sources or [])
            for k in _JIRA_KEY_RE.findall((s.get("url", "") or ""))
        )
        cited_keys = [
            k for k in _JIRA_KEY_RE.findall(answer)
            if k.upper() not in source_url_keys and k.upper() not in query_keys
        ]
        if cited_keys:
            return 0.15, f"fabricated_keys_on_empty_sentries: {cited_keys[:3]}"
        return 0.0, "empty_sentries_no_keys_cited"

    # Build a ground-truth corpus from the summary dicts
    summary_corpus = " ".join(
        str(item.get("key", ""))  + " "
        + str(item.get("id",  ""))  + " "
        + str(item.get("title", "")) + " "
        + str(item.get("summary", ""))
        for item in sentries_summary
    ).upper()

    # ── Jira keys cross-check ─────────────────────────────────────────────
    summary_keys = set(_JIRA_KEY_RE.findall(summary_corpus))
    cited_keys   = set(_JIRA_KEY_RE.findall(answer.upper()))

    if cited_keys and summary_keys:
        orphans = cited_keys - summary_keys
        if len(orphans) > len(cited_keys) // 2:
            return 0.20, f"keys_not_in_sentries_summary: {list(orphans)[:3]}"
        return 0.0, f"summary_keys_ok: cited={len(cited_keys)} matched={len(cited_keys)-len(orphans)}"

    # ── GitLab MR / issue refs cross-check (!N and #N) ───────────────────
    # When the sentries_summary contains GitLab data (get_merge_requests,
    # get_issues, get_pipelines), items have "id" and "iid" numeric fields.
    # Collect all IDs from the summary and verify that any !N or #N refs
    # cited in the answer actually appear in the returned data.
    # Query-echo exemption: refs that appeared in the query (!N from the user)
    # are expected in the answer and must not be penalised.
    query_gl_refs = set(_GITLAB_REF_RE.findall(query))
    cited_gl_refs = set(_GITLAB_REF_RE.findall(answer))
    non_echo_gl   = cited_gl_refs - query_gl_refs

    if non_echo_gl:
        # Extract numeric IDs from the summary (iid field is the user-facing !N)
        summary_gl_ids: set = set()
        for item in sentries_summary:
            for field in ("iid", "id"):
                val = item.get(field)
                if val is not None:
                    summary_gl_ids.add(str(val))

        if summary_gl_ids:
            orphan_gl = non_echo_gl - summary_gl_ids
            if len(orphan_gl) > len(non_echo_gl) // 2:
                return 0.20, f"gitlab_refs_not_in_sentries_summary: {list(orphan_gl)[:3]}"
            return 0.0, f"gitlab_refs_ok: cited={len(non_echo_gl)} matched={len(non_echo_gl)-len(orphan_gl)}"

    # Summary has data but answer reports empty — model missed the results
    if (summary_keys or cited_gl_refs) and _HONEST_EMPTY_RE.search(answer):
        return 0.10, "falsely_reported_empty_when_summary_has_data"

    return 0.0, "summary_coverage_ok"


def validate_sentry_answer(
    answer:          str,
    sources:         List[Dict[str, Any]],
    sentries_summary: Optional[List[Dict[str, Any]]] = None,
    query:           str = "",
) -> Tuple[float, Dict[str, Any]]:
    """
    Validate a sentries-intent answer and return a quality score.

    Args:
        answer:           The generated answer text.
        sources:          List of source dicts from chat_history
                          (title, url, source, score).
        sentries_summary: Optional raw Sentries API payload persisted at
                          inference time (available in schema ≥ v1.1).
                          When provided, enables precise cross-validation of
                          cited keys/IDs against actual API results.
                          None → old schema, skip that check.
                          []   → Sentries returned no data.
        query:            Original user query. Used to exempt query-echoed
                          Jira keys from jira_prefix_mismatch penalties.

    Returns:
        (score, details)
        score:   0.0–1.0  (replaces RAGAS faithfulness in the weighted formula)
        details: breakdown of each check for auditing
    """
    if not answer:
        return 0.10, {"error": "empty_answer"}

    # Accumulate penalties (each check returns (penalty, note))
    checks: List[Tuple[str, float, str]] = []

    def run(name: str, fn, *args):
        penalty, note = fn(*args)
        checks.append((name, penalty, note))
        return penalty

    total_penalty = 0.0
    total_penalty += run("api_refusal",             _check_api_refusal, answer)
    total_penalty += run("jira_key_format",          _check_jira_key_format, answer, sources, query)
    total_penalty += run("url_validity",             _check_url_validity, answer)
    total_penalty += run("honest_empty",             _check_honest_empty, answer, sentries_summary)
    total_penalty += run("round_stat_fabrication",   _check_round_stat_fabrication, answer)
    total_penalty += run("source_coverage",          _check_source_coverage, answer, sources, query)
    # New in v1.1: precise cross-check against persisted Sentries payload.
    # Skipped gracefully (0 penalty, "not_persisted" note) for old schema docs.
    total_penalty += run("sentries_summary_coverage",
                         _check_sentries_summary_coverage, answer, sentries_summary, sources, query)

    # Base score — adjusted for verifiability:
    #   0.85  when sentries_summary is present (we can cross-check actual API results)
    #   0.75  when sentries_summary is None (old schema: no payload to verify against)
    #         and no Jira keys cited (nothing checkable) — prevents silent over-scoring
    #         of ambiguous answers where all penalty checks return 0 by default.
    cited_keys_for_base = _JIRA_KEY_RE.findall(answer.upper())
    urls_for_base       = _URL_RE.findall(answer)
    has_verifiable_entities = bool(cited_keys_for_base or urls_for_base)

    if sentries_summary is None and not has_verifiable_entities:
        base = 0.75   # unverifiable: old schema + no checkable entities
    else:
        base = 0.85   # verifiable: summary present OR checkable entities found

    score = round(max(0.0, min(1.0, base - total_penalty)), 4)

    details = {
        "score":          score,
        "base":           base,
        "total_penalty":  round(total_penalty, 4),
        "checks": [
            {"name": n, "penalty": p, "note": note}
            for n, p, note in checks
        ],
    }

    return score, details


# ─── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        (
            "I don't have access to the GitLab API so I can't retrieve issues.",
            [], None,
            "should score LOW (api_refusal)",
        ),
        (
            "Here are the open issues in AUTH-12, AUTH-15, AUTH-22. "
            "See https://mycompany.atlassian.net/browse/AUTH-12.",
            [{"source": "jira", "title": "AUTH-12: Login fails",
              "url": "https://mycompany.atlassian.net/browse/AUTH-12"}],
            [{"key": "AUTH-12"}, {"key": "AUTH-15"}, {"key": "AUTH-22"}],
            "should score HIGH (real Jira data, summary match)",
        ),
        (
            "No results were found for that query. The project has no open issues currently.",
            [], [],
            "should score OK (honest empty, empty summary)",
        ),
        (
            "There are exactly 10 issues and exactly 5 merge requests.",
            [], None,
            "should score MEDIUM (suspicious round stats)",
        ),
        (
            "The current status of PROJ-18473 is Resolved.",
            [], [{"key": "PROJ-18473", "summary": "Critical alarm"}],
            "should score HIGH (key in summary)",
        ),
        (
            "AUTH-99 is a critical issue.",
            [], [{"key": "AUTH-12"}],
            "should score LOWER (orphan key not in summary)",
        ),
    ]

    print(f"\n{'Case':<52} {'Score':>6}  Signal")
    print("─" * 75)
    for answer, sources, sentries_summary, label in test_cases:
        score, details = validate_sentry_answer(answer, sources, sentries_summary)
        signal = "HIGH" if score >= 0.80 else ("LOW" if score < 0.50 else "MED")
        print(f"{label[:52]:<52} {score:>6.3f}  {signal}")
        for c in details["checks"]:
            if c["penalty"] != 0.0:
                print(f"  └ {c['name']}: penalty={c['penalty']} ({c['note']})")

    print("\n── entity_match_relevance numeric fast path ──")
    count_cases = [
        ("How many PROJ tickets are in the backlog approximately, just give me a number",
         "Answer: 15",   "should return 0.85 (count query + numeric answer)"),
        ("How many PROJ tickets are in the backlog approximately, just give me a number",
         "15",           "should return 0.85 (bare number)"),
        ("What is PROJ-18473 about?",
         "The ticket PROJ-18473 is about CC errors.",
         "should return 1.0 (jira key match)"),
        ("List open issues",
         "No open issues were found.",
         "should return token overlap"),
    ]
    for q, a, label in count_cases:
        score = entity_match_relevance(q, a)
        print(f"  {label[:52]:<52} {score:.4f}")
    print()


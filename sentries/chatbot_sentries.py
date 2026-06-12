#!/usr/bin/env python3
"""
Real-Time AI Chatbot  –  Ollama + API Sentries  v4
────────────────────────────────────────────────────
Key changes vs v3:
  - ProjectRegistry: loads all GitLab AND Jira projects from their APIs
    on startup — no hardcoded project lists anywhere
  - _LIST_PROJ_RE: now matches count/how-many phrasing so queries like
    "how many projects on gitlab?" and "count projects" route correctly
  - Jira name-to-key map built dynamically via _jira_aliases()
  - count_results(): deduplicates fan-out results before counting
"""

import os
import re
import sys
import requests
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

from sentries.sentry_dispatcher import SentryDispatcher, OPERATION_CATALOGUE

load_dotenv()


# ── FLATTENED helpers (from the optimized overlay) ─────────────────────────────

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def apply_threadsafe_ratelimiter() -> bool:
    """
    OPT-IN (AGENT_TEST_THREADSAFE_RL=1): wrap base_sentry.RateLimiter.wait with
    a per-instance lock. The stock limiter is not thread-safe under the agent's
    ThreadPoolExecutor — racing bursts can exceed the API limit and trigger
    429 retries. Class-level patch; affects the whole process.
    """
    import threading
    from sentries import base_sentry as _base_sentry

    if getattr(_base_sentry.RateLimiter, "_ts_patched", False):
        return True
    _orig_wait = _base_sentry.RateLimiter.wait

    def _locked_wait(self):
        lock = self.__dict__.get("_ts_lock")
        if lock is None:
            lock = self.__dict__.setdefault("_ts_lock", threading.Lock())
        with lock:
            _orig_wait(self)

    _base_sentry.RateLimiter.wait = _locked_wait
    _base_sentry.RateLimiter._ts_patched = True
    return True


if os.getenv("AGENT_TEST_THREADSAFE_RL", "0") == "1":
    apply_threadsafe_ratelimiter()

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3.5").strip()


# ─── Ollama helpers ───────────────────────────────────────────────────────────

def resolve_model_name(requested: str, host: str) -> str:
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        if resp.status_code != 200:
            return requested
        available = [m["name"] for m in resp.json().get("models", [])]
        if requested in available:
            return requested
        req_base = requested.split(":")[0]
        for name in available:
            if name.split(":")[0] == req_base:
                print(f"  i  Model name resolved: '{requested}' -> '{name}'")
                return name
        return requested
    except Exception:
        return requested


def check_ollama() -> bool:
    global OLLAMA_MODEL
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        available = [m["name"] for m in resp.json().get("models", [])]
        OLLAMA_MODEL = resolve_model_name(OLLAMA_MODEL, OLLAMA_HOST)
        return OLLAMA_MODEL in available
    except Exception:
        return False


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ollama_generate(prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx":     8192,   # FIX: was unset — Ollama defaulted to ~2048,
                                           # silently truncating large sentry payloads
                    # FIX: correct stop tokens for qwen3 (ChatML format).
                    # "<|end|>" and "<|user|>" are not qwen3 tokens — the model
                    # never matched them, causing runaway generation or mid-sentence cuts.
                    "stop": ["<|im_end|>", "<|endoftext|>"],
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        return _strip_think_tags(raw)
    except requests.exceptions.ConnectionError:
        return "ERROR:OLLAMA_DOWN"
    except Exception as exc:
        return f"ERROR:{exc}"


# ─── Project Registry ─────────────────────────────────────────────────────────

def _find_project(query: str, registry: Any) -> Optional[str]:
    """Helper to resolve project keys/IDs. Checks Jira first."""
    if not registry:
        return None
    # 1. Jira Project Priority (e.g., 'TM')
    for key in registry.jira_projects:
        if f"{key}-" in query.upper() or key.lower() == query.lower():
            return key
    # 2. GitLab Paths fallback
    for path in registry.all_paths():
        if path.lower() in query.lower():
            return path
    return None


class ProjectRegistry:
    """
    Loads all GitLab projects AND Jira projects from their APIs on startup.
    Resolves short names/aliases deterministically with no hardcoded maps.
    """

    def __init__(self):
        # GitLab
        self._name_to_path: Dict[str, str] = {}
        self._all_projects: List[Dict]     = []
        self.gl_loaded = False

        # Jira — built dynamically from the live API
        self.jira_projects:    Dict[str, str] = {}   # KEY -> display name
        self.jira_name_to_key: Dict[str, str] = {}   # alias -> KEY
        self.jira_loaded = False

    # ── GitLab ────────────────────────────────────────────────────────────────

    def load_gitlab(self, dispatcher: SentryDispatcher):
        result = dispatcher.dispatch({
            "source": "gitlab", "operation": "list_projects",
            "params": {"limit": 500},   # raised from 100 — enterprise may have hundreds of repos
        })
        if not result.success:
            print(f"  Warning: GitLab registry load failed: {result.error}")
            return

        for p in result.data:
            path  = p["path_with_ns"]
            short = path.split("/")[-1].lower()
            name  = p["name"].lower()

            if short not in self._name_to_path:
                self._name_to_path[short] = path
            elif p.get("namespace", "").lower() == os.getenv(
                "GITLAB_PREFERRED_NAMESPACE", ""
            ).lower() and os.getenv("GITLAB_PREFERRED_NAMESPACE", ""):
                # Preferred namespace wins short-name collisions (env-driven)
                self._name_to_path[short] = path

            self._name_to_path[name]         = path
            self._name_to_path[path.lower()] = path

        self._all_projects = result.data
        self.gl_loaded = True

    # ── Jira ──────────────────────────────────────────────────────────────────

    def load_jira(self, dispatcher: SentryDispatcher):
        """
        Discover ALL Jira projects via the list_projects API.
        Falls back to sampling recent issues if list_projects is unavailable.
        This replaces the old 50-issue sample which missed most projects.
        """
        # Primary: use list_projects (paginated, returns ALL projects)
        result = dispatcher.dispatch({
            "source": "jira", "operation": "list_projects",
            "params": {"limit": 500},
        })

        if result.success and result.data:
            for p in result.data:
                key  = (p.get("key") or "").upper().strip()
                name = p.get("name") or key
                if not key:
                    continue
                if key not in self.jira_projects:
                    self.jira_projects[key] = name
                    for alias in self._jira_aliases(key, name):
                        if alias and alias not in self.jira_name_to_key:
                            self.jira_name_to_key[alias] = key
            self.jira_loaded = bool(self.jira_projects)
            print(f"  ✓ Jira registry: {len(self.jira_projects)} projects loaded via list_projects")
            return

        # Fallback: sample recent issues (limited but still useful when
        # list_projects is not available e.g. on older Jira Server)
        print(f"  ⚠ list_projects unavailable ({result.error}), falling back to issue sampling")
        result = dispatcher.dispatch({
            "source": "jira", "operation": "get_issues",
            "params": {"jql": "project is not EMPTY ORDER BY created DESC", "limit": 200},
        })
        if not result.success:
            print(f"  Warning: Jira project discovery failed: {result.error}")
            return

        for issue in result.data:
            key  = (issue.get("project_key") or issue.get("project") or "").upper()
            name = issue.get("project_name") or issue.get("project") or key
            if not key:
                continue
            if key not in self.jira_projects:
                self.jira_projects[key] = name
                for alias in self._jira_aliases(key, name):
                    if alias and alias not in self.jira_name_to_key:
                        self.jira_name_to_key[alias] = key

        self.jira_loaded = bool(self.jira_projects)
        print(f"  ✓ Jira registry: {len(self.jira_projects)} projects loaded via issue sampling")

    @staticmethod
    def _jira_aliases(key: str, name: str) -> List[str]:
        """
        Generate lookup aliases so natural-language queries resolve
        without any hardcoded mappings.

        Examples:
          key="AUTH"  name="Auth Service"
            -> ["auth", "auth service", "auth-service", "authservice"]

          key="ECOM"  name="ECommerce Platform"
            -> ["ecom", "ecommerce platform", "ecommerce-platform",
               "ecommerceplatform", "ecommerce", "platform"]
        """
        aliases = []
        kl = key.lower()
        nl = name.lower().strip()

        aliases.append(kl)                   # "auth"
        aliases.append(nl)                   # "auth service"
        aliases.append(nl.replace(" ", "-")) # "auth-service"
        aliases.append(nl.replace(" ", ""))  # "authservice"

        # Individual meaningful words from the project name
        _SKIP = {"service", "platform", "system", "project", "app", "the", "a"}
        for word in nl.split():
            if word not in _SKIP and len(word) > 2:
                aliases.append(word)

        return [a for a in aliases if a]

    # ── Load both at once ─────────────────────────────────────────────────────

    def load(self, dispatcher: SentryDispatcher):
        self.load_gitlab(dispatcher)
        self.load_jira(dispatcher)

    # ── GitLab resolution ─────────────────────────────────────────────────────

    def resolve(self, name: str) -> Optional[str]:
        if not name:
            return None
        key = name.lower().strip()
        if key in self._name_to_path:
            return self._name_to_path[key]
        for path in self.all_paths():
            if key in path.lower():
                return path
        return None

    def all_paths(self) -> List[str]:
        seen: set = set()
        out = []
        for v in self._name_to_path.values():
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def personal_projects(self) -> List[str]:
        # FLATTENED FIXES: (1) excluded namespaces are env-driven (were
        # hardcoded personal values); (2) fan-out CAP — several route_query
        # rules emit one API call per project here; uncapped, broad queries
        # scaled with repo count (verified 40 -> 12 calls with the cap).
        excluded_raw = os.getenv("SENTRY_EXCLUDED_NAMESPACES", "")
        # comma-separated namespaces to skip in fan-outs (e.g. archived groups)
        excluded = {ns.strip().lower() for ns in excluded_raw.split(",") if ns.strip()}
        out = [
            p["path_with_ns"]
            for p in self._all_projects
            if p.get("namespace", "").lower() not in excluded
            and not p.get("namespace", "").lower().startswith("test")
        ]
        cap = _env_int("SENTRY_FANOUT_MAX_GITLAB", 12)
        return out[:cap] if cap > 0 else out

    def project_list_str(self) -> str:
        return "\n".join(f"  - {p}" for p in self.all_paths())

    # ── Jira resolution ───────────────────────────────────────────────────────

    def find_jira_key(self, q: str) -> Optional[str]:
        """
        Resolve a natural-language query to a Jira project key.

        Priority order (highest to lowest):
          1. Explicit issue key (e.g. AUTH-12) — extract project part
          2. Explicit KEY word match (e.g. "AUTH", "ECOM")
          3. Exact project name match (case-insensitive full name)
          4. Alias lookup — longest alias wins
          5. Substring match on known project names
        """
        q_upper = q.upper()
        ql      = q.lower()

        # 1. Explicit issue key (e.g. AUTH-12) — extract project part
        ikm = re.search(r"\b([A-Z]{2,8}-\d+)\b", q_upper)
        if ikm:
            candidate = ikm.group(1).split("-")[0]
            if candidate in self.jira_projects:
                return candidate

        # 2. Explicit KEY word match (e.g. "\bAUTH\b")
        for key in self.jira_projects:
            if re.search(rf"\b{re.escape(key)}\b", q_upper):
                return key

        # 3. Exact project name match (Issue D fix — prevents "Media Libraries"
        #    from fuzzy-matching to "TW" (Technical Writing) due to word overlap)
        for key, name in self.jira_projects.items():
            if name.lower() == ql.strip() or name.lower() in ql:
                return key

        # 4. Alias lookup — longest alias wins (more specific match)
        for alias in sorted(self.jira_name_to_key, key=len, reverse=True):
            if alias in ql:
                return self.jira_name_to_key[alias]

        return None

    def all_jira_keys(self) -> List[str]:
        # FLATTENED FIX: fan-out cap (rule 20 emitted one call per Jira
        # project, uncapped — verified 30 -> 10 with the cap). 0 disables.
        keys = list(self.jira_projects.keys())
        cap = _env_int("SENTRY_FANOUT_MAX_JIRA", 10)
        return keys[:cap] if cap > 0 else keys


# ─── Routing regexes ─────────────────────────────────────────────────────────

_ISSUE_RE = re.compile(
    r"\b("
    r"issue|issues|bug|bugs|ticket|tickets|task|tasks|"
    r"backlog|defect|defects|problem|problems|"
    r"feature|story|stories|user.?stor(?:y|ies)|epic|epics|sub.?tasks?|"
    r"request|requests|item|items|incident|incidents|"
    r"blocker|blockers"
    r")\b", re.I)

_MR_RE = re.compile(
    r"\b(merge.?request|merge.?requests|mr|mrs|pull.?request|"
    r"pr|prs|review|reviews|code.?review)\b", re.I)

_PIPELINE_RE = re.compile(
    r"\b(pipeline|pipelines|ci|cd|build|builds|deploy|deployment|"
    r"workflow|action|job|jobs)\b", re.I)

_COMMIT_RE = re.compile(
    r"\b(commit|commits|history|changes|changelog|recent.?change|push)\b", re.I)

_FILE_TREE_RE = re.compile(
    r"\b(files|file.?list|tree|directory|structure|folder|"
    r"what.?files|list.?files|show.?files|repo.?content)\b", re.I)

_BRANCH_RE = re.compile(
    r"\b(branch|branches|branche|git.?branch|"
    r"liste.?les.?branches?|show.?branches?|all.?branches?|"
    r"how.?many.?branches?|list.?branches?)\b", re.I)

_RECHECK_RE = re.compile(
    r"\b(are.?you.?sure|check.?again|re.?check|double.?check|verify|confirm|"
    r"v[eé]rifie|v[eé]rifier|t.?es.?s[uûu]r|c.?est.?s[uûu]r|"
    r"look.?again|try.?again|fetch.?again|get.?again)\b", re.I)

_SPECIFIC_FILE_RE = re.compile(
    r"(\.gitlab-ci\.yml|dockerfile|docker-compose|readme|"
    r"requirements\.txt|package\.json|pyproject\.toml|gemfile|"
    r"\.env\.example|makefile|\.gitignore)", re.I)

_NAMED_CODE_FILE_RE = re.compile(
    r"\b([\w_\-]+\.(py|js|ts|go|java|rb|rs|cpp|c|cs|sh|yaml|yml|toml|json|md))\b", re.I)

_MILESTONE_RE = re.compile(r"\bmile?s?t?o?r?n?e?s?\b", re.I)

_INFO_RE = re.compile(
    r"\b(info|information|about|description|created|when.?was|"
    r"details|exist|exists|visibility|public|private|language|stats)\b", re.I)

# Fix 1: include count/how-many phrasing so "count projects" and
# "how many projects on gitlab?" route to list_projects correctly.
_LIST_PROJ_RE = re.compile(
    r"(list|show|get|find|all|available|my|how.?many|count|number.?of|total)"
    r"\s.{0,20}(project|repo)s?\b"
    r"(?!.*(issue|bug|ticket|mr|merge|commit|pipeline|milestone))",
    re.I)

_SECURITY_RE = re.compile(
    r"\b(security|vulnerability|vulnerabilities|injection|exploit|"
    r"cve|breach|attack|xss|csrf|sqli|owasp)\b", re.I)

_WIP_RE = re.compile(
    r"\b(in.?progress|wip|work.?in.?progress|active|current|ongoing|"
    r"doing|working.?on)\b", re.I)

_JIRA_RE       = re.compile(r"\bjira\b", re.I)
_CONFLUENCE_RE = re.compile(r"\bconfluence\b", re.I)

# Broader Confluence trigger — word-boundary safe (avoids "docker", "homepage")
_CONFLUENCE_TRIGGER_RE = re.compile(
    r"\b(confluence|docs?|documentation|wiki|knowledge.?base)\b", re.I)

# "recently updated / latest pages" → get_recent_pages
_CONFLUENCE_RECENT_RE = re.compile(
    r"\b(recent|latest|new|updated|modified|changed)\b.{0,30}"
    r"\b(page|pages|doc|docs|confluence|wiki|content)\b|"
    r"\b(page|pages|doc|docs|confluence|wiki|content)\b.{0,30}"
    r"\b(recent|latest|new|updated|modified|changed)\b",
    re.I)

# "list/show [confluence] spaces" — generic space listing only
_CONFLUENCE_LIST_SPACES_RE = re.compile(
    r"\b(list|show|get|all|available)\b.{0,20}\bspaces?\b|"
    r"\bconfluence\s+space\b",
    re.I)

# Extract explicit space key: "in/for DEV space", "space: DEV"
_CONFLUENCE_SPACE_KEY_RE = re.compile(
    r"\b(?:in|for|within|from|search)\s+(?:the\s+)?([A-Z]{2,6})\s+space\b|"
    r"\bspace[:\s]+([A-Z]{2,6})\b"
)

_CODE_SEARCH_RE = re.compile(
    r"\b(middleware|router\b|controller|handler|decorator|"
    r"auth\s+router|auth\s+handler|login\s+function|auth\s+module|"
    r"webhook\s+handler|payment\s+handler|rate\s+limit\s+middleware)\b", re.I)

_ENV_VARS_RE = re.compile(
    r"\b(env(ironment)?.?var(iable)?s?|\.env|config|configuration|"
    r"secret|secrets|api.?key|env.?file|dotenv)\b", re.I)

_ALL_MR_RE = re.compile(
    r"(all|every|across).{0,30}(merge.?request|mr|pull.?request)|"
    r"(merge.?request|mr|pull.?request).{0,30}(all|every|across)",
    re.I)

_ALL_COMMITS_RE = re.compile(
    r"(all|every|across|latest).{0,30}commit|"
    r"commit.{0,30}(all|every|across|latest|personal)",
    re.I)

_COMPOUND_COMMIT_MR_RE = re.compile(
    r"\b(commit|commits).{0,25}\band\b.{0,25}(merge.?request|mr|mrs|pull.?request)s?\b|"
    r"\b(merge.?request|mr|mrs|pull.?request)s?.{0,25}\band\b.{0,25}(commit|commits)\b",
    re.I)

_COUNT_RE = re.compile(
    r"\b("
    r"how many|how much|"
    r"count|counting|"
    r"total|totals|total number|"
    r"number of|"
    r"tally|"
    r"give me (?:a |the )?count|"
    r"what.?s the (?:total|count|number)|"
    r"tell me how many|show me how many|"
    r"quantit(?:y|ies)"
    r")\b", re.I)

_DETAIL_RE = re.compile(
    r"\b("
    r"explain|describe|detail|details|"
    r"what are they|show (?:me )?(?:them|each|all)|list (?:them|each|all)|"
    r"and who|by whom|who made|who created|who assigned|"
    r"what (?:is|are|were|was) (?:each|they|it)|"
    r"impact|affect|cause|why|reason|"
    r"are any|is any|which (?:one|ones|are)|"
    r"tell me (?:about|more)|give me (?:more|details)|"
    r"summarize|summary"
    r")\b", re.I)

def is_pure_count(q: str) -> bool:
    """True only when query is purely asking for a number with no
    detail/reasoning/listing intent alongside it."""
    return bool(_COUNT_RE.search(q)) and not bool(_DETAIL_RE.search(q))


# ─── State / status / priority helpers ───────────────────────────────────────

def _state(q: str) -> str:
    l = q.lower()
    if any(w in l for w in ("closed", "merged", "done", "resolved", "fixed", "completed")):
        return "closed"
    if any(w in l for w in ("all", "every", "both")):
        return "all"
    return "opened"


_JIRA_STATUS_MAP: Dict[str, str] = {
    "closed":      "Done",
    "close":       "Done",
    "done":        "Done",
    "resolved":    "Done",
    "fixed":       "Done",
    "completed":   "Done",
    "finished":    "Done",
    "finish":      "Done",
    "open":        "Open",
    "opened":      "Open",
    "new":         "Open",
    "created":     "Open",
    "in progress":  "In Progress",
    "in-progress":  "In Progress",
    "ongoing":      "In Progress",
    "wip":          "In Progress",
    "started":      "In Progress",
    "in review":   "In Review",
    "review":      "In Review",
    "reviewing":   "In Review",
    "testing":     "Testing",
    "qa":          "Testing",
    "blocked":     "Blocked",
    "block":       "Blocked",
    "todo":        "To Do",
    "to do":       "To Do",
    "backlog":     "Backlog",
    "pending":     "To Do",
    "archived":    "Closed",
    # French status aliases (accents optional — engineers often omit them)
    "\u00e0 faire":     "To Do",        # À faire
    "a faire":          "To Do",
    "non r\u00e9solu":  "To Do",        # Non résolu
    "non resolu":       "To Do",
    "en cours":         "In Progress",
    "en-cours":         "In Progress",
    "en cours de":      "In Progress",
    "termin\u00e9":     "Done",         # Terminé
    "termine":          "Done",
    "termin\u00e9e":    "Done",
    "terminee":         "Done",
    "r\u00e9solu":      "Done",         # Résolu
    "resolu":           "Done",
    "ferm\u00e9":       "Done",         # Fermé
    "ferme":            "Done",
    "bloqu\u00e9":      "Blocked",      # Bloqué
    "bloque":           "Blocked",
    "en attente":       "To Do",
    "en r\u00e9vision": "In Review",    # En révision
    "en revision":      "In Review",
    "en test":          "Testing",
    "en recette":       "Testing",
}

def _jira_status(q: str) -> Optional[str]:
    ql = q.lower()
    # Check multi-word phrases first via simple substring — avoids \b boundary
    # issues with accented French characters and space-separated terms.
    _MULTI_WORD_PHRASES = (
        "in progress", "in-progress", "in review", "to do",
        "en cours de", "en cours", "en-cours",
        "\u00e0 faire", "a faire",
        "non r\u00e9solu", "non resolu",
        "en attente",
        "en r\u00e9vision", "en revision",
        "en test", "en recette",
        "high priority", "low priority",
    )
    for phrase in _MULTI_WORD_PHRASES:
        if phrase in ql:
            val = _JIRA_STATUS_MAP.get(phrase)
            if val:
                return val
    for word, status in _JIRA_STATUS_MAP.items():
        if re.search(rf"\b{re.escape(word)}\b", ql):
            return status
    return None


# Locale-safe JQL clauses for the fallback get_issues path.
# statusCategory names are always English in Jira regardless of instance language,
# so these work even when display status names are French/German/etc.
_JIRA_STATUS_JQL: Dict[str, str] = {
    "Done":        'statusCategory = "Done"',
    "Open":        'statusCategory = "To Do"',
    "To Do":       'statusCategory = "To Do"',
    "Backlog":     'statusCategory = "To Do"',
    "In Progress": 'statusCategory = "In Progress"',
    "In Review":   'statusCategory = "In Progress"',
    "Testing":     'statusCategory = "In Progress"',
    "Blocked":     'statusCategory = "In Progress"',
    "Closed":      'statusCategory = "Done"',
}

def _jira_status_jql(status: str) -> str:
    """Convert a _jira_status() return value to a locale-safe JQL clause."""
    return _JIRA_STATUS_JQL.get(status, f'status = "{status}"')


_JIRA_PRIORITY_MAP: Dict[str, str] = {
    "urgent":        "Highest",
    "critical":      "Highest",
    "blocker":       "Highest",
    "blocking":      "Highest",
    "highest":       "Highest",
    "high priority": "High",
    "high":          "High",
    "medium":        "Medium",
    "normal":        "Medium",
    "low priority":  "Low",
    "low":           "Low",
    "minor":         "Low",
    "lowest":        "Lowest",
    "trivial":       "Lowest",
}

def _jira_priority(q: str) -> Optional[str]:
    ql = q.lower()
    for phrase in ("high priority", "low priority"):
        if phrase in ql:
            return _JIRA_PRIORITY_MAP[phrase]
    for word, priority in _JIRA_PRIORITY_MAP.items():
        if re.search(rf"\b{re.escape(word)}\b", ql):
            return priority
    return None


# ─── Reporter / author detection ──────────────────────────────────────────────

_REPORTER_KEYWORD_RE = re.compile(
    r"\b(author(?:ed)?|reporter?|reported\s+by|created\s+by|opened\s+by|"
    r"filed\s+by|submitted\s+by|raised\s+by|logged\s+by|is\s+the\s+author|"
    r"auteur|rapporteur|signal[eé]\s+par|cr[eé][eé]\s+par)\b",
    re.I,
)


def _extract_reporter(q: str) -> Optional[str]:
    """
    Extract a person name from reporter/author phrasing patterns.

    Handles:
      "tickets reported by Alan Woods"
      "Dorian You is the author"
      "where Dorian You is the author/reporter"
      "authored by Alain Saliou"
    """
    # Pattern 1: verb + "by" + Firstname [Lastname…]
    m = re.search(
        r"\b(?:reported|author(?:ed)?|filed|created|opened|raised|submitted|logged)\s+by\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
        q,
    )
    if m:
        return m.group(1)

    # Pattern 2: "Firstname Lastname is the (author|reporter)"
    m = re.search(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+is\s+the\s+(?:author|reporter)",
        q,
    )
    if m:
        return m.group(1)

    # Pattern 3: "where Firstname Lastname is …"
    m = re.search(
        r"\bwhere\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+is\b",
        q,
    )
    if m:
        return m.group(1)

    return None


# ─── Date filter extraction (Fix 2 / Fix 5) ──────────────────────────────────

_MONTH_MAP = {
    "january": "01", "jan": "01", "february": "02", "feb": "02",
    "march": "03",   "mar": "03", "april": "04",     "apr": "04",
    "may": "05",     "june": "06", "jun": "06",
    "july": "07",    "jul": "07", "august": "08",    "aug": "08",
    "september": "09","sep": "09","october": "10",   "oct": "10",
    "november": "11","nov": "11", "december": "12",  "dec": "12",
}

_ISO_DATE_RE   = re.compile(r'\b(\d{4}-\d{2}-\d{2})\b')
_PROSE_DATE_RE = re.compile(
    r'\b(january|february|march|april|may|june|july|august|september|'
    r'october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)'
    r'\s+(\d{1,2}),?\s+(\d{4})\b',
    re.I,
)
_DATE_AFTER_RE  = re.compile(
    r'\b(?:after|since|from|starting|created\s+after|added\s+after)\b', re.I
)
_DATE_BEFORE_RE = re.compile(
    r'\b(?:before|until|up\s+to|prior\s+to|created\s+before)\b', re.I
)
_RECENT_RE = re.compile(
    r'\b(most\s+recent|latest|newest|recently\s+created|last\s+created|'
    r'most\s+recently\s+created)\b',
    re.I,
)


def _parse_iso_date(q: str) -> Optional[str]:
    """Return first YYYY-MM-DD date found in q, else None."""
    m = _ISO_DATE_RE.search(q)
    return m.group(1) if m else None


def _parse_prose_date(q: str) -> Optional[str]:
    """Convert 'April 30 2026' → '2026-04-30', else None."""
    m = _PROSE_DATE_RE.search(q)
    if not m:
        return None
    month = _MONTH_MAP.get(m.group(1).lower(), "01")
    day   = m.group(2).zfill(2)
    year  = m.group(3)
    return f"{year}-{month}-{day}"


def _extract_date_filter(q: str):
    """
    Return (created_after, created_before) from the query, both as
    'YYYY-MM-DD' strings or None.
    """
    date_str = _parse_iso_date(q) or _parse_prose_date(q)
    if not date_str:
        return None, None
    if _DATE_AFTER_RE.search(q):
        return date_str, None
    if _DATE_BEFORE_RE.search(q):
        return None, date_str
    return None, None


# ─── Confluence search term extractor (Confluence fix) ───────────────────────

_CONFLUENCE_FILLER_RE = re.compile(
    r'\b(find|show|get|list|give|tell|me|us|the|a|an|in|of|for|about|'
    r'related|to|confluence|page|pages|document|documents|documentation|'
    r'wiki|space|summary|summarize|summarise|explain|describe|'
    r'what|is|are|any|some|all|and|with|that|from|its|their)\b',
    re.I,
)


def _confluence_search_terms(q: str) -> str:
    """
    Strip filler words and return 2–6 meaningful search terms for Confluence
    CQL.  Passing a full sentence to `text ~ "..."` fails on long queries.
    Example: 'Find me confluence page related to multichoice use case'
             → 'multichoice use case'
    """
    cleaned = _CONFLUENCE_FILLER_RE.sub(" ", q)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Cap at 60 chars — keeps CQL safe and avoids timeout on complex queries
    return cleaned[:60] if cleaned else q[:60]


def _clean_confluence_query(q: str) -> str:
    """
    Strip routing/command noise from a Confluence query to produce
    clean CQL-safe search terms for search_pages queries.
    Falls back to raw query if cleaning removes everything meaningful.
    """
    cleaned = re.sub(
        r"\b(confluence|search|find|show|list|get|me|our|the|a|an|"
        r"page|pages|doc|docs|documentation|wiki|knowledge|base|"
        r"in|on|about|for|what|is|are|how|do|we|have|"
        r"please|can|you|tell|give|recent|latest|updated|"
        r"new|modified|content|space|spaces)\b",
        " ",
        q.lower(),
    )
    return re.sub(r"\s+", " ", cleaned).strip()


_FILE_NAME_MAP = {
    "dockerfile":       "Dockerfile",
    "readme":           "README.md",
    "requirements.txt": "requirements.txt",
    "package.json":     "package.json",
    "pyproject.toml":   "pyproject.toml",
    "gemfile":          "Gemfile",
    ".env.example":     ".env.example",
    ".gitignore":       ".gitignore",
    "makefile":         "Makefile",
}


# ─── Route query ─────────────────────────────────────────────────────────────

def route_query(q: str, registry: ProjectRegistry, ctx: Dict) -> List[Dict]:
    ql = q.lower()

    # ── URL query fast-path (Issue C fix) ─────────────────────────────────────
    # When user explicitly asks for a URL/web URL, we need the raw value not
    # a hyperlink-formatted anchor. Flag this in params so the display layer
    # returns the raw string. Route to get_issue so url field is returned.
    _URL_ASK_RE = re.compile(r"\b(web.?url|url|link|browse.?link)\b", re.I)
    if _URL_ASK_RE.search(q):
        ikm_url = re.search(r"\b([A-Z]{2,8}-\d+)\b", q)
        if ikm_url:
            return [{"source": "jira", "operation": "get_issue",
                     "params": {"issue_key": ikm_url.group(1)},
                     "_return_raw_url": True}]  # hint to display layer

    # 0. Re-check / confirmation queries ("are you sure?", "check again", "verify")
    #    Re-dispatch the most recent sentry call using stored context.
    #    Must be rule 0 so it fires before any other pattern match.
    if _RECHECK_RE.search(q):
        calls = []
        # Re-fetch last Jira project
        if ctx.get("last_jira_key"):
            calls.append({"source": "jira", "operation": "get_project_issues",
                          "params": {"project_key": ctx["last_jira_key"], "limit": 100}})
        # Re-fetch last GitLab project — issues + commits + MRs
        if ctx.get("last_project"):
            calls.append({"source": "gitlab", "operation": "get_issues",
                          "params": {"project": ctx["last_project"], "state": "all", "limit": 100}})
            calls.append({"source": "gitlab", "operation": "get_commits",
                          "params": {"project": ctx["last_project"], "limit": 50}})
        if calls:
            return calls
        # No context — fall through to normal routing

    # 1. Security issues
    if _SECURITY_RE.search(q):
        proj = _find_project(q, registry)
        calls: List[Dict] = []
        if proj:
            calls.append({"source": "gitlab", "operation": "get_issues",
                          "params": {"project": proj, "state": "all", "limit": 30}})
        else:
            for p in registry.personal_projects():
                calls.append({"source": "gitlab", "operation": "get_issues",
                               "params": {"project": p, "state": "all", "limit": 15}})
        jk = registry.find_jira_key(q)
        if jk:
            calls.append({"source": "jira", "operation": "get_issues",
                          "params": {"jql": f'project = "{jk}" AND text ~ "security"',
                                     "limit": 10}})
        return calls

    # 2. Cross-project MR fan-out
    if _ALL_MR_RE.search(q):
        state = _state(q)
        return [{"source": "gitlab", "operation": "get_merge_requests",
                 "params": {"project": p, "state": state}}
                for p in registry.personal_projects()]

    # 3. Cross-project commit fan-out
    if _ALL_COMMITS_RE.search(q):
        return [{"source": "gitlab", "operation": "get_commits",
                 "params": {"project": p, "limit": 5}}
                for p in registry.personal_projects()]

    # 4. Compound commits AND merge requests
    if _COMPOUND_COMMIT_MR_RE.search(q):
        proj  = _find_project(q, registry) or ctx.get("last_project")
        state = _state(q)
        calls = []
        if proj:
            calls.append({"source": "gitlab", "operation": "get_merge_requests",
                          "params": {"project": proj, "state": state}})
            calls.append({"source": "gitlab", "operation": "get_commits",
                          "params": {"project": proj, "limit": 15}})
        else:
            for p in registry.personal_projects():
                calls.append({"source": "gitlab", "operation": "get_merge_requests",
                              "params": {"project": p, "state": state}})
                calls.append({"source": "gitlab", "operation": "get_commits",
                              "params": {"project": p, "limit": 5}})
        if calls:
            return calls

    # 5. Issues + project keyword collision guard
    if _ISSUE_RE.search(q) and _LIST_PROJ_RE.search(q):
        proj = _find_project(q, registry)
        jk   = registry.find_jira_key(q)
        calls = []
        state = _state(q)
        # Issue B fix: if a Jira project key/name is found, route Jira first.
        # Only route to GitLab if the query explicitly mentions 'gitlab' or
        # the project was found in the GitLab registry but NOT in Jira.
        gitlab_explicit = bool(re.search(r"\bgitlab\b", q, re.I))
        if jk:
            jp: Dict[str, Any] = {"project_key": jk}
            calls.append({"source": "jira", "operation": "get_project_issues", "params": jp})
        if proj and (gitlab_explicit or not jk):
            calls.append({"source": "gitlab", "operation": "get_issues",
                          "params": {"project": proj, "state": state}})
        if calls:
            return calls

    # 6. Milestones
    if _MILESTONE_RE.search(ql):
        proj   = _find_project(q, registry)
        params = {"project": proj} if proj else {}
        return [{"source": "gitlab", "operation": "get_milestones", "params": params}]

    # 7. List / count all projects
    if _LIST_PROJ_RE.search(q):
        return [{"source": "gitlab", "operation": "list_projects", "params": {}}]

    # 8. Specific named file
    fm = _SPECIFIC_FILE_RE.search(q)
    if fm:
        proj = _find_project(q, registry) or ctx.get("last_project")
        if proj:
            fname = _FILE_NAME_MAP.get(fm.group(0).lower(), fm.group(0))
            return [{"source": "gitlab", "operation": "get_file",
                     "params": {"project": proj, "file_path": fname}}]

    # 9. Env vars
    if _ENV_VARS_RE.search(q):
        proj = _find_project(q, registry) or ctx.get("last_project")
        if proj:
            return [{"source": "gitlab", "operation": "get_file",
                     "params": {"project": proj, "file_path": ".env.example"}},
                    {"source": "gitlab", "operation": "get_file",
                     "params": {"project": proj, "file_path": "README.md"}}]

    # 10. Named code file
    ncf = _NAMED_CODE_FILE_RE.search(q)
    if ncf:
        proj = _find_project(q, registry) or ctx.get("last_project")
        if proj:
            return [{"source": "gitlab", "operation": "get_file",
                     "params": {"project": proj, "file_path": ncf.group(1)}}]

    # 11. Project existence check
    _PROJECT_EXIST_RE = re.compile(
        r"\b(is there a project|project called|project named|"
        r"project.{0,15}exist|find a project|"
        r"project.{0,10}called|called.{0,10}project)\b", re.I)
    if _PROJECT_EXIST_RE.search(q):
        proj = _find_project(q, registry)
        if proj:
            return [{"source": "gitlab", "operation": "get_project_info",
                     "params": {"project": proj}}]
        return [{"source": "gitlab", "operation": "list_projects", "params": {}}]

    # 12. Branches
    if _BRANCH_RE.search(q):
        proj = _find_project(q, registry) or ctx.get("last_project")
        if proj:
            return [{"source": "gitlab", "operation": "get_branches",
                     "params": {"project": proj}}]
        return [{"source": "gitlab", "operation": "get_branches",
                 "params": {"project": p}}
                for p in registry.personal_projects()]

    # 13. File tree
    if _FILE_TREE_RE.search(q):
        proj = _find_project(q, registry) or ctx.get("last_project")
        if proj:
            return [{"source": "gitlab", "operation": "get_repository_tree",
                     "params": {"project": proj}}]

    # 14. Pipelines
    if _PIPELINE_RE.search(q):
        proj = _find_project(q, registry) or ctx.get("last_project")
        if proj:
            return [{"source": "gitlab", "operation": "get_pipelines",
                     "params": {"project": proj, "limit": 10}}]

    # 15. Commits (single project)
    if _COMMIT_RE.search(q):
        proj = _find_project(q, registry) or ctx.get("last_project")
        if proj:
            return [{"source": "gitlab", "operation": "get_commits",
                     "params": {"project": proj, "limit": 15}}]

    # 16. Merge requests
    if _MR_RE.search(q):
        proj  = _find_project(q, registry) or ctx.get("last_project")
        state = _state(q)
        if proj:
            return [{"source": "gitlab", "operation": "get_merge_requests",
                     "params": {"project": proj, "state": state}}]
        return [{"source": "gitlab", "operation": "get_merge_requests",
                 "params": {"project": p, "state": state}}
                for p in registry.personal_projects()]

    # 17. Code structure search
    if _CODE_SEARCH_RE.search(q):
        proj = _find_project(q, registry) or ctx.get("last_project")
        keyword = _CODE_SEARCH_RE.search(q).group(0).split()[-1]
        if proj:
            return [
                {"source": "gitlab", "operation": "search_code",
                 "params": {"query": keyword, "project": proj, "limit": 10}},
                {"source": "gitlab", "operation": "get_repository_tree",
                 "params": {"project": proj, "recursive": True, "limit": 50}},
            ]
        return [{"source": "gitlab", "operation": "search_code",
                 "params": {"query": keyword, "limit": 10}}]

    # 18. Project info / metadata
    if _INFO_RE.search(q):
        proj = _find_project(q, registry) or ctx.get("last_project")
        if proj:
            return [{"source": "gitlab", "operation": "get_project_info",
                     "params": {"project": proj}}]

    # 19b. Reporter / author fast-path — "tickets where Dorian You is the author"
    #      Fires when there is a clear reporter keyword but no other strong signal
    #      (e.g. no explicit project, no Jira project key). Builds a cross-project
    #      JQL query so results come back even without a project key in the query.
    if _REPORTER_KEYWORD_RE.search(q):
        reporter_name = _extract_reporter(q)
        if reporter_name:
            jk = registry.find_jira_key(q)
            if jk:
                return [{"source": "jira", "operation": "get_project_issues",
                         "params": {"project_key": jk, "reporter": reporter_name, "limit": 100}}]
            # No project key — use raw JQL across all projects
            jql = f'reporter = "{reporter_name}" ORDER BY updated DESC'
            return [{"source": "jira", "operation": "get_issues",
                     "params": {"jql": jql, "limit": 100}}]

    # 19c. WIP (was rule 19 — kept intact, number shifted for clarity only)
    if _WIP_RE.search(q):
        calls = []
        for p in registry.personal_projects():
            calls.append({"source": "gitlab", "operation": "get_merge_requests",
                          "params": {"project": p, "state": "opened"}})
        jk   = registry.find_jira_key(q)
        keys = [jk] if jk else registry.all_jira_keys()
        if keys:
            # Cap fanout at 10 projects — with 424k issues across many projects,
            # fanning out to ALL keys generates 50+ simultaneous API calls.
            # Prefer active projects (already sorted by last_activity in load_gitlab).
            capped_keys = keys[:10]
            for k in capped_keys:
                calls.append({"source": "jira", "operation": "get_project_issues",
                              "params": {"project_key": k, "status": "In Progress",
                                         "limit": 10}})
        else:
            calls.append({"source": "jira", "operation": "get_issues",
                          "params": {"jql": 'statusCategory = "In Progress" ORDER BY updated DESC',
                                     "limit": 25}})
        return calls

    # 19c-gl. GitLab MR refs (!123) and issue refs (#456) fast-path.
    # These must be resolved before the Jira key fast-path (19d) because they
    # use a different regex and a different API operation.
    # !N  → get_mr_diff (user wants the MR content / diff)
    # #N  → get_issues lookup on the inferred project
    # Both require a known project; if none is resolvable we fall through.
    _GL_MR_REF_RE    = re.compile(r'!(\d{1,6})')
    _GL_ISSUE_REF_RE = re.compile(r'#(\d{1,6})')
    gl_mr_match    = _GL_MR_REF_RE.search(q)
    gl_issue_match = _GL_ISSUE_REF_RE.search(q)
    if gl_mr_match or gl_issue_match:
        proj = _find_project(q, registry) or ctx.get("last_project")
        calls: List[Dict] = []
        if gl_mr_match and proj:
            mr_iid = int(gl_mr_match.group(1))
            calls.append({
                "source":    "gitlab",
                "operation": "get_mr_diff",
                "params":    {"project": proj, "mr_iid": mr_iid},
            })
            # Also fetch the MR metadata so the model has title/state/author
            calls.append({
                "source":    "gitlab",
                "operation": "get_merge_requests",
                "params":    {"project": proj, "state": "all", "limit": 50},
            })
        if gl_issue_match and proj:
            issue_id = int(gl_issue_match.group(1))
            calls.append({
                "source":    "gitlab",
                "operation": "get_issues",
                "params":    {"project": proj, "search": str(issue_id), "limit": 10},
            })
        if calls:
            return calls
        # No project resolved — fall through to broader rules

    # 19d. Specific Jira issue key(s) — MUST come before rule 20.
    #
    #      Rule 20 extracts the project prefix ("TM" from "PROJ-18487") via
    #      registry.find_jira_key() and calls get_project_issues(), returning
    #      100 items instead of the one ticket the user asked about.
    #
    #      Any query that contains a full Jira key (PROJECT-NUMBER) should
    #      resolve each key with a direct get_issue lookup — regardless of
    #      surrounding words ("list PROJ-18487", "summarize PROJ-18133",
    #      "what is LONGKEY-12355", "show PROJ-18487 and PROJ-18490").
    #
    _SPECIFIC_KEY_RE = re.compile(r'\b([A-Z]{2,8}-\d+)\b')
    specific_keys = _SPECIFIC_KEY_RE.findall(q)
    if specific_keys:
        return [
            {"source": "jira", "operation": "get_issue",
             "params": {"issue_key": key}}
            for key in dict.fromkeys(specific_keys)   # preserve order, deduplicate
        ]

    # 20. Issues (GitLab and/or Jira)
    if _ISSUE_RE.search(q):
        calls = []
        state = _state(q)
        proj  = _find_project(q, registry)
        jk    = registry.find_jira_key(q)
        jira_explicit = bool(_JIRA_RE.search(q))

        # Reporter/author extraction — checked once, used in all Jira paths below
        reporter_name = _extract_reporter(q) if _REPORTER_KEYWORD_RE.search(q) else None

        # FIX 2/5 — date filter and sort order extraction
        created_after, created_before = _extract_date_filter(q)
        sort_by = "created" if _RECENT_RE.search(q) else None

        # "jira tickets/issues" with no specific project -> fan out all Jira projects
        if jira_explicit and not jk:
            jira_st   = _jira_status(q)
            jira_pri  = _jira_priority(q)
            jira_keys = registry.all_jira_keys()

            if jira_keys:
                # Normal path: fan out per-project
                for k in jira_keys:
                    jp_fanout: Dict[str, Any] = {"project_key": k, "limit": 10}
                    if jira_st:       jp_fanout["status"]        = jira_st
                    if jira_pri:      jp_fanout["priority"]       = jira_pri
                    if reporter_name: jp_fanout["reporter"]       = reporter_name
                    if created_after: jp_fanout["created_after"]  = created_after
                    if created_before:jp_fanout["created_before"] = created_before
                    if sort_by:       jp_fanout["sort_by"]        = sort_by
                    calls.append({"source": "jira", "operation": "get_project_issues",
                                   "params": jp_fanout})
            else:
                # Fallback: Jira registry didn't load — query without a project key
                jql_parts = []
                if jira_st:        jql_parts.append(_jira_status_jql(jira_st))
                if jira_pri:       jql_parts.append(f'priority = "{jira_pri}"')
                if reporter_name:  jql_parts.append(f'reporter = "{reporter_name}"')
                if created_after:  jql_parts.append(f'created >= "{created_after}"')
                if created_before: jql_parts.append(f'created <= "{created_before}"')
                order = "created DESC" if (sort_by == "created" or created_after or created_before) else "updated DESC"
                suffix = f" ORDER BY {order}"
                jql = (" AND ".join(jql_parts) + suffix) if jql_parts else ("project is not EMPTY" + suffix)
                calls.append({"source": "jira", "operation": "get_issues",
                               "params": {"jql": jql, "limit": 100}})
            return calls

        if proj:
            p: Dict[str, Any] = {"project": proj, "state": state}
            if "bug" in ql:      p["labels"] = ["bug"]
            if "critical" in ql: p["labels"] = ["critical"]
            if "urgent" in ql:   p["labels"] = ["urgent"]
            if "blocked" in ql:  p["labels"] = ["blocked"]
            calls.append({"source": "gitlab", "operation": "get_issues", "params": p})
            if jira_explicit and jk:
                jp: Dict[str, Any] = {"project_key": jk}
                jira_st  = _jira_status(q)
                jira_pri = _jira_priority(q)
                if jira_st:        jp["status"]        = jira_st
                if jira_pri:       jp["priority"]       = jira_pri
                if reporter_name:  jp["reporter"]       = reporter_name
                if created_after:  jp["created_after"]  = created_after
                if created_before: jp["created_before"] = created_before
                if sort_by:        jp["sort_by"]        = sort_by
                calls.append({"source": "jira", "operation": "get_project_issues", "params": jp})
        else:
            if not jira_explicit:
                for personal in registry.personal_projects():
                    calls.append({"source": "gitlab", "operation": "get_issues",
                                   "params": {"project": personal, "state": state, "limit": 10}})
            if jk:
                jp2: Dict[str, Any] = {"project_key": jk}
                if "bug" in ql:    jp2["issue_type"] = "Bug"
                if "epic" in ql:   jp2["issue_type"] = "Epic"
                if "story" in ql:  jp2["issue_type"] = "Story"
                jira_st2  = _jira_status(q)
                jira_pri2 = _jira_priority(q)
                if jira_st2:       jp2["status"]        = jira_st2
                if jira_pri2:      jp2["priority"]       = jira_pri2
                if reporter_name:  jp2["reporter"]       = reporter_name
                if created_after:  jp2["created_after"]  = created_after
                if created_before: jp2["created_before"] = created_before
                if sort_by:        jp2["sort_by"]        = sort_by
                calls.append({"source": "jira", "operation": "get_project_issues", "params": jp2})

        if calls:
            return calls

    # 21. Confluence — only reached when NO explicit Jira issue keys are present
    # (rule 19d above handles those and returns early).
    # Trigger: word-boundary safe — avoids "docker", "homepage", "namespace".
    if _CONFLUENCE_RE.search(q) or _CONFLUENCE_TRIGGER_RE.search(q):

        # 21a. Generic space catalogue
        if _CONFLUENCE_LIST_SPACES_RE.search(q):
            return [{"source": "confluence", "operation": "get_spaces", "params": {}}]

        # 21b. Recently updated pages
        if _CONFLUENCE_RECENT_RE.search(q):
            sk_match  = _CONFLUENCE_SPACE_KEY_RE.search(q)
            space_key = (sk_match.group(1) or sk_match.group(2)) if sk_match else None
            params: Dict[str, Any] = {"days": 7, "limit": 20}
            if space_key:
                params["space_key"] = space_key
            return [{"source": "confluence", "operation": "get_recent_pages",
                     "params": params}]

        # 21c. Full-text search — the common case
        sk_match  = _CONFLUENCE_SPACE_KEY_RE.search(q)
        space_key = (sk_match.group(1) or sk_match.group(2)) if sk_match else None
        terms     = _clean_confluence_query(q)
        if not terms or len(terms) < 3:
            terms = _confluence_search_terms(q)   # fallback to filler-strip
        if not terms or len(terms) < 3:
            terms = q
        params = {"query": terms, "limit": 10}
        if space_key:
            params["space_key"] = space_key
        return [{"source": "confluence", "operation": "search_pages",
                 "params": params}]

    # 22. Single Jira issue key (e.g. AUTH-12)
    ikm = re.search(r"\b([A-Z]{2,5}-\d+)\b", q)
    if ikm:
        return [{"source": "jira", "operation": "get_issue",
                 "params": {"issue_key": ikm.group(1)}}]

    # 23. Jira project key / name mentioned alone
    jk = registry.find_jira_key(q)
    if jk:
        return [{"source": "jira", "operation": "get_project_issues",
                 "params": {"project_key": jk}}]

    return []


# ─── Deterministic count function ────────────────────────────────────────────

def _result_count(r) -> Tuple[int, bool]:
    meta = getattr(r, "meta", {}) or {}
    for key in ("total_in_jira", "total_in_sprint"):
        if key in meta:
            return int(meta[key]), True
    returned = len(r.data)
    limit    = meta.get("limit", None)
    inexact  = (limit is not None and returned >= limit)
    return returned, not inexact


def count_results(query: str, results: List) -> str:
    if not results:
        return "No data was retrieved -- could not count."

    error_srcs = [r.source for r in results if not r.success]
    success    = [r for r in results if r.success]

    if not success:
        msg = "0 results found"
        if error_srcs:
            msg += f" (errors from: {', '.join(set(error_srcs))})"
        return msg + "."

    grand_total = 0
    any_inexact = False
    lines_body  = []

    # Dedup: skip result sets sharing (source, operation, project).
    # Prevents double-counting when fan-out accidentally repeats a project.
    seen_keys: set = set()

    for r in success:
        if not r.data:
            continue

        sample   = r.data[0] if r.data else {}
        proj_key = (
            sample.get("project_name")
            or sample.get("project_key")
            or sample.get("project")
            or ""
        )
        dedup_key = (r.source, r.operation, proj_key)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        cnt, exact = _result_count(r)
        grand_total += cnt
        if not exact:
            any_inexact = True

        proj_label = f" [{proj_key}]" if proj_key else ""
        suffix     = "" if exact else "+"
        lines_body.append(f"  * {r.source.upper()}{proj_label} -- {r.operation}: {cnt}{suffix}")

    if grand_total == 0:
        return "0 results found."

    total_str = f"{grand_total}+" if any_inexact else str(grand_total)
    lines = [f"Total: {total_str}\n"] + lines_body

    jira_st  = _jira_status(query)
    jira_pri = _jira_priority(query)
    if jira_st or jira_pri:
        filters = []
        if jira_st:  filters.append(f"status={jira_st}")
        if jira_pri: filters.append(f"priority={jira_pri}")
        lines.append(f"\n  (filtered by: {', '.join(filters)})")

    if any_inexact:
        lines.append("  (+ = fetch limit reached; actual count may be higher)")

    return "\n".join(lines)


# ─── Response synthesizer ─────────────────────────────────────────────────────

SYNTHESIS_PROMPT = """\
/no_think
Respond in the same language as the user's question.
You are a helpful AI assistant for a software engineering team.
Real-time data was just retrieved from GitLab, Jira, and/or Confluence APIs.

User question: {query}

Retrieved data:
{data}

Instructions:
- Answer directly using ONLY the data above -- never invent anything
- Include IDs, titles, states, authors, dates, and URLs where available
- If data is empty or an error occurred, say so clearly
- Use numbered lists for multiple items
- Be concise -- no padding or preamble

Answer:"""

# Fields always suppressed in synthesized output.
# NOTE: "description" and "body" are intentionally absent — Jira ticket
# descriptions and Confluence page bodies are the primary answer content.
# Suppressing them caused the LLM to answer "what is this ticket about?"
# with no information. "content" and "diff" handled per-operation below.
_SKIP_FIELDS = {"text", "message", "snippet", "comments", "changelog"}

def synthesize_response(query: str, results: List, queries: List[Dict]) -> str:
    is_fanout = len(results) > 3

    parts = []
    for r, q in zip(results, queries):
        header = f"[{r.source.upper()} -> {r.operation}]"
        if not r.success:
            if not is_fanout:
                parts.append(f"{header}\nError: {r.error}")
            continue
        if not r.data:
            if not is_fanout:
                parts.append(f"{header}\nNo results found.")
            continue

        item_limit = 3 if is_fanout else 30   # raised from 15 — was silently dropping 35 items
        rows = []

        is_file_op = (r.operation == "get_file")
        is_diff_op = (r.operation == "get_mr_diff")

        for item in r.data[:item_limit]:
            lines = []
            for k, v in item.items():
                if v is None or v == "" or v == []:
                    continue
                if k in _SKIP_FIELDS:
                    continue
                if k == "content" and not is_file_op:
                    continue
                if k == "diff" and not is_diff_op:
                    continue
                s = str(v)
                if k == "diff":
                    cap = 3000
                elif k == "content":
                    cap = 2000
                elif k in ("description", "body"):
                    # Ticket descriptions and page bodies: useful but must be
                    # capped so fan-out results don't blow the context window.
                    cap = 120 if is_fanout else 400
                else:
                    cap = 120 if is_fanout else 250
                if len(s) > cap:
                    s = s[:cap] + "..."
                lines.append(f"  {k}: {s}")
            if lines:
                rows.append("\n".join(lines))

        if not rows:
            continue

        parts.append(f"{header}\n" + "\n---\n".join(rows))
        remaining = len(r.data) - item_limit
        if remaining > 0:
            parts.append(f"  ... and {remaining} more items")

    data = "\n\n".join(parts) if parts else "No data retrieved from any source."
    return ollama_generate(
        SYNTHESIS_PROMPT.format(query=query, data=data),
        max_tokens=1800,
        temperature=0.2,
    )


# ─── Conversation context ─────────────────────────────────────────────────────

class Context:
    def __init__(self):
        self.last_project:  Optional[str] = None
        self.last_jira_key: Optional[str] = None

    def update(self, queries: List[Dict]):
        for q in queries:
            p = q.get("params", {})
            if q.get("source") == "gitlab" and "project" in p:
                self.last_project = p["project"]
            if q.get("source") == "jira" and "project_key" in p:
                self.last_jira_key = p["project_key"]

    def as_dict(self) -> Dict:
        return {"last_project": self.last_project, "last_jira_key": self.last_jira_key}


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(
    query: str,
    dispatcher: SentryDispatcher,
    registry: ProjectRegistry,
    ctx: Context,
    verbose: bool = False,
) -> Tuple[str, List]:

    if verbose:
        print("  Routing...", end=" ", flush=True)

    queries = route_query(query, registry, ctx.as_dict())

    if not queries:
        examples = "\n".join(f"  * {p}" for p in registry.all_paths()[:4])
        return (
            "I couldn't determine which API to call.\n\n"
            f"Available projects:\n{examples}\n\n"
            "Try: 'Show open issues in auth-service' or "
            "'List merge requests in ecommerce-backend'",
            [],
        )

    if verbose:
        print(f"-> {len(queries)} call(s)")
        for q in queries:
            print(f"    {q['source']}.{q['operation']}({q.get('params', {})})")

    if verbose:
        print("  Fetching...", end=" ", flush=True)

    results = dispatcher.dispatch_multi(queries)
    ctx.update(queries)

    if verbose:
        total = sum(len(r.data) for r in results if r.success)
        errs  = sum(1 for r in results if not r.success)
        print(f"-> {total} items, {errs} errors")

    if is_pure_count(query):
        if verbose:
            print("  Counting (no LLM)...")
        return count_results(query, results), results

    if verbose:
        print("  Synthesizing...", end=" ", flush=True)

    answer = synthesize_response(query, results, queries)

    if verbose:
        print("done\n")

    return answer, results


# ─── CLI ──────────────────────────────────────────────────────────────────────

HEADER = """
+======================================================================+
|      REAL-TIME AI ASSISTANT  (Ollama + API Sentries)  v4             |
+======================================================================+
|  Live data from:  GitLab  *  Jira  *  Confluence                     |
+======================================================================+
|  Commands:  /help  /projects  /status  /verbose  /sources  quit      |
+======================================================================+
"""

HELP_TEXT = """
Example Queries
------------------------------------------------------------------------
GitLab:
  * List all projects on GitLab
  * How many projects do I have?
  * Show open issues in auth-service
  * List all merge requests in notification-service
  * Show recent commits in ecommerce-frontend
  * Show pipeline status for ecommerce-backend
  * What files are in auth-service?

Jira:
  * How many jira tickets are open?
  * How many jira tickets are closed?
  * How many urgent jira tickets are there?
  * Count blocked tickets
  * Count critical bugs in AUTH
  * Show in-progress tickets in NOTIF
  * Show epics in AUTH / Show stories in ECOM
  * What is AUTH-12?

Confluence:
  * Search Confluence for "deployment"
  * List all Confluence spaces

Cross-platform:
  * Show all security issues across all projects
  * What work is in progress right now?
  * How many open issues do I have total?
------------------------------------------------------------------------
"""


def print_sources(results: List):
    if not results or not any(r.success and r.data for r in results):
        return
    print("\n  Sources:")
    for r in results:
        ok    = "OK" if r.success else "FAIL"
        count = len(r.data) if r.success else 0
        err   = f" -> {r.error[:60]}" if not r.success and r.error else ""
        print(f"    [{ok}] [{r.source.upper()}] {r.operation}  ({count} items){err}")


def main():
    print(HEADER)

    print("Checking Ollama...", end=" ", flush=True)
    if not check_ollama():
        print(f"\nOllama unreachable or model '{OLLAMA_MODEL}' missing.")
        print(f"  -> ollama serve  /  ollama pull {OLLAMA_MODEL}")
        sys.exit(1)
    print(f"OK: {OLLAMA_MODEL} ready\n")

    print("Initializing sentries...", end=" ", flush=True)
    dispatcher = SentryDispatcher(verbose=False)
    st = dispatcher.status()
    print(f"OK: {st['available']}")
    if st["unavailable"]:
        print(f"  Warning: Unavailable: {st['unavailable']}")

    print("Loading project registry...", end=" ", flush=True)
    registry = ProjectRegistry()
    registry.load(dispatcher)
    gl_count   = len(registry.all_paths())
    jira_count = len(registry.jira_projects)
    print(f"OK: {gl_count} GitLab project(s), {jira_count} Jira project(s)")
    print()

    ctx     = Context()
    verbose = False
    count   = 0

    while True:
        try:
            tag   = " [v]" if verbose else ""
            query = input(f"You{tag}> ").strip()
            if not query:
                continue

            qlow = query.lower()
            if qlow in ("quit", "exit", "q"):
                print(f"\nGoodbye! ({count} questions answered)\n"); break
            if qlow == "/help":
                print(HELP_TEXT); continue
            if qlow == "/verbose":
                verbose = not verbose
                print(f"  Verbose: {'ON' if verbose else 'OFF'}\n"); continue
            if qlow == "/status":
                s = dispatcher.status()
                print(f"\n  Available  : {s['available']}")
                print(f"  Unavailable: {s['unavailable']}\n"); continue
            if qlow == "/projects":
                print("\n  GitLab projects:")
                for p in registry.all_paths():
                    print(f"    * {p}")
                print("\n  Jira projects:")
                for k, name in registry.jira_projects.items():
                    print(f"    * {k}  --  {name}")
                print(); continue
            if qlow == "/sources":
                for src, ops in OPERATION_CATALOGUE.items():
                    print(f"\n  [{src.upper()}]")
                    for e in ops:
                        print(f"    {e['op']:<32} {e['params']}")
                print(); continue

            count += 1
            print()
            if verbose:
                print("-" * 70)

            answer, results = run_pipeline(query, dispatcher, registry, ctx, verbose)

            print("Answer:")
            print("-" * 70)
            if answer.startswith("ERROR:OLLAMA_DOWN"):
                print("  Ollama not responding. Is it running?")
            elif answer.startswith("ERROR:"):
                print(f"  {answer[6:]}")
            else:
                for line in answer.splitlines():
                    print(f"  {line}")

            print_sources(results)
            print()
            print("=" * 70)
            print()

        except KeyboardInterrupt:
            print(f"\n\nGoodbye! ({count} questions answered)\n"); break
        except Exception as exc:
            print(f"\n  Error: {exc}")
            import traceback; traceback.print_exc()
            print()


if __name__ == "__main__":
    main()
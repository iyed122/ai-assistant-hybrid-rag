#!/usr/bin/env python3
"""
Sentry Dispatcher
─────────────────
Central router that receives a structured query from the Intent Agent
and routes it to the correct API Sentry.

Design principles
─────────────────
  • No LLM here — purely deterministic routing on query["source"] + query["operation"]
  • Lazy sentry instantiation (only connect to APIs that are needed + configured)
  • Returns SentryResult envelopes ready for Weaver consumption
  • Thread-safe singleton sentries (one session per sentry)

Usage (from Intent Agent / LangGraph node)
──────────────────────────────────────────
    from sentries.sentry_dispatcher import SentryDispatcher

    dispatcher = SentryDispatcher()

    result = dispatcher.dispatch({
        "source": "gitlab",
        "operation": "get_issues",
        "params": {
            "project": "mygroup/myrepo",
            "state":   "opened",
            "labels":  ["bug"],
            "limit":   30,
        }
    })

    if result.success:
        for item in result.data:
            print(item)
    else:
        print("Error:", result.error)
"""

import logging
from typing import Any, Dict, List, Optional

from sentries.base_sentry import SentryResult
from sentries.gitlab_sentry import GitLabSentry
from sentries.jira_sentry import JiraSentry
from sentries.confluence_sentry import ConfluenceSentry

logger = logging.getLogger("sentry.dispatcher")


# ──────────────────────────────────────────────────────────────────────────────
# Route table: source → operation → (sentry_attr, method_name, required_params)
# ──────────────────────────────────────────────────────────────────────────────

ROUTE_TABLE: Dict[str, Dict[str, str]] = {
    "gitlab": {
        "list_projects":        "list_projects",
        "get_issues":           "get_issues",
        "get_merge_requests":   "get_merge_requests",
        "get_mr_diff":          "get_mr_diff",
        "get_pipelines":        "get_pipelines",
        "get_commits":          "get_commits",
        "get_file":             "get_file",
        "get_project_info":     "get_project_info",
        "get_milestones":       "get_milestones",
        "get_repository_tree":  "get_repository_tree",
        "search_code":          "search_code",
        "get_branches":         "get_branches",
    },
    "jira": {
        "list_projects":      "list_projects",
        "get_issues":         "get_issues",
        "get_project_issues": "get_project_issues",
        "get_issue":          "get_issue",
        "get_issue_changelog":"get_issue_changelog",
        "get_boards":         "get_boards",
        "get_sprint_issues":  "get_sprint_issues",
        "get_user_issues":    "get_user_issues",
    },
    "confluence": {
        "get_spaces":       "get_spaces",
        "get_pages":        "get_pages",
        "get_page":         "get_page",
        "get_page_comments":"get_page_comments",
        "search_pages":     "search_pages",
        "get_recent_pages": "get_recent_pages",
        "get_child_pages":  "get_child_pages",
    },
}

# Human-readable operation catalogue (shown by list_operations)
OPERATION_CATALOGUE: Dict[str, List[Dict]] = {
    "gitlab": [
        {"op": "list_projects",       "params": "[owned, membership, search, limit]"},
        {"op": "get_issues",          "params": "project, [state, labels, assignee, search, limit]"},
        {"op": "get_merge_requests",  "params": "project, [state, target_branch, author, search, limit]"},
        {"op": "get_mr_diff",         "params": "project, mr_iid"},
        {"op": "get_pipelines",       "params": "project, [ref, status, limit]"},
        {"op": "get_commits",         "params": "project, [branch, file_path, since, limit]"},
        {"op": "get_file",            "params": "project, file_path, [ref]"},
        {"op": "get_project_info",    "params": "project"},
        {"op": "get_milestones",      "params": "[project, state, search, limit]"},
        {"op": "get_repository_tree", "params": "project, [path, ref, recursive, limit]"},
        {"op": "search_code",         "params": "query, [project, limit]"},
        {"op": "get_branches",        "params": "project, [search, limit]"},
    ],
    "jira": [
        {"op": "list_projects",       "params": "[limit]"},
        {"op": "get_issues",          "params": "jql, [limit]"},
        {"op": "get_project_issues",  "params": "project_key, [status, assignee, issue_type, priority, search, limit]"},
        {"op": "get_issue",           "params": "issue_key, [include_changelog]"},
        {"op": "get_issue_changelog", "params": "issue_key"},
        {"op": "get_boards",          "params": "[project_key, limit]"},
        {"op": "get_sprint_issues",   "params": "board_id, [sprint_name, limit]"},
        {"op": "get_user_issues",     "params": "assignee, [project_key, status, limit]"},
    ],
    "confluence": [
        {"op": "get_spaces",       "params": "[limit]"},
        {"op": "get_pages",        "params": "space_key, [title_contains, limit]"},
        {"op": "get_page",         "params": "page_id, [include_body, include_comments]"},
        {"op": "get_page_comments","params": "page_id"},
        {"op": "search_pages",     "params": "query, [space_key, limit]"},
        {"op": "get_recent_pages", "params": "[space_key, days, limit]"},
        {"op": "get_child_pages",  "params": "parent_page_id, [limit]"},
    ],
}


class SentryDispatcher:
    """
    Routes structured queries from the Intent Agent to the right API sentry.

    Sentries are initialised lazily on first use; if credentials for a
    source are missing, that source's sentry is marked unavailable.
    """

    def __init__(self, verbose: bool = False):
        self._sentries: Dict[str, Any] = {}
        self._unavailable: Dict[str, str] = {}
        self.verbose = verbose

        if verbose:
            logging.basicConfig(
                level=logging.DEBUG,
                format="[%(asctime)s] [%(name)s] %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

    # ── Sentry accessor (lazy init) ──────────────────────────────────────────

    def _get_sentry(self, source: str):
        if source in self._sentries:
            return self._sentries[source]

        if source in self._unavailable:
            return None

        cls_map = {
            "gitlab":     GitLabSentry,
            "jira":       JiraSentry,
            "confluence": ConfluenceSentry,
        }
        cls = cls_map.get(source)
        if not cls:
            self._unavailable[source] = f"Unknown source '{source}'"
            return None

        try:
            sentry = cls()
            self._sentries[source] = sentry
            logger.info("✓ %s sentry initialised", source)
            return sentry
        except (ValueError, ImportError) as exc:
            self._unavailable[source] = str(exc)
            logger.warning("✗ %s sentry unavailable: %s", source, exc)
            return None

    # ── Core dispatch ────────────────────────────────────────────────────────

    def dispatch(self, query: Dict[str, Any]) -> SentryResult:
        """
        Route a query dict to the correct sentry method.

        Expected query shape:
        {
            "source":    "gitlab" | "jira" | "confluence",
            "operation": "<operation_name>",
            "params":    { ... kwargs for the operation ... }   # optional
        }

        Returns a SentryResult (always, even on error).
        """
        source    = (query.get("source") or "").lower().strip()
        operation = (query.get("operation") or "").lower().strip()
        params    = query.get("params") or {}

        # Validate source
        if source not in ROUTE_TABLE:
            return SentryResult(
                success=False,
                source=source or "unknown",
                operation=operation or "unknown",
                error=f"Unknown source '{source}'. Valid sources: {list(ROUTE_TABLE)}",
            )

        # Validate operation
        method_name = ROUTE_TABLE[source].get(operation)
        if not method_name:
            valid_ops = list(ROUTE_TABLE[source].keys())
            return SentryResult(
                success=False,
                source=source,
                operation=operation,
                error=f"Unknown operation '{operation}' for '{source}'. Valid: {valid_ops}",
            )

        # Get sentry
        sentry = self._get_sentry(source)
        if sentry is None:
            reason = self._unavailable.get(source, "Not configured")
            return SentryResult(
                success=False,
                source=source,
                operation=operation,
                error=f"{source} sentry unavailable: {reason}",
            )

        # Call the sentry method
        method = getattr(sentry, method_name)
        try:
            result: SentryResult = method(**params)
        except TypeError as exc:
            return SentryResult(
                success=False,
                source=source,
                operation=operation,
                error=f"Invalid params for '{operation}': {exc}",
            )

        return result

    # ── Multi-source dispatch ────────────────────────────────────────────────

    def dispatch_multi(self, queries: List[Dict[str, Any]]) -> List[SentryResult]:
        """
        Run multiple queries in sequence and return all results.
        Useful when the Intent Agent determines that BOTH Qdrant and live APIs are needed.

        Args:
            queries: List of query dicts (same format as dispatch)

        Returns:
            List of SentryResult objects in the same order
        """
        return [self.dispatch(q) for q in queries]

    # ── Introspection ────────────────────────────────────────────────────────

    def list_operations(self, source: Optional[str] = None) -> Dict:
        """
        Return the full operation catalogue (or for a single source).
        Useful for the Intent Agent to know what it can call.
        """
        if source:
            return {source: OPERATION_CATALOGUE.get(source, [])}
        return OPERATION_CATALOGUE

    def status(self) -> Dict[str, Any]:
        """
        Return connectivity status for all three sources.
        Triggers lazy initialisation of all sentries.
        """
        for src in ["gitlab", "jira", "confluence"]:
            self._get_sentry(src)

        return {
            "available":   list(self._sentries.keys()),
            "unavailable": self._unavailable,
        }

    def __repr__(self):
        avail = list(self._sentries.keys())
        unavail = list(self._unavailable.keys())
        return f"SentryDispatcher(available={avail}, unavailable={unavail})"

#!/usr/bin/env python3
"""
Jira API Sentry
───────────────
Deterministic, real-time fetcher for Jira data.
No LLM involved — pure programmatic retrieval.

Supported operations
────────────────────
  get_issues          – JQL-powered issue search (with full pagination)
  get_issue           – single issue with all comments and transitions
  get_project_issues  – shorthand for project + optional status/assignee
  get_sprint_issues   – all issues in a board's active (or named) sprint
  get_boards          – list boards accessible to the token
  get_user_issues     – issues assigned to a specific user
  get_issue_changelog – full history / field changes for an issue
"""

import os
import requests
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

from sentries.base_sentry import BaseSentry, SentryResult

load_dotenv()

ATLASSIAN_EMAIL     = os.getenv("ATLASSIAN_EMAIL", "")
ATLASSIAN_API_TOKEN = os.getenv("ATLASSIAN_API_TOKEN", "")
ATLASSIAN_DOMAIN    = os.getenv("ATLASSIAN_DOMAIN", "")   # e.g. https://mycompany.atlassian.net


class JiraSentry(BaseSentry):
    """Real-time Jira data fetcher using Jira REST API v3."""

    SOURCE = "jira"

    # ── Setup ────────────────────────────────

    def _check_credentials(self):
        missing = [
            name for name, val in [
                ("ATLASSIAN_EMAIL",     ATLASSIAN_EMAIL),
                ("ATLASSIAN_API_TOKEN", ATLASSIAN_API_TOKEN),
                ("ATLASSIAN_DOMAIN",    ATLASSIAN_DOMAIN),
            ] if not val
        ]
        if missing:
            raise ValueError(
                f"Missing Jira credentials: {', '.join(missing)}. "
                "Add them to your .env file."
            )

    def _build_session(self):
        self.base_url = ATLASSIAN_DOMAIN.rstrip("/") + "/rest/api/3"
        self.agile_url = ATLASSIAN_DOMAIN.rstrip("/") + "/rest/agile/1.0"
        self.session = requests.Session()
        self.session.auth = (ATLASSIAN_EMAIL, ATLASSIAN_API_TOKEN)
        self.session.headers.update({
            "Accept":       "application/json",
            "Content-Type": "application/json",
        })

    # ── Internal helpers ─────────────────────

    def _get(self, path: str, params: Dict = None, base: str = None) -> requests.Response:
        self.rate_limiter.wait()
        root = (base or self.base_url).rstrip("/")
        url  = f"{root}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params or {}, timeout=30)
        resp.raise_for_status()
        return resp

    def _resolve_reporter_jql(self, name: str) -> str:
        """
        Resolve a display name to a Jira accountId for use in JQL reporter clause.
        Jira Cloud JQL requires accountId — using displayName silently returns 0 results.
        Falls back to displayName string if user search fails or returns no match
        (works on some Jira Server instances that still accept displayName in JQL).
        """
        try:
            resp = self.session.get(
                f"{self.base_url}/user/search",
                params={"query": name, "maxResults": 10},
                timeout=30,
            )
            if resp.status_code == 200:
                users = resp.json()
                name_lower = name.lower()
                # Prefer exact displayName match
                for user in users:
                    if user.get("displayName", "").lower() == name_lower:
                        return f'reporter = "{user["accountId"]}"'
                # Partial match fallback
                for user in users:
                    dn = user.get("displayName", "").lower()
                    if name_lower in dn or dn in name_lower:
                        return f'reporter = "{user["accountId"]}"'
        except Exception as exc:
            self.logger.debug("reporter user search failed for %r: %s", name, exc)
        # Fallback: displayName (Jira Server / older instances)
        return f'reporter = "{name}"'

    def _jql_paginate(
        self,
        jql:        str,
        fields:     List[str],
        max_items:  int = 200,
    ) -> Tuple[List[Dict], int]:
        """
        Paginate using GET /rest/api/3/search/jql with nextPageToken cursor.
        This is the endpoint used by ingestion_newest.py — proven to work on this account.
        Falls back to the legacy /issue/search if the new endpoint returns 404.
        """
        all_issues: List[Dict] = []
        next_page_token: Optional[str] = None
        page_size = min(100, max_items)
        total = 0

        while True:
            self.rate_limiter.wait()
            params: Dict[str, Any] = {
                "jql":        jql,
                "maxResults": page_size,
                "fields":     ",".join(fields),
            }
            if next_page_token:
                params["nextPageToken"] = next_page_token

            resp = self.session.get(
                f"{self.base_url}/search/jql",
                params=params,
                timeout=30,
            )

            # Fallback: if new endpoint not available, try legacy endpoint
            if resp.status_code == 404:
                resp = self.session.get(
                    f"{self.base_url}/issue/search",
                    params={**params, "startAt": len(all_issues)},
                    timeout=30,
                )

            # Silently skip archived/empty projects
            if resp.status_code == 410:
                break

            resp.raise_for_status()
            data = resp.json()

            issues = data.get("issues", [])
            if not issues:
                break

            all_issues.extend(issues)
            total = data.get("total", len(all_issues))

            next_page_token = data.get("nextPageToken")
            if not next_page_token or len(all_issues) >= max_items:
                break

        return all_issues[:max_items], total

    @staticmethod
    def _flatten_issue(issue: Dict) -> Dict:
        """Convert raw Jira issue JSON into a clean flat dict."""
        f = issue.get("fields", {})

        # Comments
        comments = []
        for c in (f.get("comment") or {}).get("comments", []):
            body_text = ""
            body = c.get("body", {})
            if isinstance(body, dict):
                # ADF (Atlassian Document Format) → extract plain text
                for block in body.get("content", []):
                    for inline in block.get("content", []):
                        if inline.get("type") == "text":
                            body_text += inline.get("text", "") + " "
            else:
                body_text = str(body)
            comments.append({
                "author":     c.get("author", {}).get("displayName"),
                "body":       body_text.strip()[:500],
                "created_at": c.get("created"),
            })

        # Description (ADF → plain text)
        desc_text = ""
        desc = f.get("description") or {}
        if isinstance(desc, dict):
            for block in desc.get("content", []):
                for inline in block.get("content", []):
                    if inline.get("type") == "text":
                        desc_text += inline.get("text", "") + " "
        else:
            desc_text = str(desc)

        return {
            "key":          issue.get("key"),
            "id":           issue.get("id"),
            "project_key":  (issue.get("key") or "").split("-")[0],
            "project_name": (f.get("project") or {}).get("name", ""),
            "summary":     f.get("summary"),
            "status":      (f.get("status") or {}).get("name"),
            "issue_type":  (f.get("issuetype") or {}).get("name"),
            "priority":    (f.get("priority") or {}).get("name"),
            "assignee":    (f.get("assignee") or {}).get("displayName"),
            "reporter":    (f.get("reporter") or {}).get("displayName"),
            "labels":      f.get("labels", []),
            "components":  [c["name"] for c in f.get("components", [])],
            "fix_versions":[v["name"] for v in f.get("fixVersions", [])],
            "sprint":      _extract_sprint(f),
            "created_at":  f.get("created"),
            "updated_at":  f.get("updated"),
            "resolved_at": f.get("resolutiondate"),
            "description": desc_text.strip()[:800],
            "comments":    comments,
            "url":         (
                ATLASSIAN_DOMAIN.rstrip("/") + "/browse/" + (issue.get("key") or "")
            ),
        }

    # ── Public operations ────────────────────

    def list_projects(self, limit: int = 500) -> SentryResult:
        """
        List all Jira projects accessible to the token.
        Uses /rest/api/3/project/search with pagination — returns every
        project regardless of recent activity, unlike the sampling fallback
        in ProjectRegistry.load_jira().

        Args:
            limit: Max projects to return (default 500, enterprise safe)
        """
        op = "list_projects"
        try:
            all_projects: list = []
            start_at = 0
            page_size = 50  # Jira Cloud max per page for project/search

            while True:
                self.rate_limiter.wait()
                resp = self.session.get(
                    f"{self.base_url}/project/search",
                    params={
                        "startAt":    start_at,
                        "maxResults": page_size,
                        "expand":     "description,lead",
                        "orderBy":    "NAME",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                values = data.get("values", [])
                if not values:
                    break

                for p in values:
                    all_projects.append({
                        "key":         p.get("key", ""),
                        "id":          p.get("id", ""),
                        "name":        p.get("name", ""),
                        "type":        p.get("projectTypeKey", ""),
                        "style":       p.get("style", ""),      # "next-gen" vs "classic"
                        "lead":        (p.get("lead") or {}).get("displayName", ""),
                        "description": (p.get("description") or "")[:300],
                        "url":         (p.get("self") or "").replace(
                            "/rest/api/3/project/", "/browse/"
                        ),
                    })

                start_at += len(values)
                total = data.get("total", 0)
                if start_at >= total or start_at >= limit:
                    break

            return self._ok(op, all_projects[:limit], total=len(all_projects))

        except Exception as exc:
            return self._err(op, str(exc))

    def get_issues(
        self,
        jql:   str,
        limit: int = 100,
    ) -> SentryResult:
        """
        Run an arbitrary JQL query and return structured issues.

        Args:
            jql:   Any valid JQL string
                   e.g. 'project = MYPROJ AND status = "In Progress"'
            limit: Max issues to return
        """
        op = "get_issues"
        fields = [
            "summary", "status", "issuetype", "priority",
            "assignee", "reporter", "labels", "components",
            "fixVersions", "created", "updated", "resolutiondate",
            "description", "comment", "customfield_10020",  # sprint
        ]
        try:
            raw_issues, total = self._jql_paginate(jql, fields, max_items=limit)
            items = [self._flatten_issue(i) for i in raw_issues]
            return self._ok(op, items, jql=jql, total_in_jira=total, returned=len(items))
        except Exception as exc:
            return self._err(op, str(exc), jql=jql)

    def get_project_issues(
        self,
        project_key:    str,
        status:         Optional[str] = None,
        assignee:       Optional[str] = None,
        reporter:       Optional[str] = None,
        issue_type:     Optional[str] = None,
        priority:       Optional[str] = None,
        search:         Optional[str] = None,
        created_after:  Optional[str] = None,   # "YYYY-MM-DD" — FIX 2
        created_before: Optional[str] = None,   # "YYYY-MM-DD" — FIX 2
        sort_by:        Optional[str] = None,   # "created" | "updated" — FIX 5
        limit:          int = 100,
    ) -> SentryResult:
        """
        Convenience method: fetch issues from a project with common filters.

        Args:
            project_key:    Jira project key (e.g. "DEMO", "PLAT")
            status:         Status name, e.g. "In Progress", "Done", "To Do"
            assignee:       Assignee's display name or accountId
            reporter:       Reporter's display name or accountId (maps to
                            'author' in natural-language queries)
            issue_type:     "Bug", "Story", "Task", "Epic", etc.
            priority:       "Highest", "High", "Medium", "Low"
            search:         Full-text search within project
            created_after:  ISO date "YYYY-MM-DD" — only tickets created >= this date
            created_before: ISO date "YYYY-MM-DD" — only tickets created <= this date
            sort_by:        "created" to sort newest-first by creation date;
                            default is "updated" (most recently updated first)
            limit:          Max issues
        """
        op = "get_project_issues"
        clauses = [f'project = "{project_key}"']

        if status:
            _STATUS_TO_CATEGORY = {
                "In Progress": 'statusCategory = "In Progress"',
                "In Review":   'statusCategory = "In Progress"',
                "Testing":     'statusCategory = "In Progress"',
                "Blocked":     'statusCategory = "In Progress"',
                "Done":        'statusCategory = "Done"',
                "Closed":      'statusCategory = "Done"',
                "Open":        'statusCategory = "To Do"',
                "To Do":       'statusCategory = "To Do"',
                "Backlog":     'statusCategory = "To Do"',
            }
            jql_status = _STATUS_TO_CATEGORY.get(status, f'status = "{status}"')
            clauses.append(jql_status)
        if assignee:       clauses.append(f'assignee = "{assignee}"')
        if reporter:       clauses.append(self._resolve_reporter_jql(reporter))
        if issue_type:     clauses.append(f'issuetype = "{issue_type}"')
        if priority:       clauses.append(f'priority = "{priority}"')
        if search:         clauses.append(f'text ~ "{search}"')
        # FIX 2 — date range filters (ISO format passed directly to JQL)
        if created_after:  clauses.append(f'created >= "{created_after}"')
        if created_before: clauses.append(f'created <= "{created_before}"')

        # FIX 5 — honour explicit sort_by; also default to "created DESC" when
        # a date filter is active so "tickets after DATE" returns newest-first.
        use_created_sort = sort_by == "created" or bool(created_after or created_before)
        order_field = "created DESC" if use_created_sort else "updated DESC"

        jql = " AND ".join(clauses) + f" ORDER BY {order_field}"

        result = self.get_issues(jql, limit=limit)
        result.operation = op
        result.meta.update(project_key=project_key)
        return result

    def get_issue(
        self,
        issue_key: str,
        include_changelog: bool = False,
    ) -> SentryResult:
        """
        Fetch a single issue with full detail (comments, description, history).

        Args:
            issue_key:         e.g. "DEMO-42"
            include_changelog: Also fetch the change history
        """
        op = "get_issue"
        try:
            resp = self._get(
                f"issue/{issue_key}",
                {"fields": "*all", "expand": "renderedFields,changelog" if include_changelog else "renderedFields"},
            )
            issue = resp.json()
            item  = self._flatten_issue(issue)

            if include_changelog:
                item["changelog"] = _extract_changelog(issue)

            return self._ok(op, [item], issue_key=issue_key)

        except Exception as exc:
            return self._err(op, str(exc), issue_key=issue_key)

    def get_issue_changelog(self, issue_key: str) -> SentryResult:
        """
        Fetch the full change history of an issue (all field transitions).
        """
        op = "get_issue_changelog"
        try:
            resp  = self._get(f"issue/{issue_key}/changelog")
            data  = resp.json()
            items = []
            for entry in data.get("values", []):
                for change in entry.get("items", []):
                    items.append({
                        "author":    entry.get("author", {}).get("displayName"),
                        "created":   entry.get("created"),
                        "field":     change.get("field"),
                        "from":      change.get("fromString"),
                        "to":        change.get("toString"),
                    })
            return self._ok(op, items, issue_key=issue_key, total_changes=len(items))
        except Exception as exc:
            return self._err(op, str(exc), issue_key=issue_key)

    def get_boards(self, project_key: Optional[str] = None, limit: int = 20) -> SentryResult:
        """
        List Jira Software boards.

        Args:
            project_key: Optionally filter boards by project
            limit:       Max boards
        """
        op = "get_boards"
        try:
            params: Dict[str, Any] = {"maxResults": limit}
            if project_key:
                params["projectKeyOrId"] = project_key

            resp  = self._get("board", params=params, base=self.agile_url)
            data  = resp.json()
            items = [
                {
                    "id":           b["id"],
                    "name":         b["name"],
                    "type":         b.get("type"),
                    "project_key":  (b.get("location") or {}).get("projectKey"),
                    "project_name": (b.get("location") or {}).get("projectName"),
                }
                for b in data.get("values", [])
            ]
            return self._ok(op, items, project_key=project_key)
        except Exception as exc:
            return self._err(op, str(exc), project_key=project_key)

    def get_sprint_issues(
        self,
        board_id:    int,
        sprint_name: Optional[str] = None,   # None → active sprint
        limit:       int = 100,
    ) -> SentryResult:
        """
        Get all issues in a sprint.

        Args:
            board_id:    Board numeric ID (get from get_boards)
            sprint_name: Name or partial name of the sprint; None = active sprint
            limit:       Max issues
        """
        op = "get_sprint_issues"
        try:
            # First verify board type — kanban boards don't have sprints
            board_resp = self._get(f"board/{board_id}", base=self.agile_url)
            board_data = board_resp.json()
            board_type = board_data.get("type", "")

            if board_type == "kanban":
                return self._err(
                    op,
                    f"Board {board_id} is a Kanban board — Kanban boards don't use sprints. "
                    f"Use get_project_issues instead, or pick a Scrum board from get_boards().",
                    board_id=board_id,
                    board_type="kanban",
                )

            # Fetch sprints — request active+future for name search, just active otherwise
            state_filter = "active,closed,future" if sprint_name else "active"
            sprints_resp = self._get(
                f"board/{board_id}/sprint",
                {"state": state_filter, "maxResults": 50},
                base=self.agile_url,
            )
            sprints = sprints_resp.json().get("values", [])

            if not sprints:
                return self._err(op, f"No sprints found on board {board_id}")

            if sprint_name:
                matches = [s for s in sprints if sprint_name.lower() in s["name"].lower()]
                if not matches:
                    names = [s["name"] for s in sprints]
                    return self._err(
                        op,
                        f"No sprint matching '{sprint_name}' on board {board_id}. "
                        f"Available: {names}",
                    )
                sprint = matches[0]
            else:
                active = [s for s in sprints if s.get("state") == "active"]
                if not active:
                    # Fall back to most recent future sprint
                    future = [s for s in sprints if s.get("state") == "future"]
                    if future:
                        sprint = future[0]
                    else:
                        return self._err(op, f"No active sprint on board {board_id}")
                else:
                    sprint = active[0]

            sprint_id   = sprint["id"]
            sprint_name = sprint["name"]

            fields = [
                "summary", "status", "issuetype", "priority",
                "assignee", "reporter", "labels", "created",
                "updated", "description", "comment", "customfield_10020",
            ]
            raw_issues, total = self._jql_paginate(
                f"sprint = {sprint_id}",
                fields,
                max_items=limit,
            )
            items = [self._flatten_issue(i) for i in raw_issues]

            return self._ok(
                op, items,
                board_id=board_id,
                sprint_id=sprint_id,
                sprint_name=sprint_name,
                sprint_state=sprint.get("state"),
                total_in_sprint=total,
            )

        except Exception as exc:
            return self._err(op, str(exc), board_id=board_id)

    def get_user_issues(
        self,
        assignee:    str,
        project_key: Optional[str] = None,
        status:      Optional[str] = None,
        limit:       int = 100,
    ) -> SentryResult:
        """
        Get all issues assigned to a specific user.

        Args:
            assignee:    Username or display name (quoted in JQL)
            project_key: Optionally restrict to one project
            status:      Optionally filter by status
            limit:       Max issues
        """
        op = "get_user_issues"
        clauses = [f'assignee = "{assignee}"']
        if project_key: clauses.append(f'project = "{project_key}"')
        if status:      clauses.append(f'status = "{status}"')
        jql = " AND ".join(clauses) + " ORDER BY updated DESC"

        result = self.get_issues(jql, limit=limit)
        result.operation = op
        result.meta.update(assignee=assignee)
        return result


# ── Module-level helpers (not part of the class) ──────────────────────────────

def _extract_sprint(fields: Dict) -> Optional[str]:
    """Extract sprint name from the sprint custom field (handles list or single)."""
    sprint_field = fields.get("customfield_10020")
    if not sprint_field:
        return None
    if isinstance(sprint_field, list) and sprint_field:
        sprint_field = sprint_field[-1]
    if isinstance(sprint_field, dict):
        return sprint_field.get("name")
    return str(sprint_field)


def _extract_changelog(issue: Dict) -> List[Dict]:
    """Extract simplified changelog from an expanded issue response."""
    entries = []
    for entry in (issue.get("changelog") or {}).get("histories", []):
        for item in entry.get("items", []):
            entries.append({
                "author":  entry.get("author", {}).get("displayName"),
                "created": entry.get("created"),
                "field":   item.get("field"),
                "from":    item.get("fromString"),
                "to":      item.get("toString"),
            })
    return entries
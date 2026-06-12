#!/usr/bin/env python3
"""
GitLab API Sentry
─────────────────
Deterministic, real-time fetcher for GitLab data.
No LLM involved — pure programmatic retrieval.

Supported operations
────────────────────
  list_projects       – discover all accessible projects for the token
  get_issues          – issues filtered by project/state/labels/assignee/search
  get_merge_requests  – MRs filtered by project/state/branch/author
  get_mr_diff         – full diff for a specific MR
  get_pipelines       – CI pipeline runs for a project
  get_commits         – commit history with optional path/branch filter
  get_file            – raw content of a single file at a given ref
  get_project_info    – project metadata (language, stars, topics…)
  get_milestones      – milestones for a project (or all projects)
  search_code         – global code search across accessible projects
  get_branches        – list branches for a project
"""

import os
import requests
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from sentries.base_sentry import BaseSentry, SentryResult

load_dotenv()

GITLAB_URL   = os.getenv("GITLAB_URL", "https://gitlab.com")
GITLAB_TOKEN = os.getenv("GITLAB_TOKEN", "")


class GitLabSentry(BaseSentry):
    """Real-time GitLab data fetcher."""

    SOURCE = "gitlab"

    # ── Setup ────────────────────────────────────────────────────────────────

    def _check_credentials(self):
        if not GITLAB_TOKEN:
            raise ValueError("GITLAB_TOKEN is not set. Add it to your .env file.")

    def _build_session(self):
        self.base_url = GITLAB_URL.rstrip("/") + "/api/v4"
        self.session  = requests.Session()
        self.session.headers.update({
            "PRIVATE-TOKEN": GITLAB_TOKEN,
            "Content-Type":  "application/json",
        })

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get(self, path: str, params: Dict = None) -> requests.Response:
        """Single GET with rate limiting and 429 retry."""
        import time as _time
        url = f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(3):
            self.rate_limiter.wait()
            response = self.session.get(url, params=params or {}, timeout=30)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 10))
                _time.sleep(retry_after)
                continue
            response.raise_for_status()
            return response
        response.raise_for_status()  # re-raise after exhausting retries
        return response

    def _resolve_project_id(self, project: str) -> str:
        """
        Accept either a numeric project ID or a namespace/path string.
        Returns URL-encoded path suitable for the GitLab REST API.
        """
        if str(project).isdigit():
            return project
        return requests.utils.quote(str(project), safe="")

    def _paginate_gitlab(
        self,
        path:      str,
        params:    Dict,
        max_items: int = 200,
    ) -> List[Dict]:
        """Paginate GitLab REST responses via X-Next-Page header."""
        import time as _time
        all_items: List[Dict] = []
        params = {**params, "per_page": 100, "page": 1}

        while True:
            url = f"{self.base_url}/{path.lstrip('/')}"
            for attempt in range(3):
                self.rate_limiter.wait()
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 10))
                    _time.sleep(retry_after)
                    continue
                break
            resp.raise_for_status()

            items = resp.json()
            if not items:
                break

            all_items.extend(items)

            next_page = resp.headers.get("X-Next-Page", "")
            if not next_page or len(all_items) >= max_items:
                break

            params["page"] = int(next_page)

        return all_items[:max_items]

    # ── Public operations ────────────────────────────────────────────────────

    def list_projects(
        self,
        owned:      bool = False,   # True → only projects owned by the token user
        membership: bool = True,    # True → projects the user is a member of
        search:     Optional[str] = None,
        limit:      int = 100,
    ) -> SentryResult:
        """
        Discover all projects accessible to the current token.

        Args:
            owned:      Return only projects owned by the authenticated user
            membership: Return projects the user is a member of
            search:     Filter by project name
            limit:      Max projects to return
        """
        op = "list_projects"
        try:
            params: Dict[str, Any] = {"order_by": "last_activity_at", "sort": "desc"}
            if owned:
                params["owned"] = "true"
            elif membership:
                params["membership"] = "true"
            if search:
                params["search"] = search

            raw = self._paginate_gitlab("projects", params, max_items=limit)

            items = [
                {
                    "id":            p["id"],
                    "name":          p["name"],
                    "path_with_ns":  p["path_with_namespace"],
                    "namespace":     p.get("namespace", {}).get("full_path"),
                    "visibility":    p.get("visibility"),
                    "description":   p.get("description"),
                    "default_branch":p.get("default_branch"),
                    "stars":         p.get("star_count"),
                    "forks":         p.get("forks_count"),
                    "open_issues":   p.get("open_issues_count"),
                    "last_activity": p.get("last_activity_at"),
                    "created_at":    p.get("created_at"),
                    "topics":        p.get("topics", []),
                    "url":           p.get("web_url"),
                }
                for p in raw
            ]

            return self._ok(op, items, total=len(items))

        except Exception as exc:
            return self._err(op, str(exc))

    def get_milestones(
        self,
        project:    Optional[str] = None,   # namespace/path or project ID
        state:      Optional[str] = None,   # active | closed  (None = all)
        search:     Optional[str] = None,
        limit:      int = 50,
    ) -> SentryResult:
        """
        Fetch milestones for a specific project, or across ALL known projects
        if no project is specified.

        Args:
            project:  Project ID or namespace/path.  If omitted, milestones are
                      fetched from every project the token can see.
            state:    'active', 'closed', or None for all
            search:   Filter by milestone title
            limit:    Max milestones per project (or total when scanning all)
        """
        op = "get_milestones"
        try:
            params: Dict[str, Any] = {}
            if state:  params["state"]  = state
            if search: params["search"] = search

            # ── Single project ───────────────────────────────────────────────
            if project:
                pid = self._resolve_project_id(project)
                raw = self._paginate_gitlab(
                    f"projects/{pid}/milestones", params, max_items=limit
                )
                items = self._format_milestones(raw, project)
                return self._ok(op, items, project=project, total=len(items))

            # ── All accessible projects ──────────────────────────────────────
            projects_result = self.list_projects(membership=True, limit=200)
            if not projects_result.success:
                return self._err(op, "Could not list projects: " + projects_result.error)

            all_items: List[Dict] = []
            for proj in projects_result.data:
                pid      = self._resolve_project_id(str(proj["id"]))
                proj_path = proj["path_with_ns"]
                try:
                    raw = self._paginate_gitlab(
                        f"projects/{pid}/milestones", params, max_items=limit
                    )
                    all_items.extend(self._format_milestones(raw, proj_path))
                except Exception:
                    pass   # skip projects where milestones are inaccessible

            return self._ok(op, all_items, total=len(all_items))

        except Exception as exc:
            return self._err(op, str(exc), project=project)

    @staticmethod
    def _format_milestones(raw: List[Dict], project: str) -> List[Dict]:
        return [
            {
                "id":          m["id"],
                "iid":         m["iid"],
                "title":       m["title"],
                "description": (m.get("description") or "")[:400],
                "state":       m["state"],
                "due_date":    m.get("due_date"),
                "start_date":  m.get("start_date"),
                "created_at":  m.get("created_at"),
                "updated_at":  m.get("updated_at"),
                "url":         m.get("web_url"),
                "project":     project,
            }
            for m in raw
        ]

    def get_issues(
        self,
        project:  str,
        state:    Optional[str] = "opened",   # opened | closed | all
        labels:   Optional[List[str]] = None,
        assignee: Optional[str] = None,
        search:   Optional[str] = None,
        limit:    int = 50,
    ) -> SentryResult:
        op = "get_issues"
        try:
            pid    = self._resolve_project_id(project)
            params: Dict[str, Any] = {"scope": "all"}

            if state and state != "all":
                params["state"] = state
            if labels:
                params["labels"] = ",".join(labels)
            if assignee:
                params["assignee_username"] = assignee
            if search:
                params["search"] = search

            raw = self._paginate_gitlab(
                f"projects/{pid}/issues", params, max_items=limit
            )

            items = [
                {
                    "id":          issue["iid"],
                    "title":       issue["title"],
                    "state":       issue["state"],
                    "author":      issue.get("author", {}).get("username"),
                    "assignees":   [a["username"] for a in issue.get("assignees", [])],
                    "labels":      issue.get("labels", []),
                    "description": (issue.get("description") or "")[:800],
                    "created_at":  issue["created_at"],
                    "updated_at":  issue["updated_at"],
                    "closed_at":   issue.get("closed_at"),
                    "url":         issue["web_url"],
                    "comments":    issue.get("user_notes_count", 0),
                    "milestone":   (issue.get("milestone") or {}).get("title"),
                    "project":     project,
                }
                for issue in raw
            ]

            return self._ok(op, items, project=project, state=state, total=len(items))

        except Exception as exc:
            return self._err(op, str(exc), project=project)

    def get_merge_requests(
        self,
        project:       str,
        state:         Optional[str] = "opened",
        target_branch: Optional[str] = None,
        author:        Optional[str] = None,
        search:        Optional[str] = None,
        limit:         int = 50,
    ) -> SentryResult:
        op = "get_merge_requests"
        try:
            pid    = self._resolve_project_id(project)
            params: Dict[str, Any] = {"scope": "all"}

            if state and state != "all":
                params["state"] = state
            if target_branch:
                params["target_branch"] = target_branch
            if author:
                params["author_username"] = author
            if search:
                params["search"] = search

            raw = self._paginate_gitlab(
                f"projects/{pid}/merge_requests", params, max_items=limit
            )

            items = [
                {
                    "id":             mr["iid"],
                    "title":          mr["title"],
                    "state":          mr["state"],
                    "author":         mr.get("author", {}).get("username"),
                    "source_branch":  mr["source_branch"],
                    "target_branch":  mr["target_branch"],
                    "description":    (mr.get("description") or "")[:600],
                    "labels":         mr.get("labels", []),
                    "created_at":     mr["created_at"],
                    "updated_at":     mr["updated_at"],
                    "merged_at":      mr.get("merged_at"),
                    "url":            mr["web_url"],
                    "pipeline_status":(mr.get("pipeline") or {}).get("status"),
                    "changes_count":  mr.get("changes_count"),
                    "project":        project,
                }
                for mr in raw
            ]

            return self._ok(op, items, project=project, state=state, total=len(items))

        except Exception as exc:
            return self._err(op, str(exc), project=project)

    def get_mr_diff(self, project: str, mr_iid: int) -> SentryResult:
        op = "get_mr_diff"
        try:
            pid = self._resolve_project_id(project)
            # /changes is the current endpoint (replaces deprecated /diffs in GL 15.7+).
            # It returns the same structure with an added "diff" key per file and
            # supports pagination unlike the old /diffs endpoint which truncated at 100.
            raw = self._paginate_gitlab(
                f"projects/{pid}/merge_requests/{mr_iid}/changes",
                params={},
                max_items=200,
            )
            # /changes wraps files under "changes" key in the first item
            # when called as a single resource — but paginated returns list of files.
            # Handle both: single dict with "changes" or list of file dicts.
            if raw and isinstance(raw[0], dict) and "changes" in raw[0]:
                file_list = raw[0]["changes"]
            else:
                file_list = raw

            items = [
                {
                    "old_path":     d.get("old_path"),
                    "new_path":     d.get("new_path"),
                    "new_file":     d.get("new_file"),
                    "deleted_file": d.get("deleted_file"),
                    "renamed_file": d.get("renamed_file"),
                    "diff":         d.get("diff", ""),   # cap applied in weaver_node
                }
                for d in file_list
            ]

            return self._ok(op, items, project=project, mr_iid=mr_iid, files_changed=len(items))

        except Exception as exc:
            return self._err(op, str(exc), project=project, mr_iid=mr_iid)

    def get_pipelines(
        self,
        project: str,
        ref:     Optional[str] = None,
        status:  Optional[str] = None,
        limit:   int = 30,
    ) -> SentryResult:
        op = "get_pipelines"
        try:
            pid    = self._resolve_project_id(project)
            params: Dict[str, Any] = {}
            if ref:    params["ref"]    = ref
            if status: params["status"] = status

            raw = self._paginate_gitlab(
                f"projects/{pid}/pipelines", params, max_items=limit
            )

            items = [
                {
                    "id":         p["id"],
                    "status":     p["status"],
                    "ref":        p.get("ref"),
                    "sha":        p.get("sha"),
                    "created_at": p.get("created_at"),
                    "updated_at": p.get("updated_at"),
                    "url":        p.get("web_url"),
                    "duration":   p.get("duration"),
                    "project":    project,
                }
                for p in raw
            ]

            return self._ok(op, items, project=project, ref=ref, status=status)

        except Exception as exc:
            return self._err(op, str(exc), project=project)

    def get_commits(
        self,
        project:   str,
        branch:    Optional[str] = None,
        file_path: Optional[str] = None,
        since:     Optional[str] = None,
        limit:     int = 50,
    ) -> SentryResult:
        op = "get_commits"
        try:
            pid    = self._resolve_project_id(project)
            params: Dict[str, Any] = {}
            if branch:    params["ref_name"] = branch
            if file_path: params["path"]     = file_path
            if since:     params["since"]    = since

            raw = self._paginate_gitlab(
                f"projects/{pid}/repository/commits", params, max_items=limit
            )

            items = [
                {
                    "sha":           c["id"],
                    "short_sha":     c["short_id"],
                    "title":         c["title"],
                    "author_name":   c.get("author_name"),
                    "author_email":  c.get("author_email"),
                    "authored_date": c.get("authored_date"),
                    "committed_date":c.get("committed_date"),
                    "message":       (c.get("message") or "")[:400],
                    "url":           c.get("web_url"),
                    "project":       project,
                }
                for c in raw
            ]

            return self._ok(op, items, project=project, branch=branch, total=len(items))

        except Exception as exc:
            return self._err(op, str(exc), project=project)

    def get_file(
        self,
        project:   str,
        file_path: str,
        ref:       str = "",   # empty → resolved from project's default branch
    ) -> SentryResult:
        op = "get_file"
        try:
            pid = self._resolve_project_id(project)

            # Resolve default branch dynamically — the same approach used by
            # get_repository_tree.  Hardcoding "main" causes silent 404s on
            # repos whose default branch is "master", "develop", etc.
            if not ref:
                info_resp = self._get(f"projects/{pid}")
                ref = info_resp.json().get("default_branch") or "main"

            encoded_path = requests.utils.quote(file_path, safe="")
            resp         = self._get(
                f"projects/{pid}/repository/files/{encoded_path}",
                {"ref": ref},
            )
            raw = resp.json()

            import base64
            content = ""
            if raw.get("encoding") == "base64":
                content = base64.b64decode(raw["content"]).decode("utf-8", errors="replace")
            else:
                content = raw.get("content", "")

            return self._ok(
                op,
                [{"file_path": file_path, "ref": ref, "content": content, "size": raw.get("size")}],
                project=project,
            )

        except Exception as exc:
            return self._err(op, str(exc), project=project, file_path=file_path)

    def get_project_info(self, project: str) -> SentryResult:
        op = "get_project_info"
        try:
            pid  = self._resolve_project_id(project)
            resp = self._get(f"projects/{pid}")
            p    = resp.json()

            return self._ok(
                op,
                [{
                    "id":             p["id"],
                    "name":           p["name"],
                    "namespace":      p.get("namespace", {}).get("full_path"),
                    "description":    p.get("description"),
                    "default_branch": p.get("default_branch"),
                    "stars":          p.get("star_count"),
                    "forks":          p.get("forks_count"),
                    "open_issues":    p.get("open_issues_count"),
                    "last_activity":  p.get("last_activity_at"),
                    "created_at":     p.get("created_at"),
                    "topics":         p.get("topics", []),
                    "visibility":     p.get("visibility"),
                    "url":            p.get("web_url"),
                    "ci_config_path": p.get("ci_config_path"),
                }],
                project=project,
            )

        except Exception as exc:
            return self._err(op, str(exc), project=project)

    def get_repository_tree(
        self,
        project:   str,
        path:      str = "",
        ref:       Optional[str] = None,
        recursive: bool = False,
        limit:     int = 100,
    ) -> SentryResult:
        """
        List files and directories in a repository.

        Args:
            project:   Project ID or namespace/path
            path:      Sub-directory path (empty = root)
            ref:       Branch/tag/commit. Defaults to the project default branch.
            recursive: Recursively list all files
            limit:     Max items to return
        """
        op = "get_repository_tree"
        try:
            pid = self._resolve_project_id(project)

            # Resolve default branch if ref not provided
            if not ref:
                info_resp = self._get(f"projects/{pid}")
                ref = info_resp.json().get("default_branch", "main")

            params: Dict[str, Any] = {"ref": ref}
            if path:
                params["path"] = path
            if recursive:
                params["recursive"] = "true"

            raw = self._paginate_gitlab(
                f"projects/{pid}/repository/tree", params, max_items=limit
            )

            items = [
                {
                    "project": project,
                    "name": f["name"],
                    "type": f["type"],   # "blob" = file, "tree" = directory
                    "path": f["path"],
                    "id":   f["id"],
                }
                for f in raw
            ]

            return self._ok(op, items, project=project, path=path, ref=ref, total=len(items))

        except Exception as exc:
            return self._err(op, str(exc), project=project)

    def search_code(
        self,
        query:   str,
        project: Optional[str] = None,
        limit:   int = 20,
    ) -> SentryResult:
        op = "search_code"
        try:
            params: Dict[str, Any] = {"scope": "blobs", "search": query}

            if project:
                pid  = self._resolve_project_id(project)
                path = f"projects/{pid}/search"
            else:
                path = "search"

            raw = self._paginate_gitlab(path, params, max_items=limit)

            items = [
                {
                    "filename":   r.get("filename"),
                    "ref":        r.get("ref"),
                    "project_id": r.get("project_id"),
                    "snippet":    (r.get("data") or "")[:600],
                    "start_line": r.get("startline"),
                }
                for r in raw
            ]

            return self._ok(op, items, query=query, project=project, total=len(items))

        except Exception as exc:
            return self._err(op, str(exc), query=query)

    def get_branches(
        self,
        project: str,
        search:  Optional[str] = None,
        limit:   int = 50,
    ) -> SentryResult:
        """
        List branches for a project.

        Args:
            project: Project ID or namespace/path
            search:  Filter branches by name (substring)
            limit:   Max branches to return
        """
        op = "get_branches"
        try:
            pid    = self._resolve_project_id(project)
            params: Dict[str, Any] = {}
            if search:
                params["search"] = search

            raw = self._paginate_gitlab(
                f"projects/{pid}/repository/branches", params, max_items=limit
            )

            items = [
                {
                    "name":             b["name"],
                    "default":          b.get("default", False),
                    "protected":        b.get("protected", False),
                    "merged":           b.get("merged", False),
                    "last_commit_sha":  (b.get("commit") or {}).get("id"),
                    "last_commit_date": (b.get("commit") or {}).get("committed_date"),
                    "last_commit_msg":  ((b.get("commit") or {}).get("title") or "")[:120],
                    "project":          project,
                }
                for b in raw
            ]

            return self._ok(op, items, project=project, total=len(items))

        except Exception as exc:
            return self._err(op, str(exc), project=project)

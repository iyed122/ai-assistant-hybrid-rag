#!/usr/bin/env python3
"""
Knowledge Base Ingestion Script
Automatically discovers and ingests data from GitLab, Jira, and Confluence.

GitLab coverage:
  - Issues (with comments)
  - Merge Requests (with comments)
  - Milestones
  - Code files (README, source, CI/CD, Dockerfiles, configs)

Discovery uses /projects?membership=true which returns ALL projects the token
can see: personal namespace (primary-namespace/*), group projects, and shared projects.
"""

import os
import time
import base64
import threading
import requests
import gitlab
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

MONGO_URI        = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
MONGO_DB         = os.getenv('MONGO_DB', 'knowledge_base')
RATE_LIMIT_DELAY     = float(os.getenv('RATE_LIMIT_DELAY', '0.1'))
CONFLUENCE_WORKERS   = int(os.getenv('CONFLUENCE_WORKERS', '4'))

ATLASSIAN_EMAIL     = os.getenv('ATLASSIAN_EMAIL')
ATLASSIAN_API_TOKEN = os.getenv('ATLASSIAN_API_TOKEN')
ATLASSIAN_DOMAIN    = os.getenv('ATLASSIAN_DOMAIN')

GITLAB_TOKEN = os.getenv('GITLAB_TOKEN')
GITLAB_URL   = os.getenv('GITLAB_URL', 'https://gitlab.com')

# Code file ingestion settings
MAX_FILE_SIZE_BYTES = int(os.getenv('MAX_CODE_FILE_BYTES', str(150 * 1024)))  # 150 KB

# File extensions to ingest for code review / RAG
CODE_EXTENSIONS = {
    # Python
    '.py', '.pyi',
    # JavaScript / TypeScript
    '.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs',
    # Web
    '.html', '.css', '.scss', '.sass',
    # JVM
    '.java', '.kt', '.scala', '.groovy',
    # Systems
    '.go', '.rs', '.cpp', '.c', '.h', '.hpp', '.cs',
    # Ruby / PHP
    '.rb', '.php',
    # Shell
    '.sh', '.bash', '.zsh',
    # Config / infra
    '.yml', '.yaml', '.toml', '.ini', '.cfg', '.conf',
    '.json',   # but we skip package-lock.json / yarn.lock below
    '.env.example', '.env.sample',
    # CI/CD & containers
    'Dockerfile', '.dockerfile',
    'docker-compose.yml', 'docker-compose.yaml',
    '.gitlab-ci.yml',
    'Gemfile',
    # Docs
    '.md', '.rst', '.txt',
    # SQL
    '.sql',
    
}

# Paths / filenames to skip entirely
SKIP_PATHS = {
    'node_modules', '.git', '__pycache__', 'dist', 'build',
    'venv', '.venv', 'env', '.env',
    'vendor', 'coverage', '.nyc_output',
    'package-lock.json', 'yarn.lock', 'poetry.lock', 'Pipfile.lock',
    'pnpm-lock.yaml', 'composer.lock', 'Gemfile.lock',
}


def mask_token(token):
    if not token:
        return "Not set"
    if len(token) < 8:
        return "***"
    return f"{token[:4]}...{token[-4:]}"


print("🔧 Configuration loaded:")
print(f"  GITLAB_TOKEN: {mask_token(GITLAB_TOKEN)}")
print(f"  GITLAB_URL:   {GITLAB_URL}")
print(f"  ATLASSIAN_EMAIL:  {ATLASSIAN_EMAIL}")
print(f"  ATLASSIAN_DOMAIN: {ATLASSIAN_DOMAIN}")
print()

client = MongoClient(MONGO_URI)
db     = client[MONGO_DB]


# ── Utilities ──────────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, delay: float = 0.1):
        self.delay     = delay
        self.last_call = 0

    def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_call = time.time()


rate_limiter = RateLimiter(RATE_LIMIT_DELAY)


def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def should_skip_path(path: str) -> bool:
    """Return True if any path component is in the skip set."""
    parts = path.replace('\\', '/').split('/')
    for part in parts:
        if part in SKIP_PATHS:
            return True
    return False


def is_code_file(path: str) -> bool:
    """Return True if the file should be ingested."""
    if should_skip_path(path):
        return False
    filename = path.split('/')[-1]
    # Exact filename matches (Dockerfile, Makefile, etc.)
    if filename in CODE_EXTENSIONS:
        return True
    # Extension match
    _, ext = os.path.splitext(filename)
    return ext.lower() in CODE_EXTENSIONS


# ── GitLab Ingester ────────────────────────────────────────────────────────────

class GitLabIngester:
    """Handles GitLab data ingestion for ALL accessible projects."""

    def __init__(self):
        self.gl       = gitlab.Gitlab(GITLAB_URL, private_token=GITLAB_TOKEN)
        self.gl.auth()
        self.base_url = GITLAB_URL.rstrip('/') + '/api/v4'
        self.session  = requests.Session()
        self.session.headers.update({'PRIVATE-TOKEN': GITLAB_TOKEN})
        log("GitLab connection established")

    # ── Project discovery ──────────────────────────────────────────────────────

    def discover_all_projects(self) -> List[Any]:
        """
        Return every project the token can see:
          - personal namespace (primary-namespace/*)
          - group projects
          - projects shared with the user

        Uses GET /projects?membership=true which is a single paginated endpoint
        — much faster than iterating groups and avoids missing personal projects.
        """
        log("Discovering all accessible projects (personal + groups)...")

        all_projects = []
        seen_ids: set = set()
        page = 1

        while True:
            rate_limiter.wait()
            resp = self.session.get(
                f"{self.base_url}/projects",
                params={
                    "membership":    "true",
                    "order_by":      "last_activity_at",
                    "sort":          "desc",
                    "per_page":      100,
                    "page":          page,
                    "with_issues_enabled": "false",   # metadata only, no filter
                },
                timeout=30,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break

            for proj in batch:
                if proj['id'] not in seen_ids:
                    seen_ids.add(proj['id'])
                    all_projects.append(proj)

            next_page = resp.headers.get('X-Next-Page', '')
            if not next_page:
                break
            page = int(next_page)

        log(f"Discovered {len(all_projects)} projects total")
        for p in all_projects:
            log(f"  • {p['path_with_namespace']}  (id={p['id']}, "
                f"visibility={p.get('visibility', '?')})")
        return all_projects

    # ── Issues ─────────────────────────────────────────────────────────────────

    def ingest_project_issues(self, project_meta: Dict):
        """Ingest all issues (with comments) from a project."""
        project_id   = project_meta['id']
        project_name = project_meta['name']
        try:
            full_project = self.gl.projects.get(project_id)
            issues       = full_project.issues.list(all=True)

            for issue in issues:
                rate_limiter.wait()
                full_issue = full_project.issues.get(issue.iid)
                notes      = full_issue.notes.list(all=True)

                doc = {
                    "source":       "gitlab",
                    "type":         "issue",
                    "group_name":   project_meta.get('namespace', {}).get('full_path', ''),
                    "project_id":   project_id,
                    "project_name": project_name,
                    "issue_id":     issue.iid,
                    "title":        issue.title,
                    "description":  issue.description or "",
                    "state":        issue.state,
                    "author":       issue.author.get('username', 'unknown'),
                    "created_at":   issue.created_at,
                    "updated_at":   issue.updated_at,
                    "labels":       issue.labels,
                    "web_url":      issue.web_url,
                    "milestone":    (issue.milestone or {}).get('title'),
                    "comments": [
                        {
                            "author":     note.author.get('username', 'unknown'),
                            "body":       note.body,
                            "created_at": note.created_at,
                        }
                        for note in notes
                        if not note.system   # skip system events
                    ],
                    "ingested_at": datetime.utcnow(),
                }

                db.gitlab_issues.update_one(
                    {"project_id": project_id, "issue_id": issue.iid},
                    {"$set": doc},
                    upsert=True,
                )

            log(f"  ✓ Issues:  {len(issues)} from {project_name}")
        except Exception as e:
            log(f"  ✗ Issues error in {project_name}: {e}", "ERROR")

    # ── Merge Requests ─────────────────────────────────────────────────────────

    def ingest_project_merge_requests(self, project_meta: Dict):
        """Ingest all MRs (with comments) from a project."""
        project_id   = project_meta['id']
        project_name = project_meta['name']
        try:
            full_project = self.gl.projects.get(project_id)
            mrs          = full_project.mergerequests.list(all=True)

            for mr in mrs:
                rate_limiter.wait()
                full_mr = full_project.mergerequests.get(mr.iid)
                notes   = full_mr.notes.list(all=True)

                doc = {
                    "source":        "gitlab",
                    "type":          "merge_request",
                    "group_name":    project_meta.get('namespace', {}).get('full_path', ''),
                    "project_id":    project_id,
                    "project_name":  project_name,
                    "mr_id":         mr.iid,
                    "title":         mr.title,
                    "description":   mr.description or "",
                    "state":         mr.state,
                    "author":        mr.author.get('username', 'unknown'),
                    "source_branch": mr.source_branch,
                    "target_branch": mr.target_branch,
                    "created_at":    mr.created_at,
                    "updated_at":    mr.updated_at,
                    "merged_at":     getattr(mr, 'merged_at', None),
                    "labels":        mr.labels,
                    "web_url":       mr.web_url,
                    "comments": [
                        {
                            "author":     note.author.get('username', 'unknown'),
                            "body":       note.body,
                            "created_at": note.created_at,
                        }
                        for note in notes
                        if not note.system
                    ],
                    "ingested_at": datetime.utcnow(),
                }

                db.gitlab_merge_requests.update_one(
                    {"project_id": project_id, "mr_id": mr.iid},
                    {"$set": doc},
                    upsert=True,
                )

            log(f"  ✓ MRs:     {len(mrs)} from {project_name}")
        except Exception as e:
            log(f"  ✗ MRs error in {project_name}: {e}", "ERROR")

    # ── Milestones ─────────────────────────────────────────────────────────────

    def ingest_project_milestones(self, project_meta: Dict):
        """Ingest all milestones from a project."""
        project_id   = project_meta['id']
        project_name = project_meta['name']
        try:
            resp = self.session.get(
                f"{self.base_url}/projects/{project_id}/milestones",
                params={"per_page": 100},
                timeout=30,
            )
            resp.raise_for_status()
            milestones = resp.json()

            for ms in milestones:
                doc = {
                    "source":            "gitlab",
                    "type":              "milestone",
                    "group_name":        project_meta.get('namespace', {}).get('full_path', ''),
                    "project_id":        project_id,
                    "project_name":      project_name,
                    "milestone_id":      ms['id'],
                    "milestone_iid":     ms['iid'],
                    "title":             ms['title'],
                    "description":       ms.get('description') or "",
                    "state":             ms['state'],
                    "due_date":          ms.get('due_date'),
                    "start_date":        ms.get('start_date'),
                    "created_at":        ms.get('created_at', ''),
                    "updated_at":        ms.get('updated_at', ''),
                    "web_url":           ms.get('web_url', ''),
                    "ingested_at":       datetime.utcnow(),
                }

                db.gitlab_milestones.update_one(
                    {"project_id": project_id, "milestone_id": ms['id']},
                    {"$set": doc},
                    upsert=True,
                )

            log(f"  ✓ Milestones: {len(milestones)} from {project_name}")
        except Exception as e:
            log(f"  ✗ Milestones error in {project_name}: {e}", "ERROR")

    # ── Code Files ─────────────────────────────────────────────────────────────

    def _get_repo_tree(self, project_id: int, ref: str) -> List[Dict]:
        """Return flat list of all files in the repository (recursive)."""
        all_files = []
        page = 1
        while True:
            rate_limiter.wait()
            resp = self.session.get(
                f"{self.base_url}/projects/{project_id}/repository/tree",
                params={"recursive": "true", "per_page": 100, "page": page, "ref": ref},
                timeout=30,
            )
            if resp.status_code == 404:
                return []   # empty repo or no default branch
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            # Only keep blobs (files), not trees (dirs)
            all_files.extend(item for item in batch if item['type'] == 'blob')
            next_page = resp.headers.get('X-Next-Page', '')
            if not next_page:
                break
            page = int(next_page)
        return all_files

    def _fetch_file_content(self, project_id: int, file_path: str, ref: str) -> Optional[str]:
        """Fetch and decode a single file. Returns None if too large or binary."""
        rate_limiter.wait()
        encoded_path = requests.utils.quote(file_path, safe='')
        resp = self.session.get(
            f"{self.base_url}/projects/{project_id}/repository/files/{encoded_path}",
            params={"ref": ref},
            timeout=30,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        size = data.get('size', 0)
        if size > MAX_FILE_SIZE_BYTES:
            return None  # skip large files

        if data.get('encoding') == 'base64':
            try:
                content = base64.b64decode(data['content']).decode('utf-8', errors='replace')
            except Exception:
                return None
        else:
            content = data.get('content', '')

        # Basic binary check — skip if too many non-printable chars
        non_printable = sum(1 for c in content[:1000] if not c.isprintable() and c not in '\n\r\t')
        if non_printable > 50:
            return None

        return content

    def ingest_project_code_files(self, project_meta: Dict):
        """Ingest all relevant code files from the default branch."""
        project_id   = project_meta['id']
        project_name = project_meta['name']
        ref          = project_meta.get('default_branch') or 'main'

        try:
            tree = self._get_repo_tree(project_id, ref)
            if not tree:
                log(f"  ℹ  Code:    empty repo or no branch '{ref}' in {project_name}")
                return

            eligible = [f for f in tree if is_code_file(f['path'])]
            log(f"  → Code:    {len(eligible)}/{len(tree)} files eligible in {project_name}")

            ingested = 0
            for file_item in eligible:
                file_path = file_item['path']
                content   = self._fetch_file_content(project_id, file_path, ref)
                if content is None:
                    continue

                doc = {
                    "source":       "gitlab",
                    "type":         "code_file",
                    "group_name":   project_meta.get('namespace', {}).get('full_path', ''),
                    "project_id":   project_id,
                    "project_name": project_name,
                    "file_path":    file_path,
                    "file_name":    file_path.split('/')[-1],
                    "extension":    os.path.splitext(file_path)[1].lower(),
                    "ref":          ref,
                    "content":      content,
                    "size_bytes":   len(content.encode('utf-8')),
                    "web_url":      f"{project_meta.get('web_url', '')}/-/blob/{ref}/{file_path}",
                    "ingested_at":  datetime.utcnow(),
                }

                db.gitlab_code_files.update_one(
                    {"project_id": project_id, "file_path": file_path},
                    {"$set": doc},
                    upsert=True,
                )
                ingested += 1

            log(f"  ✓ Code:    {ingested} files ingested from {project_name}")
        except Exception as e:
            log(f"  ✗ Code error in {project_name}: {e}", "ERROR")

    # ── Main flow ──────────────────────────────────────────────────────────────

    def ingest_all(self):
        log("=" * 60)
        log("Starting GitLab ingestion")
        log("=" * 60)

        projects = self.discover_all_projects()

        for i, project_meta in enumerate(projects, 1):
            name = project_meta['path_with_namespace']
            log(f"\n[{i}/{len(projects)}] {name}")
            self.ingest_project_issues(project_meta)
            self.ingest_project_merge_requests(project_meta)
            self.ingest_project_milestones(project_meta)
            self.ingest_project_code_files(project_meta)

        # Create indexes
        db.gitlab_milestones.create_index(
            [("project_id", 1), ("milestone_id", 1)], unique=True
        )
        db.gitlab_code_files.create_index(
            [("project_id", 1), ("file_path", 1)], unique=True
        )

        log("\nGitLab ingestion completed")


# ── Jira Ingester ──────────────────────────────────────────────────────────────

class JiraIngester:
    """Handles Jira data ingestion with cursor-based pagination."""

    def __init__(self):
        self.auth   = (ATLASSIAN_EMAIL, ATLASSIAN_API_TOKEN)
        self.domain = ATLASSIAN_DOMAIN
        log("Jira connection configured")

    def discover_projects(self) -> List[Dict[str, Any]]:
        log("Discovering Jira projects...")
        try:
            url      = f"{self.domain}/rest/api/3/project"
            response = requests.get(url, auth=self.auth)
            response.raise_for_status()
            projects = response.json()
            log(f"Discovered {len(projects)} Jira projects")
            return projects
        except Exception as e:
            log(f"Error discovering Jira projects: {e}", "ERROR")
            return []

    def ingest_project_issues(self, project_key: str, project_name: str):
        try:
            total_ingested  = 0
            next_page_token = None

            while True:
                rate_limiter.wait()
                url    = f"{self.domain}/rest/api/3/search/jql"
                params = {
                    "jql":        f"project='{project_key}'",
                    "maxResults": 50,
                    # Explicit list — ~40% smaller payload than *all
                    # customfield_10014 = Epic Link (classic Jira projects)
                    # customfield_10016 = Story Points
                    "fields": (
                        "summary,description,status,issuetype,priority,"
                        "assignee,reporter,created,updated,labels,comment,"
                        "issuelinks,parent,fixVersions,components,resolution,"
                        "subtasks,duedate,"
                        "customfield_10014,"
                        "customfield_10016"
                    ),
                }
                if next_page_token:
                    params["nextPageToken"] = next_page_token

                response = requests.get(url, auth=self.auth, params=params)

                if response.status_code == 410:
                    log(f"  ⚠ Project {project_name} empty/archived (HTTP 410). Skipping.")
                    return

                response.raise_for_status()
                data   = response.json()
                issues = data.get('issues', [])
                if not issues:
                    break

                for issue in issues:
                    fields    = issue.get('fields') or {}
                    issue_key = issue.get('key', '')
                    web_url   = f"{self.domain.rstrip('/')}/browse/{issue_key}" if issue_key else ''

                    # Comments
                    comments = []
                    for c in (fields.get('comment') or {}).get('comments', []):
                        comments.append({
                            "author":  (c.get('author') or {}).get('displayName', 'unknown'),
                            "body":    c.get('body', ''),
                            "created": c.get('created', ''),
                        })

                    # Parent (sub-task → task → epic hierarchy)
                    parent     = fields.get('parent') or {}
                    parent_key = parent.get('key')

                    # Issue links ("blocks", "is blocked by", "relates to", …)
                    # target_summary is stored so Neo4j has the linked ticket's
                    # title without needing a separate fetch during graph import.
                    issue_links = []
                    for lnk in fields.get('issuelinks', []):
                        lt = lnk.get('type') or {}
                        if 'inwardIssue' in lnk:
                            inward = lnk['inwardIssue']
                            issue_links.append({
                                "type":           lt.get('inward', 'relates to'),
                                "target":         inward.get('key', ''),
                                "target_summary": (inward.get('fields') or {}).get('summary', ''),
                                "target_status":  ((inward.get('fields') or {}).get('status') or {}).get('name', ''),
                                "direction":      "inward",
                            })
                        elif 'outwardIssue' in lnk:
                            outward = lnk['outwardIssue']
                            issue_links.append({
                                "type":           lt.get('outward', 'relates to'),
                                "target":         outward.get('key', ''),
                                "target_summary": (outward.get('fields') or {}).get('summary', ''),
                                "target_status":  ((outward.get('fields') or {}).get('status') or {}).get('name', ''),
                                "direction":      "outward",
                            })

                    # Fix versions — which release this belongs to
                    fix_versions = [v.get('name') for v in fields.get('fixVersions', []) if v.get('name')]

                    # Components — team/area ownership (e.g. "Frontend", "API")
                    components = [c.get('name') for c in fields.get('components', []) if c.get('name')]

                    # Resolution — WHY it was closed ("Fixed", "Won't Do", "Duplicate"…)
                    resolution = (fields.get('resolution') or {}).get('name')

                    # Epic Link — classic Jira projects store epic membership in
                    # customfield_10014, NOT in parent. Next-gen / team-managed
                    # projects use parent instead, so we check both.
                    epic_link = (
                        fields.get('customfield_10014')
                        or (parent.get('fields', {}).get('issuetype', {}).get('name', '') == 'Epic'
                            and parent_key)
                        or None
                    )

                    # Story points (customfield_10016)
                    story_points = fields.get('customfield_10016')

                    # Due date
                    due_date = fields.get('duedate')

                    # Subtasks — child issues one level down
                    # Stored as [{key, summary, status}] for context in Neo4j
                    subtasks = []
                    for st in fields.get('subtasks', []):
                        st_fields = st.get('fields') or {}
                        subtasks.append({
                            "key":     st.get('key', ''),
                            "summary": st_fields.get('summary', ''),
                            "status":  (st_fields.get('status') or {}).get('name', ''),
                        })

                    doc = {
                        "source":       "jira",
                        "type":         "issue",
                        "project_key":  project_key,
                        "project_name": project_name,
                        "issue_key":    issue_key,
                        "issue_id":     issue.get('id'),
                        "summary":      fields.get('summary', ''),
                        "description":  fields.get('description', ''),
                        "status":       (fields.get('status') or {}).get('name', ''),
                        "issue_type":   (fields.get('issuetype') or {}).get('name', ''),
                        "priority":     (fields.get('priority') or {}).get('name', ''),
                        "assignee":     (fields.get('assignee') or {}).get('displayName'),
                        "reporter":     (fields.get('reporter') or {}).get('displayName', 'unknown'),
                        "created":      fields.get('created', ''),
                        "updated":      fields.get('updated', ''),
                        "labels":       fields.get('labels', []),
                        "web_url":      web_url,
                        "comments":     comments,
                        # ── Relationship / context fields ──────────────────────
                        "parent_key":    parent_key,
                        "issue_links":   issue_links,
                        "fix_versions":  fix_versions,
                        "components":    components,
                        "resolution":    resolution,
                        "epic_link":     epic_link,
                        "story_points":  story_points,
                        "due_date":      due_date,
                        "subtasks":      subtasks,
                        "ingested_at":   datetime.utcnow(),
                    }

                    db.jira_issues.update_one(
                        {"issue_key": issue_key},
                        {"$set": doc},
                        upsert=True,
                    )
                    total_ingested += 1

                next_page_token = data.get('nextPageToken')
                if not next_page_token:
                    break

            log(f"  ✓ Ingested {total_ingested} issues from {project_name}")
        except Exception as e:
            log(f"  ✗ Error ingesting issues from {project_name}: {e}", "ERROR")

    def ingest_all(self):
        log("=" * 60)
        log("Starting Jira ingestion")
        log("=" * 60)

        projects = self.discover_projects()

        for i, project in enumerate(projects, 1):
            project_key  = project.get('key')
            project_name = project.get('name', project_key)
            is_archived  = project.get('archived', False)

            if is_archived:
                log(f"[{i}/{len(projects)}] Skipping archived: {project_name}")
                continue

            log(f"[{i}/{len(projects)}] Processing: {project_name} ({project_key})")
            self.ingest_project_issues(project_key, project_name)

        log("Jira ingestion completed")


# ── Confluence Ingester ────────────────────────────────────────────────────────

class ConfluenceIngester:
    """
    Confluence ingester — v2.
    Changes vs original:
      - body-format=storage in the list call → eliminates N+1 per-page fetch
      - ancestors + labels fetched per page (2 extra slim calls, non-fatal)
      - 4 parallel workers (CONFLUENCE_WORKERS) across spaces
      - Thread-local rate limiter so workers don't block each other
    """

    def __init__(self):
        self.auth   = (ATLASSIAN_EMAIL, ATLASSIAN_API_TOKEN)
        self.domain = ATLASSIAN_DOMAIN.rstrip('/')
        self._mongo_lock = threading.Lock()
        self._log_lock   = threading.Lock()
        self._counter    = {"done": 0, "pages": 0}
        self._c_lock     = threading.Lock()
        # Per-thread last-call timestamp for rate limiting
        self._tlocal = threading.local()
        log("Confluence connection configured")

    def _rate_wait(self):
        """Per-thread rate limit — threads don't block each other."""
        now  = time.time()
        last = getattr(self._tlocal, 'last', 0)
        if now - last < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - (now - last))
        self._tlocal.last = time.time()

    def _api_get(self, url: str, params=None, retries: int = 3):
        for attempt in range(retries):
            self._rate_wait()
            try:
                resp = requests.get(url, auth=self.auth, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429:
                    wait = int(resp.headers.get('Retry-After', '5'))
                    log(f"  429 — sleeping {wait}s", "WARN")
                    time.sleep(wait)
                    continue
                if resp.status_code in (500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                return None
            except Exception:
                time.sleep(2 ** attempt)
        return None

    def _fetch_ancestors(self, page_id: str) -> List[Dict]:
        data = self._api_get(f"{self.domain}/wiki/api/v2/pages/{page_id}/ancestors")
        if not data:
            return []
        return [{"id": a.get("id", ""), "title": a.get("title", "")}
                for a in data.get("results", [])]

    def _fetch_labels(self, page_id: str) -> List[str]:
        data = self._api_get(f"{self.domain}/wiki/api/v2/pages/{page_id}/labels")
        if not data:
            return []
        return [item.get("name", "") for item in data.get("results", []) if item.get("name")]

    def discover_spaces(self) -> List[Dict[str, Any]]:
        log("Discovering Confluence spaces...")
        spaces = []
        try:
            url = f"{self.domain}/wiki/api/v2/spaces"
            while url:
                data = self._api_get(url)
                if not data:
                    break
                spaces.extend(data.get('results', []))
                nxt = data.get('_links', {}).get('next')
                url = f"{self.domain}{nxt}" if nxt else None
            log(f"Discovered {len(spaces)} Confluence spaces")
            return spaces
        except Exception as e:
            log(f"Error discovering Confluence spaces: {e}", "ERROR")
            return []

    def ingest_space_pages(self, space_id: str, space_name: str, idx: int, total: int) -> int:
        """Ingest all pages from one space. Called from worker thread."""
        ingested = 0
        try:
            # KEY FIX: body-format=storage in the LIST call → no per-page fetch
            url    = f"{self.domain}/wiki/api/v2/pages"
            params = {
                "space-id":    space_id,
                "body-format": "storage",
                "limit":       50,
            }

            while url:
                data = self._api_get(url, params)
                if not data:
                    break
                pages = data.get('results', [])

                for page in pages:
                    page_id      = page.get('id')
                    body_content = page.get('body', {}).get('storage', {}).get('value', '')
                    version_info = page.get('version') or {}
                    author       = version_info.get('authorId', 'unknown')
                    last_mod_id  = version_info.get('authorId', '')
                    version_num  = version_info.get('number', 1)

                    web_url = page.get('_links', {}).get('webui', '')
                    if web_url and not web_url.startswith('http'):
                        web_url = f"{self.domain}/wiki{web_url}"

                    # Ancestors — breadcrumb path (non-fatal if fails)
                    ancestors  = []
                    breadcrumb = ''
                    try:
                        ancestors  = self._fetch_ancestors(page_id)
                        breadcrumb = " > ".join(a["title"] for a in ancestors)
                    except Exception:
                        pass

                    # Labels — trust/routing tags (non-fatal if fails)
                    labels = []
                    try:
                        labels = self._fetch_labels(page_id)
                    except Exception:
                        pass

                    doc = {
                        "source":            "confluence",
                        "type":              "page",
                        "space_id":          space_id,
                        "space_name":        space_name,
                        "page_id":           page_id,
                        "title":             page.get('title', ''),
                        "content":           body_content,
                        "status":            page.get('status', ''),
                        "author":            author,
                        "created_at":        page.get('createdAt', ''),
                        "web_url":           web_url,
                        "version":           version_num,
                        # ── Context fields ─────────────────────────────────────
                        "ancestors":         ancestors,
                        "breadcrumb":        breadcrumb,
                        "labels":            labels,
                        "last_modifier_id":  last_mod_id,
                        "ingested_at":       datetime.utcnow(),
                    }

                    with self._mongo_lock:
                        db.confluence_pages.update_one(
                            {"page_id": page_id},
                            {"$set": doc},
                            upsert=True,
                        )
                    ingested += 1

                nxt    = data.get('_links', {}).get('next')
                url    = f"{self.domain}{nxt}" if nxt else None
                params = None  # baked into next URL

            with self._c_lock:
                self._counter["done"]  += 1
                self._counter["pages"] += ingested
                done = self._counter["done"]
            log(f"  [{done}/{total}] ✓ {space_name} — {ingested} pages")

        except Exception as e:
            log(f"  ✗ Error in space '{space_name}': {e}", "ERROR")

        return ingested

    def ingest_all(self):
        log("=" * 60)
        log("Starting Confluence ingestion")
        log(f"Workers: {CONFLUENCE_WORKERS}")
        log("=" * 60)

        spaces = self.discover_spaces()
        total  = len(spaces)

        with ThreadPoolExecutor(max_workers=CONFLUENCE_WORKERS) as ex:
            futures = {
                ex.submit(self.ingest_space_pages,
                          s.get('id'), s.get('name', s.get('id')), i, total): s
                for i, s in enumerate(spaces, 1)
            }
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    sp = futures[fut]
                    log(f"  ✗ Unhandled error in '{sp.get('name')}': {e}", "ERROR")

        # Indexes
        db.confluence_pages.create_index("page_id",  unique=True)
        db.confluence_pages.create_index("space_id")
        db.confluence_pages.create_index("ingested_at")

        log(f"Confluence ingestion completed — {self._counter['pages']} pages across {total} spaces")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("Knowledge Base Ingestion Started")
    log("=" * 60)

    start_time = time.time()

    if GITLAB_TOKEN and GITLAB_TOKEN.strip():
        try:
            GitLabIngester().ingest_all()
        except Exception as e:
            log(f"GitLab ingestion failed: {e}", "ERROR")
            import traceback
            log(traceback.format_exc(), "ERROR")
    else:
        log(f"GITLAB_TOKEN not set — skipping GitLab", "WARN")

    if ATLASSIAN_EMAIL and ATLASSIAN_API_TOKEN and ATLASSIAN_DOMAIN:
        try:
            JiraIngester().ingest_all()
        except Exception as e:
            log(f"Jira ingestion failed: {e}", "ERROR")

        try:
            ConfluenceIngester().ingest_all()
        except Exception as e:
            log(f"Confluence ingestion failed: {e}", "ERROR")
    else:
        log("Atlassian credentials not configured — skipping Jira/Confluence", "WARN")

    elapsed = time.time() - start_time
    log("=" * 60)
    log(f"Ingestion completed in {elapsed:.2f}s")
    log("=" * 60)


if __name__ == "__main__":
    main()
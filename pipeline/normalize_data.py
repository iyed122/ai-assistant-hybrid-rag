#!/usr/bin/env python3
"""
Data Normalization Layer
Transforms GitLab, Jira, and Confluence raw data into a unified schema.

Performance model
──────────────────
The dominant cost in normalization is MongoDB write latency.  Issuing one
update_one per document costs ~440 000 round-trips for a 440k collection.
This version uses per-thread bulk writers that accumulate UpdateOne ops and
flush every BULK_FLUSH_SIZE documents, reducing round-trips by ~200×.

  Single update_one per doc  →  ~3–5  docs/sec/worker
  Bulk writes (200 ops/flush) →  ~80–150 docs/sec/worker

With NORM_WORKERS=4 and BULK_FLUSH_SIZE=200 the full 440k collection
should complete in 20–40 minutes instead of 5+ hours.

Other optimisations
────────────────────
- Lazy cursor (batch_size=200): raw collection never fully loaded into RAM
- Bounded in-flight window: futures queue stays at NORM_WORKERS × 20 max
- Resume via done-set: restarts skip already-normalised docs instantly
- Monster-doc splitting: content > SPLIT_THRESHOLD → _part_N entries
- Extra-filter hook: tiny code files skipped before thread submission

Tuning (.env knobs)
────────────────────
  NORM_WORKERS=4          # parallel threads
  BULK_FLUSH_SIZE=200     # ops per bulk_write call
  CURSOR_BATCH_SIZE=200   # docs per MongoDB network round-trip
  SPLIT_THRESHOLD=80000   # chars before a doc is split into parts
"""

import os
import re
import threading
from copy import deepcopy
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from pymongo import MongoClient
from pymongo.operations import UpdateOne
from dotenv import load_dotenv
from typing import Any, Callable, Dict, List, Set

load_dotenv()

MONGO_URI         = os.getenv('MONGO_URI',         'mongodb://localhost:27017/')
MONGO_DB          = os.getenv('MONGO_DB',          'knowledge_base')
SPLIT_THRESHOLD   = int(os.getenv('SPLIT_THRESHOLD',   '80000'))
NORM_WORKERS      = int(os.getenv('NORM_WORKERS',      '12'))
BULK_FLUSH_SIZE   = int(os.getenv('BULK_FLUSH_SIZE',   '500'))
CURSOR_BATCH_SIZE = int(os.getenv('CURSOR_BATCH_SIZE', '1000'))
MIN_CONTENT_CHARS = 30

# ── Thread-safe counters ───────────────────────────────────────────────────────
_lock        = threading.Lock()
_log_lock    = threading.Lock()
_done_count  = 0
_skip_count  = 0
_error_count = 0
_entry_count = 0


def log(message: str, level: str = "INFO") -> None:
    with _log_lock:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] [{level}] {message}")


def _inc(done: int = 0, skip: int = 0, err: int = 0, entries: int = 0) -> None:
    global _done_count, _skip_count, _error_count, _entry_count
    with _lock:
        _done_count  += done
        _skip_count  += skip
        _error_count += err
        _entry_count += entries


# ── Per-thread MongoDB connection ──────────────────────────────────────────────
_thread_local = threading.local()

def get_db():
    """One MongoDB connection per thread — no contention."""
    if not hasattr(_thread_local, 'db'):
        _thread_local.client = MongoClient(MONGO_URI)
        _thread_local.db     = _thread_local.client[MONGO_DB]
    return _thread_local.db


# ── Per-thread bulk writer ─────────────────────────────────────────────────────

class BulkWriter:
    """
    Accumulates UpdateOne operations and flushes them to MongoDB in batches.
    Each thread owns one BulkWriter — no shared state, no locks needed.

    Monster docs produce multiple UpdateOne ops (one per part) plus one
    DeleteOne to remove the original.  Both are handled transparently.
    """

    def __init__(self, flush_size: int = BULK_FLUSH_SIZE):
        self._flush_size = flush_size
        self._ops: List[UpdateOne] = []
        self._deletes: List[str]   = []   # original_ids to delete after parts are written
        self._entries = 0

    def add(self, doc: Dict[str, Any]) -> None:
        """
        Queue one normalised document.  Splits monster docs into parts
        automatically.  Flushes to MongoDB when the buffer is full.
        """
        content     = doc.get("content") or ""
        original_id = doc["document_id"]

        if len(content) <= SPLIT_THRESHOLD:
            self._ops.append(
                UpdateOne({"document_id": original_id}, {"$set": doc}, upsert=True)
            )
            self._entries += 1
        else:
            total_parts = (len(content) + SPLIT_THRESHOLD - 1) // SPLIT_THRESHOLD
            for idx in range(1, total_parts + 1):
                start    = (idx - 1) * SPLIT_THRESHOLD
                part_doc = deepcopy(doc)
                part_doc["document_id"] = f"{original_id}_part_{idx}"
                part_doc["content"]     = content[start: start + SPLIT_THRESHOLD]
                meta = part_doc.setdefault("metadata", {})
                meta.update({
                    "original_document_id": original_id,
                    "is_partial":           True,
                    "part_number":          idx,
                    "total_parts":          total_parts,
                })
                part_doc["normalized_at"] = datetime.utcnow()
                self._ops.append(
                    UpdateOne(
                        {"document_id": part_doc["document_id"]},
                        {"$set": part_doc},
                        upsert=True,
                    )
                )
                self._entries += 1
            # Schedule deletion of stale un-split version
            self._deletes.append(original_id)

        if len(self._ops) >= self._flush_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered ops to MongoDB and clear the buffer."""
        if not self._ops:
            return
        db = get_db()
        db.normalized_documents.bulk_write(self._ops, ordered=False)
        for orig_id in self._deletes:
            db.normalized_documents.delete_one({"document_id": orig_id})
        _inc(entries=self._entries)
        self._ops.clear()
        self._deletes.clear()
        self._entries = 0


# ── Resume: set of already-normalised base IDs ────────────────────────────────

def _base_id(document_id: str) -> str:
    return re.sub(r'_part_\d+$', '', document_id)


def build_done_set() -> Set[str]:
    log("Building resume set from normalized_documents …")
    c    = MongoClient(MONGO_URI)
    ids  = c[MONGO_DB].normalized_documents.distinct("document_id")
    c.close()
    done = {_base_id(d) for d in ids}
    log(f"  {len(done):,} base IDs already normalised — will skip.")
    return done


# ── Text utilities ─────────────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    if not text:
        return ""
    entities = {
        '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>',
        '&quot;': '"', '&#39;': "'", '&apos;': "'",
        '&eacute;': 'é', '&egrave;': 'è', '&agrave;': 'à',
        '&ecirc;': 'ê', '&ocirc;': 'ô', '&ucirc;': 'û',
        '&ccedil;': 'ç', '&ugrave;': 'ù', '&iuml;': 'ï',
        '&rsquo;': "'", '&lsquo;': "'", '&rdquo;': '"', '&ldquo;': '"',
        '&mdash;': '—', '&ndash;': '–', '&rarr;': '→', '&larr;': '←',
        '&hellip;': '...', '&copy;': '©', '&reg;': '®',
    }
    for k, v in entities.items():
        text = text.replace(k, v)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_text_from_adf(node: Any) -> str:
    if not node:
        return ""
    if isinstance(node, str):
        s = node.strip()
        if s.startswith('{') or s.startswith('['):
            try:
                import json
                node = json.loads(s)
            except Exception:
                return s
        else:
            return s
    if isinstance(node, list):
        return ' '.join(filter(None, [extract_text_from_adf(i) for i in node]))
    if not isinstance(node, dict):
        return str(node)
    parts     = []
    node_type = node.get('type', '')
    if 'text' in node:
        parts.append(str(node['text']))
    for item in node.get('content', []):
        t = extract_text_from_adf(item)
        if t:
            parts.append(t)
    if node_type == 'mention':
        return node.get('attrs', {}).get('text', '')
    for mark in node.get('marks', []):
        if mark.get('type') == 'link':
            href = mark.get('attrs', {}).get('href', '')
            if href and href not in parts:
                parts.append(f"[{href}]")
    result = ' '.join(filter(None, parts))
    if node_type in ('paragraph', 'heading', 'bulletList', 'orderedList',
                     'listItem', 'codeBlock', 'blockquote', 'rule'):
        return result + '\n'
    return result


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def detect_language(file_path: str, extension: str) -> str:
    ext_map = {
        '.py': 'python', '.pyi': 'python',
        '.js': 'javascript', '.mjs': 'javascript', '.cjs': 'javascript',
        '.ts': 'typescript', '.tsx': 'typescript', '.jsx': 'javascript',
        '.html': 'html', '.css': 'css', '.scss': 'scss', '.sass': 'sass',
        '.java': 'java', '.kt': 'kotlin', '.scala': 'scala', '.groovy': 'groovy',
        '.go': 'go', '.rs': 'rust',
        '.cpp': 'cpp', '.c': 'c', '.h': 'c', '.hpp': 'cpp', '.cs': 'csharp',
        '.rb': 'ruby', '.php': 'php', '.sh': 'bash', '.bash': 'bash', '.zsh': 'zsh',
        '.yml': 'yaml', '.yaml': 'yaml', '.toml': 'toml',
        '.ini': 'ini', '.cfg': 'ini', '.conf': 'conf',
        '.json': 'json', '.md': 'markdown', '.rst': 'rst', '.txt': 'text',
        '.sql': 'sql', '.dockerfile': 'dockerfile',
    }
    fname = file_path.split('/')[-1].lower()
    if fname in ('dockerfile', 'makefile') or fname.endswith('.dockerfile'):
        return fname
    if '.gitlab-ci' in fname or 'docker-compose' in fname:
        return 'yaml'
    return ext_map.get(extension.lower(), 'text')


# ── Normalizers ────────────────────────────────────────────────────────────────

class GitLabNormalizer:

    @staticmethod
    def normalize_issue(raw: Dict[str, Any]) -> Dict[str, Any]:
        parts = []
        if raw.get('description'):
            parts.append(raw['description'])
        for c in raw.get('comments', []):
            if c.get('body'):
                parts.append(f"Comment by {c['author']}: {c['body']}")
        return {
            "source": "gitlab", "source_type": "issue",
            "document_id":  f"gitlab_issue_{raw['project_id']}_{raw['issue_id']}",
            "project_id":   str(raw['project_id']),
            "project_name": raw.get('project_name', ''),
            "group_name":   raw.get('group_name', ''),
            "title":        raw.get('title', ''),
            "content":      clean_text("\n\n".join(parts)),
            "author":       raw.get('author', 'unknown'),
            "created_at":   raw.get('created_at', ''),
            "updated_at":   raw.get('updated_at', ''),
            "status":       raw.get('state', ''),
            "labels":       raw.get('labels', []),
            "url":          raw.get('web_url', ''),
            "metadata": {
                "issue_id":      raw.get('issue_id'),
                "state":         raw.get('state'),
                "milestone":     raw.get('milestone'),
                "comment_count": len(raw.get('comments', [])),
            },
            "normalized_at": datetime.utcnow(),
        }

    @staticmethod
    def normalize_merge_request(raw: Dict[str, Any]) -> Dict[str, Any]:
        parts = []
        if raw.get('description'):
            parts.append(raw['description'])
        parts.append(
            f"Merge: {raw.get('source_branch', '')} → {raw.get('target_branch', '')}"
        )
        for c in raw.get('comments', []):
            if c.get('body'):
                parts.append(f"Comment by {c['author']}: {c['body']}")
        return {
            "source": "gitlab", "source_type": "merge_request",
            "document_id":  f"gitlab_mr_{raw['project_id']}_{raw['mr_id']}",
            "project_id":   str(raw['project_id']),
            "project_name": raw.get('project_name', ''),
            "group_name":   raw.get('group_name', ''),
            "title":        raw.get('title', ''),
            "content":      clean_text("\n\n".join(parts)),
            "author":       raw.get('author', 'unknown'),
            "created_at":   raw.get('created_at', ''),
            "updated_at":   raw.get('updated_at', ''),
            "status":       raw.get('state', ''),
            "labels":       raw.get('labels', []),
            "url":          raw.get('web_url', ''),
            "metadata": {
                "mr_id":         raw.get('mr_id'),
                "source_branch": raw.get('source_branch'),
                "target_branch": raw.get('target_branch'),
                "state":         raw.get('state'),
                "merged_at":     raw.get('merged_at'),
                "comment_count": len(raw.get('comments', [])),
            },
            "normalized_at": datetime.utcnow(),
        }

    @staticmethod
    def normalize_milestone(raw: Dict[str, Any]) -> Dict[str, Any]:
        parts = [
            f"Milestone: {raw.get('title', '')}",
            f"State: {raw.get('state', 'unknown')}",
        ]
        if raw.get('description'):
            parts.append(raw['description'])
        if raw.get('due_date'):
            parts.append(f"Due date: {raw['due_date']}")
        if raw.get('start_date'):
            parts.append(f"Start date: {raw['start_date']}")
        return {
            "source": "gitlab", "source_type": "milestone",
            "document_id":  f"gitlab_milestone_{raw['project_id']}_{raw['milestone_id']}",
            "project_id":   str(raw['project_id']),
            "project_name": raw.get('project_name', ''),
            "group_name":   raw.get('group_name', ''),
            "title":        raw.get('title', ''),
            "content":      clean_text("\n".join(parts)),
            "author":       "unknown",
            "created_at":   raw.get('created_at', ''),
            "updated_at":   raw.get('updated_at', ''),
            "status":       raw.get('state', ''),
            "labels":       [],
            "url":          raw.get('web_url', ''),
            "metadata": {
                "milestone_id":  raw.get('milestone_id'),
                "milestone_iid": raw.get('milestone_iid'),
                "state":         raw.get('state'),
                "due_date":      raw.get('due_date'),
                "start_date":    raw.get('start_date'),
            },
            "normalized_at": datetime.utcnow(),
        }

    @staticmethod
    def normalize_code_file(raw: Dict[str, Any]) -> Dict[str, Any]:
        file_path = raw.get('file_path', '')
        ext       = raw.get('extension', '')
        language  = detect_language(file_path, ext)
        project   = raw.get('project_name', '')
        ref       = raw.get('ref', 'main')
        header    = (
            f"File: {file_path}  ({language})\n"
            f"Project: {project}  |  Branch: {ref}\n"
            f"{'─' * 60}\n"
        )
        return {
            "source": "gitlab", "source_type": "code_file",
            "document_id":  f"gitlab_code_{raw['project_id']}_{file_path.replace('/', '_')}",
            "project_id":   str(raw['project_id']),
            "project_name": project,
            "group_name":   raw.get('group_name', ''),
            "title":        f"{file_path}  [{project}]",
            "content":      header + raw.get('content', ''),
            "author":       "unknown",
            "created_at":   raw.get('ingested_at', datetime.utcnow()).isoformat()
                            if isinstance(raw.get('ingested_at'), datetime)
                            else str(raw.get('ingested_at', '')),
            "updated_at":   "",
            "status":       "current",
            "labels":       [language],
            "url":          raw.get('web_url', ''),
            "metadata": {
                "file_path":  file_path,
                "file_name":  raw.get('file_name', ''),
                "extension":  ext,
                "language":   language,
                "ref":        ref,
                "size_bytes": raw.get('size_bytes', 0),
            },
            "normalized_at": datetime.utcnow(),
        }


class ConfluenceNormalizer:

    @staticmethod
    def normalize_page(raw: Dict[str, Any]) -> Dict[str, Any]:
        body_text  = clean_text(strip_html(raw.get('content', '')))
        breadcrumb = raw.get('breadcrumb', '')
        labels     = raw.get('labels', [])
        parts      = []
        if breadcrumb:
            parts.append(f"Location: {breadcrumb}")
        parts.append(body_text)
        if labels:
            parts.append("Tags: " + ", ".join(labels))
        content = clean_text("\n\n".join(filter(None, parts)))
        web_url = raw.get('web_url', '')
        if not web_url:
            domain  = os.getenv('ATLASSIAN_DOMAIN', '').rstrip('/')
            page_id = raw.get('page_id', '')
            if domain and page_id:
                web_url = (
                    f"{domain}/wiki/spaces/{raw.get('space_name', '')}"
                    f"/pages/{page_id}"
                )
        return {
            "source": "confluence", "source_type": "page",
            "document_id":  f"confluence_page_{raw['page_id']}",
            "project_id":   raw.get('space_id', ''),
            "project_name": raw.get('space_name', ''),
            "group_name":   raw.get('space_name', ''),
            "title":        raw.get('title', ''),
            "content":      content,
            "author":       raw.get('author', 'unknown'),
            "created_at":   raw.get('created_at', ''),
            "updated_at":   raw.get('created_at', ''),
            "status":       raw.get('status', ''),
            "labels":       labels,
            "url":          web_url,
            "metadata": {
                "page_id":          raw.get('page_id'),
                "version":          raw.get('version'),
                "space_id":         raw.get('space_id'),
                "ancestors":        raw.get('ancestors', []),
                "breadcrumb":       breadcrumb,
                "last_modifier_id": raw.get('last_modifier_id', ''),
            },
            "normalized_at": datetime.utcnow(),
        }


# ── Worker ─────────────────────────────────────────────────────────────────────

def _process_one(raw: Dict[str, Any], normalizer_fn: Callable,
                 done_set: Set[str], label: str,
                 writer: BulkWriter) -> None:
    """
    Normalise one raw document and queue it in the thread-local BulkWriter.
    The writer flushes automatically when its buffer hits BULK_FLUSH_SIZE.
    """
    try:
        doc    = normalizer_fn(raw)
        doc_id = doc["document_id"]

        if _base_id(doc_id) in done_set:
            _inc(skip=1)
            return

        writer.add(doc)
        _inc(done=1)

    except Exception as exc:
        _inc(err=1)
        log(f"  ✗ [{label}] {type(exc).__name__}: {exc}", "ERROR")


# ── Lazy parallel runner ───────────────────────────────────────────────────────

def _run_parallel(collection_name: str, normalizer_fn: Callable,
                  done_set: Set[str], label: str,
                  extra_filter: Callable = None) -> None:
    """
    Stream raw docs via a lazy cursor, skip already-done docs, process the
    rest in a bounded thread pool.  Each thread owns its own BulkWriter so
    flushes never block each other.
    """
    c     = MongoClient(MONGO_URI)
    db    = c[MONGO_DB]
    total = db[collection_name].count_documents({})

    if total == 0:
        c.close()
        log(f"  (no documents in {collection_name})")
        return

    log(f"  {total:,} raw documents — streaming …")

    max_inflight = NORM_WORKERS * 20
    in_flight: set = set()
    submitted  = 0

    # Each thread gets its own BulkWriter (keyed by thread id)
    writers: Dict[int, BulkWriter] = {}
    writers_lock = threading.Lock()

    def get_writer() -> BulkWriter:
        tid = threading.get_ident()
        with writers_lock:
            if tid not in writers:
                writers[tid] = BulkWriter(BULK_FLUSH_SIZE)
            return writers[tid]

    def task(raw):
        _process_one(raw, normalizer_fn, done_set, label, get_writer())

    cursor = db[collection_name].find({}).batch_size(CURSOR_BATCH_SIZE)

    with ThreadPoolExecutor(max_workers=NORM_WORKERS) as pool:
        for raw in cursor:
            if extra_filter and not extra_filter(raw):
                _inc(skip=1)
                submitted += 1
                continue

            while len(in_flight) >= max_inflight:
                done_now, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for f in done_now:
                    f.result()

            in_flight.add(pool.submit(task, raw))
            submitted += 1

            if submitted % 5000 == 0:
                with _lock:
                    d, s, e = _done_count, _skip_count, _error_count
                log(f"  [{label}] {submitted:,}/{total:,} streamed — "
                    f"written {d:,}  skipped {s:,}  errors {e:,}")

        # Drain remaining futures
        for f in in_flight:
            f.result()

    # Flush any remaining buffered ops from every thread's writer
    for writer in writers.values():
        writer.flush()

    c.close()

    with _lock:
        d, s, e = _done_count, _skip_count, _error_count
    log(f"  [{label}] ✓ complete — written {d:,}  skipped {s:,}  errors {e:,}")


# ── Main ───────────────────────────────────────────────────────────────────────

def normalize_all_data() -> None:
    log("=" * 60)
    log("Starting Data Normalization")
    log("=" * 60)
    log(f"Workers:         {NORM_WORKERS}   (NORM_WORKERS)")
    log(f"Bulk flush:      {BULK_FLUSH_SIZE}  ops/flush  (BULK_FLUSH_SIZE)")
    log(f"Cursor batch:    {CURSOR_BATCH_SIZE}  docs/round-trip  (CURSOR_BATCH_SIZE)")
    log(f"Split threshold: {SPLIT_THRESHOLD:,} chars  (SPLIT_THRESHOLD)")
    log("")

    done_set = build_done_set()
    log("")

    log("Normalizing GitLab Issues …")
    _run_parallel("gitlab_issues", GitLabNormalizer.normalize_issue,
                  done_set, "gitlab_issue")

    log("Normalizing GitLab Merge Requests …")
    _run_parallel("gitlab_merge_requests", GitLabNormalizer.normalize_merge_request,
                  done_set, "gitlab_mr")

    log("Normalizing GitLab Milestones …")
    _run_parallel("gitlab_milestones", GitLabNormalizer.normalize_milestone,
                  done_set, "gitlab_milestone")

    log("Normalizing GitLab Code Files …")
    _run_parallel(
        "gitlab_code_files", GitLabNormalizer.normalize_code_file,
        done_set, "code_file",
        extra_filter=lambda r: len(r.get('content', '')) >= MIN_CONTENT_CHARS,
    )

   
    log("Normalizing Confluence Pages …")
    _run_parallel("confluence_pages", ConfluenceNormalizer.normalize_page,
                  done_set, "confluence_page")

    # ── Indexes ───────────────────────────────────────────────────────────────
    log("Creating indexes …")
    c  = MongoClient(MONGO_URI)
    db = c[MONGO_DB]
    db.normalized_documents.create_index([("document_id",   1)], unique=True)
    db.normalized_documents.create_index([("source",        1)])
    db.normalized_documents.create_index([("source_type",   1)])
    db.normalized_documents.create_index([("project_id",    1)])
    db.normalized_documents.create_index([("created_at",   -1)])
    db.normalized_documents.create_index([("normalized_at", -1)])
    db.normalized_documents.create_index([("source_type",   1), ("project_id", 1)])

    log("=" * 60)
    log("Normalization complete")
    log("=" * 60)
    with _lock:
        log(f"  Written:  {_entry_count:,} entries  ({_done_count:,} source docs)")
        log(f"  Skipped:  {_skip_count:,}")
        log(f"  Errors:   {_error_count:,}")

    log("\nBreakdown by source_type:")
    for st in ("issue", "merge_request", "milestone", "code_file", "page"):
        n = db.normalized_documents.count_documents({"source_type": st})
        log(f"  {st:<20} {n:,}")
    log(f"  {'TOTAL':<20} {db.normalized_documents.count_documents({}):,}")
    c.close()


if __name__ == "__main__":
    normalize_all_data()

#!/usr/bin/env python3
"""
neo4j_import.py  —  dual-mode  (MongoDB  |  JSONL)
===================================================
Replaces both migrate_to_qdrant.py and import_to_qdrant.py.

IMPORT_SOURCE env var (default: mongo)
  mongo  — reads from MongoDB document_chunks  (normal local pipeline)
  jsonl  — reads from DATA/embedded_*.jsonl    (one-time AWS import, already done)

Graph schema
────────────
Nodes:  Chunk · JiraTicket · ConfluencePage · GitLabDoc
        Project · Space · User · Component · Version · Label

Relationships:
  (Chunk)       -[:PART_OF]       → JiraTicket | ConfluencePage | GitLabDoc
  (JiraTicket)  -[:CHILD_OF]      → JiraTicket          (parent_key)
  (JiraTicket)  -[:BLOCKS]        → JiraTicket
  (JiraTicket)  -[:BLOCKED_BY]    → JiraTicket
  (JiraTicket)  -[:RELATES_TO]    → JiraTicket
  (JiraTicket)  -[:IS_TESTED_BY]  → JiraTicket
  (JiraTicket)  -[:TESTS]         → JiraTicket
  (JiraTicket)  -[:CAUSES]        → JiraTicket
  (JiraTicket)  -[:CAUSED_BY]     → JiraTicket
  (JiraTicket)  -[:CLONES]        → JiraTicket
  (JiraTicket)  -[:CLONED_BY]     → JiraTicket
  (JiraTicket)  -[:DUPLICATES]    → JiraTicket
  (JiraTicket)  -[:DUPLICATED_BY] → JiraTicket
  (JiraTicket)  -[:IMPLEMENTS]    → JiraTicket
  (JiraTicket)  -[:IMPLEMENTED_BY]→ JiraTicket
  (JiraTicket)  -[:AUTOMATES]     → JiraTicket
  (JiraTicket)  -[:IS_AUTOMATED_BY]→JiraTicket
  (JiraTicket)  -[:SPLIT_FROM]    → JiraTicket
  (JiraTicket)  -[:SPLIT_TO]      → JiraTicket
  (JiraTicket)  -[:REVIEWED_BY]   → JiraTicket
  (JiraTicket)  -[:IN_PROJECT]    → Project
  (JiraTicket)  -[:ASSIGNED_TO]   → User
  (JiraTicket)  -[:FIX_IN]        → Version
  (JiraTicket)  -[:HAS_COMPONENT] → Component
  (JiraTicket | ConfluencePage | GitLabDoc) -[:HAS_LABEL] → Label
  (ConfluencePage) -[:IN_SPACE]   → Space
  (ConfluencePage) -[:CHILD_OF]   → ConfluencePage       (ancestors)

.env keys
─────────
  NEO4J_URI           bolt://localhost:7687
  NEO4J_USER          neo4j
  NEO4J_PASSWORD      your_password
  NEO4J_DATABASE      neo4j
  EMBEDDING_DIM       768
  IMPORT_BATCH_SIZE   300
  IMPORT_SOURCE       mongo          # or jsonl
  MONGO_URI           mongodb://localhost:27017/
  MONGO_DB            knowledge_base
  JIRA_JSONL          embedded_jira.jsonl        # relative to DATA/
  CONFLUENCE_JSONL    embedded_confluence.jsonl
  IMPORT_GITLAB       false
"""

import os
import re
import json
import time
from datetime import datetime
from typing import Iterator, Dict, Any, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from neo4j import GraphDatabase, Driver
    NEO4J_OK = True
except ImportError:
    NEO4J_OK = False
    print("pip install neo4j")

try:
    from pymongo import MongoClient
    MONGO_OK = True
except ImportError:
    MONGO_OK = False

# ── Configuration ──────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",           "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",          "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD",      "your_password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE",      "neo4j")
EMBEDDING_DIM  = int(os.getenv("EMBEDDING_DIM",       "768"))
BATCH_SIZE     = int(os.getenv("IMPORT_BATCH_SIZE",   "300"))
IMPORT_GITLAB  = os.getenv("IMPORT_GITLAB", "false").lower() == "true"

# Data source: "mongo" for local pipeline, "jsonl" for one-time AWS import
IMPORT_SOURCE  = os.getenv("IMPORT_SOURCE", "mongo").lower()

# MongoDB
MONGO_URI      = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME  = os.getenv("MONGO_DB",  "knowledge_base")

# JSONL paths (only used when IMPORT_SOURCE=jsonl)
_HERE          = os.path.dirname(os.path.abspath(__file__))
_ROOT          = os.path.dirname(_HERE)               # AI Assistant/
_DATA          = os.path.join(_ROOT, "DATA")
JIRA_JSONL     = os.path.join(_DATA, os.getenv("JIRA_JSONL",       "embedded_jira.jsonl"))
CONF_JSONL     = os.path.join(_DATA, os.getenv("CONFLUENCE_JSONL", "embedded_confluence.jsonl"))


def log(msg: str, level: str = "INFO") -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{level}] {msg}", flush=True)


# ── ID helpers ─────────────────────────────────────────────────────────────────
_PART_RE = re.compile(r'_part_\d+$')

def extract_issue_key(pdid: str) -> str:
    if not pdid.startswith("jira_issue_"):
        return ""
    return _PART_RE.sub('', pdid[len("jira_issue_"):])

def extract_page_id(pdid: str) -> str:
    if not pdid.startswith("confluence_page_"):
        return ""
    return _PART_RE.sub('', pdid[len("confluence_page_"):])


# ── Link type map  (all 20 from your aggregate) ────────────────────────────────
_LINK_MAP = {
    "relates to":          "RELATES_TO",
    "is tested by":        "IS_TESTED_BY",
    "tests":               "TESTS",
    "created by":          "CREATED_BY",
    "created":             "CREATED_LINK",
    "blocks":              "BLOCKS",
    "is blocked by":       "BLOCKED_BY",
    "causes":              "CAUSES",
    "is caused by":        "CAUSED_BY",
    "clones":              "CLONES",
    "is cloned by":        "CLONED_BY",
    "is automated by":     "IS_AUTOMATED_BY",
    "automates":           "AUTOMATES",
    "is duplicated by":    "DUPLICATED_BY",
    "duplicates":          "DUPLICATES",
    "split from":          "SPLIT_FROM",
    "split to":            "SPLIT_TO",
    "implements":          "IMPLEMENTS",
    "is implemented by":   "IMPLEMENTED_BY",
    "is reviewed by":      "REVIEWED_BY",
    "depends on":          "DEPENDS_ON",
    "is dependency of":    "IS_DEPENDENCY_OF",
}

def link_rel(raw: str) -> str:
    return _LINK_MAP.get((raw or "").lower().strip(), "RELATES_TO")


# ── Streaming ──────────────────────────────────────────────────────────────────

def stream_jsonl(path: str) -> Iterator[Dict]:
    """Stream chunks from an embedded JSONL file (AWS pipeline output)."""
    if not os.path.exists(path):
        log(f"File not found: {path}", "WARN")
        return
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                log(f"  Bad line {i}: {e}", "WARN")


def count_jsonl(path: str) -> int:
    n = 0
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as fh:
        for _ in fh:
            n += 1
    return n


def stream_mongo(source_filter: str) -> Iterator[Dict]:
    """Stream embedded chunks from MongoDB document_chunks collection."""
    if not MONGO_OK:
        log("pymongo not installed — pip install pymongo", "ERROR")
        return
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    query: Dict[str, Any] = {"embedding": {"$exists": True}}
    if source_filter != "all":
        query["source"] = source_filter
    total = db.document_chunks.count_documents(query)
    log(f"  MongoDB: {total:,} '{source_filter}' chunks with embeddings")
    try:
        for chunk in db.document_chunks.find(query, {"_id": 0}).batch_size(500):
            yield chunk
    finally:
        client.close()


def count_mongo(source_filter: str) -> int:
    if not MONGO_OK:
        return 0
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    query: Dict[str, Any] = {"embedding": {"$exists": True}}
    if source_filter != "all":
        query["source"] = source_filter
    n = db.document_chunks.count_documents(query)
    client.close()
    return n


def get_stream(source: str):
    """Return (stream_fn, total_count) based on IMPORT_SOURCE."""
    if IMPORT_SOURCE == "jsonl":
        path = JIRA_JSONL if source == "jira" else CONF_JSONL
        return stream_jsonl(path), count_jsonl(path)
    else:
        return stream_mongo(source), count_mongo(source)


# ── Cypher DDL ─────────────────────────────────────────────────────────────────
CONSTRAINTS = """
CREATE CONSTRAINT chunk_id_uniq        IF NOT EXISTS FOR (c:Chunk)          REQUIRE c.chunk_id    IS UNIQUE;
CREATE CONSTRAINT jira_key_uniq        IF NOT EXISTS FOR (t:JiraTicket)     REQUIRE t.issue_key   IS UNIQUE;
CREATE CONSTRAINT conf_page_uniq       IF NOT EXISTS FOR (p:ConfluencePage) REQUIRE p.page_id     IS UNIQUE;
CREATE CONSTRAINT gitlab_doc_uniq      IF NOT EXISTS FOR (g:GitLabDoc)      REQUIRE g.document_id IS UNIQUE;
CREATE CONSTRAINT project_id_uniq      IF NOT EXISTS FOR (r:Project)        REQUIRE r.project_id  IS UNIQUE;
CREATE CONSTRAINT user_name_uniq       IF NOT EXISTS FOR (u:User)           REQUIRE u.name        IS UNIQUE;
CREATE CONSTRAINT component_name_uniq  IF NOT EXISTS FOR (c:Component)      REQUIRE c.name        IS UNIQUE;
CREATE CONSTRAINT version_name_uniq    IF NOT EXISTS FOR (v:Version)        REQUIRE v.name        IS UNIQUE;
CREATE CONSTRAINT label_name_uniq      IF NOT EXISTS FOR (l:Label)          REQUIRE l.name        IS UNIQUE;
CREATE CONSTRAINT space_id_uniq        IF NOT EXISTS FOR (s:Space)          REQUIRE s.space_id    IS UNIQUE;
"""

VECTOR_INDEX = """
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: $dim,
  `vector.similarity_function`: 'cosine'
}}
"""

EXTRA_INDEXES = """
CREATE INDEX chunk_source       IF NOT EXISTS FOR (c:Chunk) ON (c.source);
CREATE INDEX chunk_source_type  IF NOT EXISTS FOR (c:Chunk) ON (c.source_type);
CREATE INDEX chunk_project_name IF NOT EXISTS FOR (c:Chunk) ON (c.project_name);
CREATE INDEX chunk_status       IF NOT EXISTS FOR (c:Chunk) ON (c.status);
CREATE INDEX jira_status        IF NOT EXISTS FOR (t:JiraTicket) ON (t.status);
CREATE INDEX jira_priority      IF NOT EXISTS FOR (t:JiraTicket) ON (t.priority);
CREATE INDEX jira_project       IF NOT EXISTS FOR (t:JiraTicket) ON (t.project_name);
CREATE INDEX jira_issue_type    IF NOT EXISTS FOR (t:JiraTicket) ON (t.issue_type);
"""

# ── Jira Cypher ────────────────────────────────────────────────────────────────
JIRA_NODES_CQL = """
UNWIND $rows AS r

MERGE (c:Chunk {chunk_id: r.chunk_id})
SET c.text            = r.text,
    c.embedding       = r.embedding,
    c.token_count     = r.token_count,
    c.chunk_index     = r.chunk_index,
    c.total_chunks    = r.total_chunks,
    c.source          = r.source,
    c.source_type     = r.source_type,
    c.title           = r.title,
    c.url             = r.url,
    c.status          = r.status,
    c.author          = r.author,
    c.project_name    = r.project_name,
    c.group_name      = r.group_name,
    c.created_at      = r.created_at,
    c.embedding_model = r.embedding_model

WITH c, r
WHERE r.issue_key <> ''
MERGE (t:JiraTicket {issue_key: r.issue_key})
SET t.title        = r.title,
    t.status       = r.status,
    t.priority     = r.priority,
    t.issue_type   = r.issue_type,
    t.assignee     = r.assignee,
    t.resolution   = r.resolution,
    t.url          = r.url,
    t.project_id   = r.project_id,
    t.project_name = r.project_name
MERGE (c)-[:PART_OF]->(t)

WITH c, t, r
WHERE r.project_id <> ''
MERGE (p:Project {project_id: r.project_id})
  ON CREATE SET p.name = r.project_name, p.source = 'jira'
MERGE (t)-[:IN_PROJECT]->(p)

WITH t, r
WHERE r.assignee <> ''
MERGE (u:User {name: r.assignee})
MERGE (t)-[:ASSIGNED_TO]->(u)
"""

# FIX: WITH r WHERE instead of bare WHERE after UNWIND (Neo4j 5.x)
JIRA_RELS_CQL = """
UNWIND $rows AS r
WITH r WHERE r.issue_key <> ''
MATCH (t:JiraTicket {issue_key: r.issue_key})

FOREACH (_ IN CASE WHEN r.parent_key <> '' THEN [1] ELSE [] END |
  MERGE (par:JiraTicket {issue_key: r.parent_key})
  MERGE (t)-[:CHILD_OF]->(par)
)
FOREACH (vn IN r.fix_versions |
  MERGE (v:Version {name: vn})
  MERGE (t)-[:FIX_IN]->(v)
)
FOREACH (cn IN r.components |
  MERGE (comp:Component {name: cn})
  MERGE (t)-[:HAS_COMPONENT]->(comp)
)
FOREACH (ln IN r.labels |
  MERGE (lbl:Label {name: ln})
  MERGE (t)-[:HAS_LABEL]->(lbl)
)
"""

# Pass 3: issue_links via APOC (all 20 typed relationships)
JIRA_LINKS_APOC_CQL = """
UNWIND $rows AS r
MATCH (src:JiraTicket {issue_key: r.issue_key})
UNWIND r.issue_links AS lnk
MERGE (tgt:JiraTicket {issue_key: lnk.target})
WITH src, tgt, lnk
CALL apoc.create.relationship(src, lnk.rel_type, {}, tgt) YIELD rel
RETURN count(rel)
"""

# Fallback without APOC
JIRA_LINKS_STUB_CQL = """
UNWIND $rows AS r
MATCH (src:JiraTicket {issue_key: r.issue_key})
UNWIND r.issue_links AS lnk
MERGE (tgt:JiraTicket {issue_key: lnk.target})
MERGE (src)-[:RELATES_TO]->(tgt)
"""

# ── Confluence Cypher ──────────────────────────────────────────────────────────
CONF_NODES_CQL = """
UNWIND $rows AS r

MERGE (c:Chunk {chunk_id: r.chunk_id})
SET c.text            = r.text,
    c.embedding       = r.embedding,
    c.token_count     = r.token_count,
    c.chunk_index     = r.chunk_index,
    c.total_chunks    = r.total_chunks,
    c.source          = r.source,
    c.source_type     = r.source_type,
    c.title           = r.title,
    c.url             = r.url,
    c.status          = r.status,
    c.author          = r.author,
    c.project_name    = r.project_name,
    c.group_name      = r.group_name,
    c.created_at      = r.created_at,
    c.embedding_model = r.embedding_model

WITH c, r
WHERE r.page_id <> ''
MERGE (pg:ConfluencePage {page_id: r.page_id})
SET pg.title      = r.title,
    pg.breadcrumb = r.breadcrumb,
    pg.space_id   = r.space_id,
    pg.space_name = r.project_name,
    pg.url        = r.url,
    pg.status     = r.status
MERGE (c)-[:PART_OF]->(pg)

WITH c, pg, r
WHERE r.space_id <> ''
MERGE (sp:Space {space_id: r.space_id})
  ON CREATE SET sp.name = r.project_name
MERGE (pg)-[:IN_SPACE]->(sp)

WITH pg, r
FOREACH (ln IN r.labels |
  MERGE (lbl:Label {name: ln})
  MERGE (pg)-[:HAS_LABEL]->(lbl)
)
"""

CONF_ANCESTORS_CQL = """
UNWIND $rows AS r
WITH r WHERE r.page_id <> ''
MATCH (pg:ConfluencePage {page_id: r.page_id})
UNWIND r.ancestors AS anc
  MERGE (parent:ConfluencePage {page_id: anc.id})
    ON CREATE SET parent.title = anc.title
  MERGE (pg)-[:CHILD_OF]->(parent)
"""

# ── GitLab Cypher ──────────────────────────────────────────────────────────────
GITLAB_NODES_CQL = """
UNWIND $rows AS r

MERGE (c:Chunk {chunk_id: r.chunk_id})
SET c.text            = r.text,
    c.embedding       = r.embedding,
    c.token_count     = r.token_count,
    c.chunk_index     = r.chunk_index,
    c.total_chunks    = r.total_chunks,
    c.source          = r.source,
    c.source_type     = r.source_type,
    c.title           = r.title,
    c.url             = r.url,
    c.status          = r.status,
    c.author          = r.author,
    c.project_name    = r.project_name,
    c.group_name      = r.group_name,
    c.created_at      = r.created_at,
    c.embedding_model = r.embedding_model

WITH c, r
MERGE (g:GitLabDoc {document_id: r.parent_document_id})
SET g.title        = r.title,
    g.source_type  = r.source_type,
    g.url          = r.url,
    g.status       = r.status,
    g.author       = r.author,
    g.project_id   = r.project_id,
    g.project_name = r.project_name
MERGE (c)-[:PART_OF]->(g)

WITH g, r
WHERE r.project_id <> ''
MERGE (p:Project {project_id: r.project_id})
  ON CREATE SET p.name = r.project_name, p.source = 'gitlab'
MERGE (g)-[:IN_PROJECT]->(p)

WITH g, r
FOREACH (ln IN r.labels |
  MERGE (lbl:Label {name: ln})
  MERGE (g)-[:HAS_LABEL]->(lbl)
)
"""


# ── Row builders ───────────────────────────────────────────────────────────────

def _base(doc: Dict) -> Dict:
    return {
        "chunk_id":           doc.get("chunk_id", ""),
        "parent_document_id": doc.get("parent_document_id", ""),
        "text":               doc.get("text", ""),
        "embedding":          doc.get("embedding"),
        "token_count":        doc.get("token_count", 0),
        "chunk_index":        doc.get("chunk_index", 0),
        "total_chunks":       doc.get("total_chunks", 1),
        "source":             doc.get("source", ""),
        "source_type":        doc.get("source_type", ""),
        "title":              doc.get("title", ""),
        "url":                doc.get("url", ""),
        "status":             doc.get("status", ""),
        "author":             doc.get("author", ""),
        "project_id":         str(doc.get("project_id") or ""),
        "project_name":       doc.get("project_name", ""),
        "group_name":         doc.get("group_name", ""),
        "created_at":         str(doc.get("created_at", "")),
        "embedding_model":    doc.get("embedding_model", "snowflake-arctic-embed-m"),
        "labels":             list(doc.get("labels") or []),
    }


def build_jira_row(doc: Dict) -> Optional[Dict]:
    if not doc.get("embedding"):
        return None
    meta = doc.get("metadata") or {}
    pdid = doc.get("parent_document_id", "")

    raw_links = meta.get("issue_links") or []
    links = []
    for lnk in raw_links:
        target = (lnk.get("target") or "").strip()
        if target:
            links.append({
                "target":   target,
                "rel_type": link_rel(lnk.get("type", "relates to")),
            })

    row = _base(doc)
    row.update({
        "issue_key":   extract_issue_key(pdid),
        "issue_type":  meta.get("issue_type")  or "",
        "priority":    meta.get("priority")    or "",
        "assignee":    meta.get("assignee")    or "",
        "resolution":  meta.get("resolution")  or "",
        "parent_key":  meta.get("parent_key")  or "",
        "fix_versions":list(meta.get("fix_versions") or []),
        "components":  list(meta.get("components")   or []),
        "issue_links": links,
    })
    return row


def build_confluence_row(doc: Dict) -> Optional[Dict]:
    if not doc.get("embedding"):
        return None
    meta = doc.get("metadata") or {}
    pdid = doc.get("parent_document_id", "")

    raw_anc = meta.get("ancestors") or []
    ancestors = []
    for a in raw_anc:
        if isinstance(a, dict) and a.get("id"):
            ancestors.append({"id": str(a["id"]), "title": a.get("title", "")})
        elif isinstance(a, str) and a:
            ancestors.append({"id": a, "title": a})

    row = _base(doc)
    row.update({
        "page_id":    extract_page_id(pdid) or str(meta.get("page_id", "")),
        "space_id":   str(meta.get("space_id") or doc.get("project_id", "") or ""),
        "breadcrumb": meta.get("breadcrumb") or "",
        "ancestors":  ancestors,
    })
    return row


def build_gitlab_row(doc: Dict) -> Optional[Dict]:
    if not doc.get("embedding"):
        return None
    return _base(doc)


# ── Importer ───────────────────────────────────────────────────────────────────

class Importer:

    def __init__(self):
        if not NEO4J_OK:
            raise ImportError("pip install neo4j")
        log(f"Connecting to Neo4j at {NEO4J_URI} …")
        self.driver: Driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.driver.verify_connectivity()
        log("✓ Connected")
        self._apoc = self._check_apoc()
        if self._apoc:
            log("✓ APOC detected — full relationship type fidelity enabled")
        else:
            log("  APOC not detected — issue_links will be RELATES_TO stubs", "WARN")

    def close(self):
        self.driver.close()

    def _check_apoc(self) -> bool:
        try:
            with self.driver.session(database=NEO4J_DATABASE) as s:
                s.run("RETURN apoc.version()").single()
            return True
        except Exception:
            return False

    def _run(self, cypher: str, params: dict = None):
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run(cypher, params or {})

    def _run_multi(self, block: str):
        for stmt in block.split(";"):
            stmt = stmt.strip()
            if stmt:
                self._run(stmt)

    def _write(self, cypher: str, rows: List[Dict]):
        if rows:
            self._run(cypher, {"rows": rows})

    def existing_ids(self, source: str) -> set:
        log(f"  Checking existing {source} chunks for resume …")
        with self.driver.session(database=NEO4J_DATABASE) as s:
            ids = {r["id"] for r in s.run(
                "MATCH (c:Chunk {source:$src}) RETURN c.chunk_id AS id",
                src=source
            )}
        log(f"  {len(ids):,} already in Neo4j — will skip")
        return ids

    def setup_schema(self):
        log("Setting up constraints and indexes …")
        self._run_multi(CONSTRAINTS)
        self._run(VECTOR_INDEX, {"dim": EMBEDDING_DIM})
        self._run_multi(EXTRA_INDEXES)
        log("  ✓ Schema ready")

    # ── Jira ───────────────────────────────────────────────────────────────────

    def import_jira(self):
        log("=" * 60)
        stream, total = get_stream("jira")
        log(f"Importing Jira  ←  {'MongoDB' if IMPORT_SOURCE == 'mongo' else JIRA_JSONL}")
        log("=" * 60)

        done     = self.existing_ids("jira")
        imported = skipped = errors = 0
        t0       = time.time()
        node_buf: List[Dict] = []
        rel_buf:  List[Dict] = []
        link_buf: List[Dict] = []

        def flush():
            nonlocal imported
            self._write(JIRA_NODES_CQL, node_buf)
            self._write(JIRA_RELS_CQL,  rel_buf)
            imported += len(node_buf)
            elapsed = time.time() - t0
            rate    = imported / elapsed if elapsed else 0
            remain  = max(total - imported, 0)
            eta_m   = (remain / rate / 60) if rate else 0
            log(f"  Jira {imported:,}/{total:,} ({imported/max(total,1)*100:.1f}%) "
                f"| {rate:.0f}/s | ETA ~{eta_m:.1f}m")
            node_buf.clear(); rel_buf.clear()

        for doc in stream:
            cid = doc.get("chunk_id", "")
            if cid in done:
                skipped += 1
                continue
            row = build_jira_row(doc)
            if row is None:
                errors += 1
                continue
            node_buf.append(row)
            rel_buf.append(row)
            if row.get("issue_links"):
                link_buf.append({
                    "issue_key":   row["issue_key"],
                    "issue_links": row["issue_links"],
                })
            if len(node_buf) >= BATCH_SIZE:
                flush()

        if node_buf:
            flush()

        if link_buf:
            log(f"  Writing {len(link_buf):,} issue-link relationships …")
            cypher = JIRA_LINKS_APOC_CQL if self._apoc else JIRA_LINKS_STUB_CQL
            for i in range(0, len(link_buf), BATCH_SIZE):
                self._write(cypher, link_buf[i:i + BATCH_SIZE])
            log(f"  ✓ Issue links written")

        log(f"  ✓ Jira: {imported:,} imported | {skipped:,} skipped | {errors} errors")

    # ── Confluence ─────────────────────────────────────────────────────────────

    def import_confluence(self):
        log("=" * 60)
        stream, total = get_stream("confluence")
        log(f"Importing Confluence  ←  {'MongoDB' if IMPORT_SOURCE == 'mongo' else CONF_JSONL}")
        log("=" * 60)

        done     = self.existing_ids("confluence")
        imported = skipped = errors = 0
        t0       = time.time()
        node_buf: List[Dict] = []
        anc_buf:  List[Dict] = []

        def flush():
            nonlocal imported
            self._write(CONF_NODES_CQL, node_buf)
            if anc_buf:
                self._write(CONF_ANCESTORS_CQL, anc_buf)
            imported += len(node_buf)
            elapsed = time.time() - t0
            rate    = imported / elapsed if elapsed else 0
            remain  = max(total - imported, 0)
            eta_m   = (remain / rate / 60) if rate else 0
            log(f"  Confluence {imported:,}/{total:,} ({imported/max(total,1)*100:.1f}%) "
                f"| {rate:.0f}/s | ETA ~{eta_m:.1f}m")
            node_buf.clear(); anc_buf.clear()

        for doc in stream:
            cid = doc.get("chunk_id", "")
            if cid in done:
                skipped += 1
                continue
            row = build_confluence_row(doc)
            if row is None:
                errors += 1
                continue
            node_buf.append(row)
            if row.get("ancestors"):
                anc_buf.append({"page_id": row["page_id"], "ancestors": row["ancestors"]})
            if len(node_buf) >= BATCH_SIZE:
                flush()

        if node_buf:
            flush()

        log(f"  ✓ Confluence: {imported:,} imported | {skipped:,} skipped | {errors} errors")

    # ── GitLab ─────────────────────────────────────────────────────────────────

    def import_gitlab(self):
        log("=" * 60)
        log("Importing GitLab  ←  MongoDB document_chunks")
        log("=" * 60)

        done = self.existing_ids("gitlab")
        imported = skipped = errors = 0
        buf: List[Dict] = []

        def flush():
            nonlocal imported
            self._write(GITLAB_NODES_CQL, buf)
            imported += len(buf)
            log(f"  GitLab {imported:,} imported …")
            buf.clear()

        for doc in stream_mongo("gitlab"):
            cid = doc.get("chunk_id", "")
            if cid in done:
                skipped += 1
                continue
            row = build_gitlab_row(doc)
            if row is None:
                errors += 1
                continue
            buf.append(row)
            if len(buf) >= BATCH_SIZE:
                flush()

        if buf:
            flush()

        log(f"  ✓ GitLab: {imported:,} imported | {skipped:,} skipped | {errors} errors")

    def stats(self):
        log("")
        log("=" * 60)
        log("Neo4j Graph Statistics")
        log("=" * 60)
        qs = [
            ("Total Chunks",        "MATCH (c:Chunk) RETURN count(c) AS n"),
            ("  jira chunks",       "MATCH (c:Chunk {source:'jira'}) RETURN count(c) AS n"),
            ("  confluence chunks", "MATCH (c:Chunk {source:'confluence'}) RETURN count(c) AS n"),
            ("  gitlab chunks",     "MATCH (c:Chunk {source:'gitlab'}) RETURN count(c) AS n"),
            ("JiraTicket nodes",    "MATCH (t:JiraTicket) RETURN count(t) AS n"),
            ("ConfluencePage nodes","MATCH (p:ConfluencePage) RETURN count(p) AS n"),
            ("GitLabDoc nodes",     "MATCH (g:GitLabDoc) RETURN count(g) AS n"),
            ("Project nodes",       "MATCH (p:Project) RETURN count(p) AS n"),
            ("User nodes",          "MATCH (u:User) RETURN count(u) AS n"),
            ("PART_OF rels",        "MATCH ()-[r:PART_OF]->() RETURN count(r) AS n"),
            ("CHILD_OF rels",       "MATCH ()-[r:CHILD_OF]->() RETURN count(r) AS n"),
            ("BLOCKS rels",         "MATCH ()-[r:BLOCKS]->() RETURN count(r) AS n"),
            ("RELATES_TO rels",     "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS n"),
            ("IS_TESTED_BY rels",   "MATCH ()-[r:IS_TESTED_BY]->() RETURN count(r) AS n"),
            ("CAUSES rels",         "MATCH ()-[r:CAUSES]->() RETURN count(r) AS n"),
        ]
        with self.driver.session(database=NEO4J_DATABASE) as s:
            for label, q in qs:
                try:
                    n = s.run(q).single()["n"]
                    log(f"  {label:<28} {n:>10,}")
                except Exception:
                    pass


def main():
    if not NEO4J_OK:
        print("pip install neo4j")
        return

    log("=" * 60)
    log("Neo4j Import — knowledge_base project")
    log("=" * 60)
    log(f"  Source mode:      {IMPORT_SOURCE.upper()}")
    if IMPORT_SOURCE == "jsonl":
        log(f"  Jira JSONL:       {JIRA_JSONL}")
        log(f"  Confluence JSONL: {CONF_JSONL}")
    else:
        log(f"  MongoDB:          {MONGO_URI}  db={MONGO_DB_NAME}")
    log(f"  Import GitLab:    {IMPORT_GITLAB}")
    log(f"  Embedding dim:    {EMBEDDING_DIM}")
    log(f"  Batch size:       {BATCH_SIZE}")
    log("")

    imp = Importer()
    try:
        imp.setup_schema()
        imp.import_jira()
        imp.import_confluence()
        if IMPORT_GITLAB:
            imp.import_gitlab()
        imp.stats()
    finally:
        imp.close()

    log("")
    log("=" * 60)
    log("✓ Import complete")
    log("=" * 60)
    log("  Next:")
    log("  python -m rag.neo4j_search hybrid 'your query'")
    log("  python -m rag.neo4j_rag chat")


if __name__ == "__main__":
    main()

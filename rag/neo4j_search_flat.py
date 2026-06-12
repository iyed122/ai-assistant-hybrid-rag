#!/usr/bin/env python3
"""
neo4j_search.py  —  replaces qdrant_search.py
==============================================
Three search modes:
  vector  — pure ANN cosine search on chunk_embedding index
  graph   — BLOCKS / CHILD_OF / RELATES_TO traversal from vector hits
  hybrid  — vector + graph expansion + keyword re-rank  (recommended)

CLI
───
  python -m rag.neo4j_search vector  'your query'
  python -m rag.neo4j_search graph   'your query'
  python -m rag.neo4j_search hybrid  'your query'
  python -m rag.neo4j_search stats
  python -m rag.neo4j_search filter  'your query' --source jira --project MYPROJ

.env keys
─────────
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
  EMBEDDING_MODEL   Snowflake/snowflake-arctic-embed-m
  SEARCH_TOP_K      10
"""

import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from neo4j import GraphDatabase
    NEO4J_OK = True
except ImportError:
    NEO4J_OK = False
    print("pip install neo4j")

try:
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except ImportError:
    ST_OK = False
    print("pip install sentence-transformers")

# ── Config ─────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",       "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",      "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD",  "your_password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE",  "neo4j")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL","Snowflake/snowflake-arctic-embed-m")
TOP_K          = int(os.getenv("SEARCH_TOP_K","10"))
ARCTIC_PREFIX  = "Represent this sentence for searching relevant passages: "


def log(msg: str, level: str = "INFO"):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{level}] {msg}", flush=True)


class Neo4jSearch:

    def __init__(self):
        if not NEO4J_OK:
            raise ImportError("pip install neo4j")
        if not ST_OK:
            raise ImportError("pip install sentence-transformers")

        log(f"Connecting to Neo4j at {NEO4J_URI} …")
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            notifications_min_severity="WARNING",
            notifications_disabled_categories=["UNRECOGNIZED"],
        )
        self.driver.verify_connectivity()
        log("✓ Connected")

        log(f"Loading embedding model: {EMBEDDING_MODEL} …")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.model.max_seq_length = 512
        log("✓ Model loaded")

    def close(self):
        self.driver.close()

    def embed(self, query: str) -> List[float]:
        return self.model.encode(
            ARCTIC_PREFIX + query,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).tolist()

    # ── Vector search ──────────────────────────────────────────────────────────
    def vector_search(
        self,
        query: str,
        top_k: int = TOP_K,
        source: Optional[str] = None,
        source_filter: Optional[str] = None,
        project: Optional[str] = None,
        project_filter: Optional[str] = None,
    ) -> List[Dict]:
        source  = source  or source_filter
        project = project or project_filter
        qvec = self.embed(query)

        # Build optional WHERE
        filters = []
        if source:
            filters.append("c.source = $source")
        if project:
            filters.append("c.project_name = $project")
        where = ("WHERE " + " AND ".join(filters)) if filters else ""

        cypher = f"""
        CALL db.index.vector.queryNodes('chunk_embedding', $k, $vec)
        YIELD node AS c, score
        {where}
        RETURN c.chunk_id       AS chunk_id,
               c.text           AS text,
               c.source         AS source,
               c.source_type    AS source_type,
               c.title          AS title,
               c.url            AS url,
               c.project_name   AS project_name,
               c.status         AS status,
               score
        ORDER BY score DESC
        """
        with self.driver.session(database=NEO4J_DATABASE) as s:
            results = list(s.run(cypher, vec=qvec, k=top_k * 2,
                                 source=source or "", project=project or ""))

        hits = []
        for r in results[:top_k]:
            hits.append(dict(r))
        return hits

    # ── Graph expansion ────────────────────────────────────────────────────────
    def graph_expand(self, issue_keys: List[str]) -> List[Dict]:
        """Follow all 20 typed relationship edges from a set of Jira issue keys.
        Uses single-hop [r:...] so type(r) returns a string, not a list.
        Variable-length paths (*1..2) make r a List<Relationship> which breaks type().
        """
        if not issue_keys:
            return []
        cypher = """
        UNWIND $keys AS k
        MATCH (t:JiraTicket {issue_key: k})
        MATCH (t)-[r:BLOCKS|BLOCKED_BY|RELATES_TO|DUPLICATES|DUPLICATED_BY|
                   CHILD_OF|IS_TESTED_BY|TESTS|CAUSES|CAUSED_BY|
                   CLONES|CLONED_BY|IS_AUTOMATED_BY|AUTOMATES|
                   SPLIT_FROM|SPLIT_TO|IMPLEMENTS|IMPLEMENTED_BY|
                   REVIEWED_BY]->(related:JiraTicket)
        OPTIONAL MATCH (chunk:Chunk)-[:PART_OF]->(related)
        RETURN related.issue_key  AS issue_key,
               related.title      AS title,
               related.status     AS status,
               related.priority   AS priority,
               related.url        AS url,
               type(r)            AS rel_type,
               chunk.text         AS text,
               chunk.chunk_id     AS chunk_id,
               0.6                AS score
        LIMIT 60
        """
        with self.driver.session(database=NEO4J_DATABASE) as s:
            results = list(s.run(cypher, keys=issue_keys))
        return [dict(r) for r in results if r["chunk_id"]]

    # ── Hybrid search ──────────────────────────────────────────────────────────
    def hybrid_search(
        self,
        query: str,
        top_k: int = TOP_K,
        source: Optional[str] = None,
        source_filter: Optional[str] = None,   # alias — some callers use this name
        project: Optional[str] = None,
        project_filter: Optional[str] = None,  # alias for symmetry
    ) -> List[Dict]:
        """
        Vector → graph expansion → keyword re-rank.
        Accepts both source/source_filter and project/project_filter for backwards compat.
        """
        # Resolve aliases
        source  = source  or source_filter
        project = project or project_filter
        # Step 1: vector hits
        vec_hits = self.vector_search(query, top_k=top_k, source=source, project=project)

        # Step 2: extract Jira issue keys for graph hop — strip _part_N suffix
        _part_re = __import__('re').compile(r'_part_\d+$')
        issue_keys = []
        for h in vec_hits:
            cid = _part_re.sub('', h.get("chunk_id", ""))
            if "jira_issue_" in cid:
                issue_keys.append(cid.split("jira_issue_")[-1])

        graph_hits = self.graph_expand(issue_keys)

        # Step 3: merge + keyword re-rank
        all_hits = vec_hits + [h for h in graph_hits if h not in vec_hits]
        query_terms = set(query.lower().split())

        def score(hit):
            base = float(hit.get("score", 0.0))
            text = (hit.get("text") or "").lower()
            title = (hit.get("title") or "").lower()
            kw = (
                sum(1 for t in query_terms if t in text) +
                sum(2 for t in query_terms if t in title)
            ) / max(len(query_terms) * 3, 1)
            return 0.7 * base + 0.3 * kw

        all_hits.sort(key=score, reverse=True)

        # Tag graph-expanded results
        vec_ids = {h["chunk_id"] for h in vec_hits}
        for h in all_hits:
            h["from_graph"] = h["chunk_id"] not in vec_ids

        return all_hits[:top_k]

    # ── Stats ──────────────────────────────────────────────────────────────────
    def stats(self):
        log("=" * 60)
        log("Neo4j Collection Statistics")
        log("=" * 60)
        qs = [
            ("Total Chunks",         "MATCH (c:Chunk) RETURN count(c) AS n"),
            ("  jira",               "MATCH (c:Chunk {source:'jira'}) RETURN count(c) AS n"),
            ("  confluence",         "MATCH (c:Chunk {source:'confluence'}) RETURN count(c) AS n"),
            ("  gitlab",             "MATCH (c:Chunk {source:'gitlab'}) RETURN count(c) AS n"),
            ("JiraTicket nodes",     "MATCH (t:JiraTicket) RETURN count(t) AS n"),
            ("ConfluencePage nodes", "MATCH (p:ConfluencePage) RETURN count(p) AS n"),
            ("PART_OF rels",         "MATCH ()-[r:PART_OF]->() RETURN count(r) AS n"),
            ("CHILD_OF rels",        "MATCH ()-[r:CHILD_OF]->() RETURN count(r) AS n"),
            ("BLOCKS rels",          "MATCH ()-[r:BLOCKS]->() RETURN count(r) AS n"),
            ("RELATES_TO rels",      "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS n"),
        ]
        with self.driver.session(database=NEO4J_DATABASE) as s:
            for label, q in qs:
                try:
                    n = s.run(q).single()["n"]
                    log(f"  {label:<24} {n:>10,}")
                except Exception:
                    pass


# ── Display helpers ────────────────────────────────────────────────────────────

def _display(results: List[Dict], mode: str):
    print(f"\n{'='*60}")
    print(f"Search mode: {mode.upper()}  |  {len(results)} results")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        score = float(r.get("score", 0))
        tag   = " [GRAPH]" if r.get("from_graph") else ""
        src   = (r.get("source") or "").upper()
        if score > 0.7:
            rel = "🟢 High"
        elif score > 0.45:
            rel = "🟡 Medium"
        else:
            rel = "🔴 Low"
        print(f"\n{i}. {rel} (score {score:.4f}){tag}")
        print(f"   [{src}] {r.get('source_type','')}  |  {r.get('title','')[:70]}")
        print(f"   {(r.get('text') or '')[:200]} …")
        print(f"   URL: {r.get('url','N/A')}")
        print(f"   Project: {r.get('project_name','N/A')}  |  Status: {r.get('status','')}")
    print(f"\n{'='*60}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    if not NEO4J_OK or not ST_OK:
        log("Missing dependencies — pip install neo4j sentence-transformers", "ERROR")
        return

    if len(sys.argv) < 2:
        print("\nNeo4j Semantic Search")
        print("=" * 60)
        print("Usage:")
        print("  python -m rag.neo4j_search vector  'your query'")
        print("  python -m rag.neo4j_search graph   'your query'")
        print("  python -m rag.neo4j_search hybrid  'your query'")
        print("  python -m rag.neo4j_search filter  'query' --source jira")
        print("  python -m rag.neo4j_search stats")
        return

    cmd = sys.argv[1]
    searcher = Neo4jSearch()

    try:
        if cmd == "stats":
            searcher.stats()

        elif cmd in ("vector", "hybrid", "graph"):
            if len(sys.argv) < 3:
                print(f"Usage: python -m rag.neo4j_search {cmd} 'your query'")
                return
            query = " ".join(sys.argv[2:])
            log(f"Query: {query!r}")
            if cmd == "vector":
                results = searcher.vector_search(query)
            elif cmd == "graph":
                results = searcher.vector_search(query, top_k=5)
                keys = [
                    h["chunk_id"].split("_chunk_")[0].replace("jira_issue_", "")
                    for h in results if "jira_issue_" in h.get("chunk_id", "")
                ]
                results = searcher.graph_expand(keys)
            else:
                results = searcher.hybrid_search(query)
            _display(results, cmd)

        elif cmd == "filter":
            if len(sys.argv) < 3:
                print("Usage: python -m rag.neo4j_search filter 'query' [--source X] [--project Y]")
                return
            query = sys.argv[2]
            source = project = None
            i = 3
            while i < len(sys.argv):
                if sys.argv[i] == "--source" and i + 1 < len(sys.argv):
                    source = sys.argv[i + 1]; i += 2
                elif sys.argv[i] == "--project" and i + 1 < len(sys.argv):
                    project = sys.argv[i + 1]; i += 2
                else:
                    i += 1
            results = searcher.hybrid_search(query, source=source, project=project)
            _display(results, "filter")

        else:
            print(f"Unknown command: {cmd}")

    finally:
        searcher.close()


if __name__ == "__main__":
    main()
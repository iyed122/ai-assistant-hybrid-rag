#!/usr/bin/env python3
"""
neo4j_search_test.py  —  OPTIMIZED retrieval engine  (A/B test variant)
=======================================================================
Drop-in replacement for neo4j_search.py. Same public class `Neo4jSearch`
and same `hybrid_search(...)` / `vector_search(...)` signatures, so
rag_generator_test.py (and intent_agent.py) can use it unchanged.

WHAT CHANGED vs neo4j_search.py  (see DIAGNOSIS.md for the full write-up)
────────────────────────────────────────────────────────────────────────
1. CROSS-ENCODER RERANK (required).  The old "0.7*cosine + 0.3*keyword"
   lexical blend is replaced by a real cross-encoder reranker applied to the
   merged vector+graph candidate pool. This is the single biggest precision
   win. Model is configurable via RERANKER_MODEL (default a small, fast
   ms-marco MiniLM). Falls back to an improved lexical scorer only if the
   model cannot load.

2. ENRICHED VECTOR SEEDS (quality + latency).  vector_search now does an
   OPTIONAL MATCH from each chunk to its parent JiraTicket / ConfluencePage
   and returns issue_key, priority, assignee, ticket_status, page_id directly.
   Consequences:
     • Graph seed extraction is now EXACT (uses jt.issue_key / cp.page_id)
       instead of regex-parsing parent_document_id, which the old code itself
       flagged as frequently NULL. Fewer missed expansions.
     • The LLM finally sees assignee / priority / issue_key for *vector* hits,
       not only graph-expanded ones (old pipeline returned assignee = "" always).

3. FILTER PUSH-DOWN (latency).  The project filter is applied inside the
   vector query (Cypher WHERE) instead of over-fetching top_k*5 in
   rag_generator and graph-expanding seeds that get thrown away afterward.

4. DISTANCE-DECAYED GRAPH SCORING.  Graph candidates carry a decayed prior
   (closer hops > far hops) used for pool selection / tie-breaks, instead of
   a flat 0.6 constant.

5. PARENT-DOC DIVERSITY CAP (quality).  At most PARENT_DOC_CAP chunks from the
   same parent document survive into the rerank pool, so one chatty ticket
   can't crowd out everything else. Also shrinks the rerank batch (latency).

6. RERANK-DRIVEN GRAPH QUOTA.  Graph results are still guaranteed a few slots
   (the whole point of hybrid retrieval), but ordering inside the final set is
   by reranker relevance, and the quota is gateable by GRAPH_MIN_RERANK.

7. PER-STAGE TIMINGS.  After each hybrid_search, self.last_timings holds
   {embed_ms, vector_ms, graph_ms, rerank_ms, total_ms} for the A/B harness.

.env keys (new ones marked ★)
─────────────────────────────
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
  EMBEDDING_MODEL     Snowflake/snowflake-arctic-embed-m
  SEARCH_TOP_K        10
  GRAPH_HOPS          2
  MAX_GRAPH_EXPAND    20
★ RERANKER_MODEL      cross-encoder/ms-marco-MiniLM-L-6-v2
★ RERANK_POOL         50     max candidates sent to the cross-encoder
★ VEC_CANDIDATES      40     vector pool size fetched before rerank
★ PARENT_DOC_CAP      3      max chunks kept per parent document
★ GRAPH_MIN_SLOTS     0      0 = auto (top_k // 4); reserve N graph slots
★ GRAPH_MIN_RERANK    0.0    drop reserved graph hits below this rerank score
★ RERANK_TEXT_CHARS   800    passage length sent to reranker
"""

import os
import sys
import math
import time
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

# CrossEncoder ships with sentence-transformers; import separately so a missing
# class (very old versions) degrades gracefully instead of killing the module.
try:
    from sentence_transformers import CrossEncoder
    CE_OK = True
except ImportError:
    CE_OK = False

# ── Config ──────────────────────────────────────────────────────────────────────
NEO4J_URI        = os.getenv("NEO4J_URI",        "bolt://localhost:7687")
NEO4J_USER       = os.getenv("NEO4J_USER",       "neo4j")
NEO4J_PASSWORD   = os.getenv("NEO4J_PASSWORD",   "your_password")
NEO4J_DATABASE   = os.getenv("NEO4J_DATABASE",   "neo4j")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",  "Snowflake/snowflake-arctic-embed-m")
TOP_K            = int(os.getenv("SEARCH_TOP_K",    "10"))
GRAPH_HOPS       = int(os.getenv("GRAPH_HOPS",      "2"))
MAX_GRAPH_EXPAND = int(os.getenv("MAX_GRAPH_EXPAND", "20"))
ARCTIC_PREFIX    = "Represent this sentence for searching relevant passages: "

# Reranking / pooling
RERANKER_MODEL   = os.getenv("RERANKER_MODEL",   "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_POOL      = int(os.getenv("RERANK_POOL",     "50"))
VEC_CANDIDATES   = int(os.getenv("VEC_CANDIDATES",  "40"))
PARENT_DOC_CAP   = int(os.getenv("PARENT_DOC_CAP",  "3"))
GRAPH_MIN_SLOTS  = int(os.getenv("GRAPH_MIN_SLOTS", "0"))    # 0 → auto (top_k // 4)
GRAPH_MIN_RERANK = float(os.getenv("GRAPH_MIN_RERANK", "0.0"))
RERANK_TEXT_CHARS = int(os.getenv("RERANK_TEXT_CHARS", "800"))
GRAPH_DECAY      = float(os.getenv("GRAPH_DECAY", "0.85"))   # per-hop prior decay

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "what", "which", "who", "how", "does", "do", "this", "that", "with", "about",
    "be", "by", "at", "as", "it", "its", "from", "was", "were", "has", "have",
}


# ── Cypher templates ────────────────────────────────────────────────────────────
# Neo4j does NOT allow parameters inside relationship length specifiers (*1..$hops);
# the hop count must be a literal int baked into the string.

def _jira_expand_cypher(hops: int) -> str:
    return f"""
UNWIND $keys AS k
MATCH (seed:JiraTicket {{issue_key: k}})
MATCH path = (seed)-[:BLOCKS|BLOCKED_BY|RELATES_TO|DUPLICATES|DUPLICATED_BY|
              CHILD_OF|IS_TESTED_BY|TESTS|CAUSES|CAUSED_BY|CLONES|CLONED_BY|
              IS_AUTOMATED_BY|AUTOMATES|SPLIT_FROM|SPLIT_TO|IMPLEMENTS|
              IMPLEMENTED_BY|REVIEWED_BY*1..{hops}]-(neighbor:JiraTicket)
  WHERE neighbor.issue_key <> k
MATCH (chunk:Chunk)-[:PART_OF]->(neighbor)
WITH neighbor,
     collect(chunk)[0]                        AS best_chunk,
     min(length(path))                        AS dist,
     [r IN relationships(path) | type(r)][0]  AS rel_type
ORDER BY dist ASC
RETURN
  best_chunk.chunk_id           AS chunk_id,
  best_chunk.text               AS text,
  best_chunk.title              AS title,
  best_chunk.source             AS source,
  best_chunk.source_type        AS source_type,
  best_chunk.project_name       AS project_name,
  best_chunk.url                AS url,
  best_chunk.status             AS status,
  best_chunk.created_at         AS created_at,
  best_chunk.parent_document_id AS parent_document_id,
  neighbor.issue_key            AS issue_key,
  neighbor.priority             AS priority,
  neighbor.assignee             AS assignee,
  neighbor.status               AS neighbor_status,
  rel_type                      AS relationship,
  dist                          AS graph_distance
LIMIT $max_expand
"""


def _conf_expand_cypher(hops: int) -> str:
    return f"""
UNWIND $page_ids AS pid
MATCH (seed:ConfluencePage {{page_id: pid}})
MATCH path = (seed)-[:CHILD_OF*1..{hops}]-(neighbor:ConfluencePage)
  WHERE neighbor.page_id <> pid
MATCH (chunk:Chunk)-[:PART_OF]->(neighbor)
WITH neighbor, chunk,
     min(length(path)) AS dist
ORDER BY dist ASC, chunk.chunk_index ASC
WITH neighbor, collect(chunk)[0] AS best_chunk, min(dist) AS dist
RETURN
  best_chunk.chunk_id           AS chunk_id,
  best_chunk.text               AS text,
  best_chunk.title              AS title,
  best_chunk.source             AS source,
  best_chunk.source_type        AS source_type,
  best_chunk.project_name       AS project_name,
  best_chunk.url                AS url,
  best_chunk.status             AS status,
  best_chunk.created_at         AS created_at,
  best_chunk.parent_document_id AS parent_document_id,
  null                          AS issue_key,
  null                          AS priority,
  null                          AS assignee,
  null                          AS neighbor_status,
  'CHILD_OF'                    AS relationship,
  dist                          AS graph_distance
LIMIT $max_expand
"""


def log(msg: str, level: str = "INFO"):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{level}] {msg}", flush=True)


# ── Reranker wrapper ─────────────────────────────────────────────────────────────

class _Reranker:
    """Thin cross-encoder wrapper. Scores are sigmoid-squashed to (0, 1)."""

    def __init__(self, model_name: str):
        if not CE_OK:
            raise ImportError("CrossEncoder unavailable — upgrade sentence-transformers")
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=512)

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        ex = math.exp(x)
        return ex / (1.0 + ex)

    def score(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        raw = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
        return [self._sigmoid(float(s)) for s in raw]


class Neo4jSearch:

    def __init__(self, enable_rerank: bool = True):
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
        log("✓ Embedding model loaded")

        self.reranker: Optional[_Reranker] = None
        if enable_rerank:
            try:
                log(f"Loading reranker: {RERANKER_MODEL} …")
                self.reranker = _Reranker(RERANKER_MODEL)
                log("✓ Cross-encoder reranker ready")
            except Exception as exc:  # pragma: no cover - depends on local env
                log(f"Reranker unavailable ({exc}); falling back to lexical re-rank", "WARN")
                self.reranker = None

        self.last_timings: Dict[str, float] = {}

    def close(self):
        self.driver.close()

    def _run(self, cypher: str, params: dict = None) -> List[Dict]:
        with self.driver.session(database=NEO4J_DATABASE) as s:
            return [dict(r) for r in s.run(cypher, params or {})]

    def embed(self, query: str) -> List[float]:
        return self.model.encode(
            ARCTIC_PREFIX + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

    # ── Vector search (enriched + filter push-down) ─────────────────────────────
    def vector_search(
        self,
        query: str,
        top_k: int = TOP_K,
        source: Optional[str] = None,
        source_filter: Optional[str] = None,
        project: Optional[str] = None,
        project_filter: Optional[str] = None,
        _qvec: Optional[List[float]] = None,
    ) -> List[Dict]:
        source  = source  or source_filter
        project = project or project_filter
        qvec    = _qvec if _qvec is not None else self.embed(query)

        filters = []
        if source:
            filters.append("c.source = $source")
        if project:
            filters.append("toLower(c.project_name) CONTAINS toLower($project)")
        where = ("WHERE " + " AND ".join(filters)) if filters else ""

        # Over-fetch from the ANN index, then (optionally) filter, then trim.
        cypher = f"""
        CALL db.index.vector.queryNodes('chunk_embedding', $k, $vec)
        YIELD node AS c, score
        WITH c, score
        {where}
        OPTIONAL MATCH (c)-[:PART_OF]->(jt:JiraTicket)
        OPTIONAL MATCH (c)-[:PART_OF]->(cp:ConfluencePage)
        RETURN c.chunk_id           AS chunk_id,
               c.text               AS text,
               c.source             AS source,
               c.source_type        AS source_type,
               c.title              AS title,
               c.url                AS url,
               c.project_name       AS project_name,
               c.status             AS status,
               c.parent_document_id AS parent_document_id,
               jt.issue_key         AS issue_key,
               jt.priority          AS priority,
               jt.assignee          AS assignee,
               jt.status            AS ticket_status,
               cp.page_id           AS page_id,
               score
        ORDER BY score DESC
        """
        # Pull extra from the index so post-filtering still yields ~top_k.
        k = top_k * 2 if (source or project) else top_k
        results = self._run(cypher, {
            "vec":     qvec,
            "k":       k,
            "source":  source or "",
            "project": project or "",
        })
        return results[:top_k]

    # ── Graph expansion ─────────────────────────────────────────────────────────
    def jira_expand(self, issue_keys, hops=GRAPH_HOPS, max_expand=MAX_GRAPH_EXPAND):
        if not issue_keys:
            return []
        return self._run(_jira_expand_cypher(hops), {
            "keys":       list(set(issue_keys)),
            "max_expand": max_expand,
        })

    def conf_expand(self, page_ids, hops=GRAPH_HOPS, max_expand=MAX_GRAPH_EXPAND):
        if not page_ids:
            return []
        return self._run(_conf_expand_cypher(hops), {
            "page_ids":   list(set(page_ids)),
            "max_expand": max_expand,
        })

    def graph_expand(self, issue_keys):
        """Backwards-compatible alias."""
        return self.jira_expand(issue_keys)

    # ── Helpers ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _parent_key(hit: Dict) -> str:
        """Stable identity for a chunk's parent document (for diversity cap)."""
        return (
            hit.get("issue_key")
            or hit.get("page_id")
            or hit.get("parent_document_id")
            or hit.get("chunk_id")
            or ""
        )

    @staticmethod
    def _passage(hit: Dict) -> str:
        title = (hit.get("title") or "").strip()
        text  = (hit.get("text") or "")[:RERANK_TEXT_CHARS]
        return f"{title}\n{text}" if title else text

    def _dedup_by_parent(self, hits: List[Dict], cap: int) -> List[Dict]:
        seen_chunk = set()
        per_parent: Dict[str, int] = {}
        out = []
        for h in hits:
            cid = h.get("chunk_id")
            if cid in seen_chunk:
                continue
            pk = self._parent_key(h)
            if pk and per_parent.get(pk, 0) >= cap:
                continue
            seen_chunk.add(cid)
            per_parent[pk] = per_parent.get(pk, 0) + 1
            out.append(h)
        return out

    def _lexical_score(self, query: str, hit: Dict) -> float:
        """Fallback only (reranker unavailable): stopword-filtered, word-boundary."""
        terms = {t for t in query.lower().split() if t not in _STOPWORDS and len(t) > 2}
        if not terms:
            return float(hit.get("score", 0.0))
        text  = set((hit.get("text") or "").lower().split())
        title = set((hit.get("title") or "").lower().split())
        kw = (sum(1 for t in terms if t in text)
              + 2 * sum(1 for t in terms if t in title)) / (len(terms) * 3)
        return 0.7 * float(hit.get("score", 0.0)) + 0.3 * kw

    # ── Hybrid search ────────────────────────────────────────────────────────────
    def hybrid_search(
        self,
        query: str,
        top_k: int = TOP_K,
        source: Optional[str] = None,
        source_filter: Optional[str] = None,
        project: Optional[str] = None,
        project_filter: Optional[str] = None,
        hops: int = GRAPH_HOPS,
        max_expand: int = MAX_GRAPH_EXPAND,
        rerank: bool = True,
    ) -> List[Dict]:
        source  = source  or source_filter
        project = project or project_filter
        t = {}
        t_total = time.perf_counter()

        # 1. Embed once, reuse for the vector query.
        t0 = time.perf_counter()
        qvec = self.embed(query)
        t["embed_ms"] = (time.perf_counter() - t0) * 1000

        # 2. Vector candidate pool (enriched, filtered at the source).
        t0 = time.perf_counter()
        n_cand = max(top_k, VEC_CANDIDATES)
        vec_hits = self.vector_search(
            query, top_k=n_cand, source=source, project=project, _qvec=qvec,
        )
        for h in vec_hits:
            h["from_graph"] = False
        t["vector_ms"] = (time.perf_counter() - t0) * 1000

        # 3. Graph expansion from EXACT seed keys (no regex parsing).
        t0 = time.perf_counter()
        jira_keys  = [h["issue_key"] for h in vec_hits if h.get("issue_key")]
        conf_pages = [h["page_id"]   for h in vec_hits if h.get("page_id")]
        log(f"  Graph seeds — jira: {len(set(jira_keys))}, confluence: "
            f"{len(set(conf_pages))}  (hops={hops})")

        graph_hits: List[Dict] = []
        if jira_keys:
            graph_hits += self.jira_expand(jira_keys, hops=hops, max_expand=max_expand)
        if conf_pages:
            graph_hits += self.conf_expand(conf_pages, hops=hops, max_expand=max_expand)
        for h in graph_hits:
            h["from_graph"] = True
            dist = h.get("graph_distance") or 1
            # decayed prior; only used for pool ordering / tie-breaks
            h["score"] = round(0.6 * (GRAPH_DECAY ** (int(dist) - 1)), 4)
        t["graph_ms"] = (time.perf_counter() - t0) * 1000
        log(f"  Graph hits — {len(graph_hits)}")

        # 4. Merge, dedup by chunk_id, diversity-cap per parent document.
        vec_ids = {h["chunk_id"] for h in vec_hits}
        merged = list(vec_hits)
        for h in graph_hits:
            if h.get("chunk_id") not in vec_ids:
                merged.append(h)
        merged = self._dedup_by_parent(merged, cap=PARENT_DOC_CAP)

        # 5. Rerank the pool (cross-encoder, or lexical fallback).
        t0 = time.perf_counter()
        pool = merged[:RERANK_POOL]
        if rerank and self.reranker is not None:
            scores = self.reranker.score(query, [self._passage(h) for h in pool])
            for h, s in zip(pool, scores):
                h["rerank_score"] = s
            method = "cross-encoder"
        else:
            for h in pool:
                h["rerank_score"] = self._lexical_score(query, h)
            method = "lexical-fallback"
        # Anything beyond the pool keeps a demoted prior so it never outranks reranked hits.
        for h in merged[RERANK_POOL:]:
            h["rerank_score"] = float(h.get("score", 0.0)) * 0.10
        t["rerank_ms"] = (time.perf_counter() - t0) * 1000

        merged.sort(key=lambda h: h.get("rerank_score", 0.0), reverse=True)

        # 6. Final selection — reranker order, with a gated graph quota so the
        #    graph signal survives even when vector hits dominate raw relevance.
        graph_quota = GRAPH_MIN_SLOTS if GRAPH_MIN_SLOTS > 0 else max(1, top_k // 4)
        vec_pool   = [h for h in merged if not h["from_graph"]]
        graph_pool = [h for h in merged if h["from_graph"]
                      and h.get("rerank_score", 0.0) >= GRAPH_MIN_RERANK]

        if graph_pool:
            n_graph = min(len(graph_pool), graph_quota, top_k)
            n_vec   = top_k - n_graph
            final   = vec_pool[:n_vec] + graph_pool[:n_graph]
            final.sort(key=lambda h: h.get("rerank_score", 0.0), reverse=True)
            log(f"  Final mix — {n_vec} vector + {n_graph} graph "
                f"(quota={graph_quota}, rerank={method})")
        else:
            final = merged[:top_k]
            log(f"  Final — {len(final)} hits (rerank={method})")

        # expose final relevance as `score` too, so existing display/normalise code
        # that reads `score` shows the reranked value.
        for h in final:
            h["score"] = float(h.get("rerank_score", h.get("score", 0.0)))

        t["total_ms"] = (time.perf_counter() - t_total) * 1000
        self.last_timings = t
        return final

    # ── Stats ────────────────────────────────────────────────────────────────────
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


# ── Display helpers ──────────────────────────────────────────────────────────────

def _display(results: List[Dict], mode: str, timings: Dict = None):
    print(f"\n{'='*60}")
    print(f"Search mode: {mode.upper()}  |  {len(results)} results")
    if timings:
        print("  timings(ms): " + "  ".join(f"{k}={v:.0f}" for k, v in timings.items()))
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        score = float(r.get("score", 0))
        tag   = f" [GRAPH +{r.get('graph_distance', 1)}hop]" if r.get("from_graph") else ""
        src   = (r.get("source") or "").upper()
        rel   = "🟢 High" if score > 0.7 else ("🟡 Medium" if score > 0.4 else "🔴 Low")
        print(f"\n{i}. {rel} (score {score:.4f}){tag}")
        print(f"   [{src}] {r.get('source_type','')}  |  {r.get('title','')[:70]}")
        print(f"   {(r.get('text') or '')[:200]} …")
        print(f"   URL: {r.get('url','N/A')}")
        print(f"   Project: {r.get('project_name','N/A')}  |  Status: {r.get('status','')}")
        if r.get("relationship"):
            print(f"   Relationship: {r['relationship']}")
    print(f"\n{'='*60}")


# ── CLI ──────────────────────────────────────────────────────────────────────────

def main():
    if not NEO4J_OK or not ST_OK:
        log("Missing dependencies — pip install neo4j sentence-transformers", "ERROR")
        return

    if len(sys.argv) < 2:
        print("\nNeo4j Semantic Search (OPTIMIZED test build)")
        print("=" * 60)
        print("  python -m rag.neo4j_search_test vector  'your query'")
        print("  python -m rag.neo4j_search_test hybrid  'your query'")
        print("  python -m rag.neo4j_search_test filter  'query' --source jira --project X")
        print("  python -m rag.neo4j_search_test stats")
        return

    cmd      = sys.argv[1]
    searcher = Neo4jSearch()

    try:
        if cmd == "stats":
            searcher.stats()
        elif cmd in ("vector", "hybrid"):
            if len(sys.argv) < 3:
                print(f"Usage: python -m rag.neo4j_search_test {cmd} 'your query'")
                return
            query = " ".join(sys.argv[2:])
            log(f"Query: {query!r}")
            if cmd == "vector":
                results = searcher.vector_search(query)
                _display(results, cmd)
            else:
                results = searcher.hybrid_search(query)
                _display(results, cmd, searcher.last_timings)
        elif cmd == "filter":
            if len(sys.argv) < 3:
                print("Usage: python -m rag.neo4j_search_test filter 'query' [--source X] [--project Y] [--hops N]")
                return
            query  = sys.argv[2]
            source = project = None
            hops   = GRAPH_HOPS
            i      = 3
            while i < len(sys.argv):
                if   sys.argv[i] == "--source"  and i + 1 < len(sys.argv): source  = sys.argv[i+1]; i += 2
                elif sys.argv[i] == "--project" and i + 1 < len(sys.argv): project = sys.argv[i+1]; i += 2
                elif sys.argv[i] == "--hops"    and i + 1 < len(sys.argv): hops    = int(sys.argv[i+1]); i += 2
                else: i += 1
            results = searcher.hybrid_search(query, source=source, project=project, hops=hops)
            _display(results, "filter", searcher.last_timings)
        else:
            print(f"Unknown command: {cmd}")
    finally:
        searcher.close()


if __name__ == "__main__":
    main()

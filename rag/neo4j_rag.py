#!/usr/bin/env python3
"""
neo4j_rag.py
============
Replaces rag_generator.py for the Neo4j-backed pipeline.

What's different from the Qdrant RAG
──────────────────────────────────────
1. Graph-augmented retrieval — after vector search, the graph is traversed
   to pull related tickets, blocked items, parent epics, and Confluence
   pages linked to the seed results.

2. Structured context — the LLM prompt is built from rich graph context:
     [Ticket XY-123]  BLOCKS  [Ticket XY-456] (status: Open, priority: High)
   rather than just raw chunk text.

3. Ticket-aware routing — if the user mentions a specific ticket key
   (e.g. "PAY-456"), it is fetched directly from the graph, bypassing
   the embedding step for that part.

4. Source citation — each source includes the URL and graph relationship
   so the user can trace why a result was included.

Environment variables (.env)
──────────────────────────────
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
  EMBEDDING_MODEL    Snowflake/snowflake-arctic-embed-m
  OLLAMA_HOST        http://localhost:11434
  OLLAMA_MODEL       qwen3
  MAX_TOKENS         1024
  TEMPERATURE        0.7
  NUM_CTX            16384   (higher than qdrant version — graph context is richer)
  TOP_K              10
  GRAPH_HOPS         2
  MAX_GRAPH_EXPAND   20
"""

import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠  requests not installed.  pip install requests")

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠  neo4j not installed.  pip install neo4j")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("⚠  sentence-transformers not installed.")

# ── Configuration ──────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "Snowflake/snowflake-arctic-embed-m")
ARCTIC_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3")
MAX_TOKENS   = int(os.getenv("MAX_TOKENS",    "1024"))
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.2"))   # FIX: was 0.7 — too high for factual enterprise RAG
# Raised vs Qdrant version — graph context has structured metadata that's denser
NUM_CTX      = int(os.getenv("NUM_CTX",       "8000"))

TOP_K            = int(os.getenv("TOP_K",            "10"))
GRAPH_HOPS       = int(os.getenv("GRAPH_HOPS",       "2"))
MAX_GRAPH_EXPAND = int(os.getenv("MAX_GRAPH_EXPAND",  "20"))

# Regex to detect Jira issue keys in user queries (e.g. PAY-456, PROJ-1234)
ISSUE_KEY_PATTERN = re.compile(r'\b([A-Z][A-Z0-9]+-\d+)\b')


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


# ── Cypher ─────────────────────────────────────────────────────────────────────

VECTOR_SEARCH = """
CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding)
YIELD node AS chunk, score
RETURN
  chunk.chunk_id            AS chunk_id,
  chunk.text                AS text,
  chunk.title               AS title,
  chunk.source              AS source,
  chunk.source_type         AS source_type,
  chunk.project_name        AS project_name,
  chunk.url                 AS url,
  chunk.status              AS status,
  chunk.author              AS author,
  chunk.created_at          AS created_at,
  chunk.parent_document_id  AS parent_document_id,
  score
ORDER BY score DESC
"""

GRAPH_EXPAND = """
UNWIND $issue_keys AS ik
MATCH (seed:JiraTicket {issue_key: ik})
MATCH path = (seed)-[:BLOCKS|BLOCKED_BY|RELATES_TO|DUPLICATES|DUPLICATED_BY|
              CHILD_OF|IS_TESTED_BY|TESTS|CAUSES|CAUSED_BY|CLONES|CLONED_BY|
              IS_AUTOMATED_BY|AUTOMATES|SPLIT_FROM|SPLIT_TO|IMPLEMENTS|
              IMPLEMENTED_BY|REVIEWED_BY*1..$hops]-(neighbor:JiraTicket)
MATCH (nc:Chunk)-[:PART_OF]->(neighbor)
WITH nc, neighbor,
     [r IN relationships(path) | type(r)][0] AS rel_type,
     length(path) AS dist
ORDER BY dist ASC, nc.chunk_index ASC
WITH neighbor, collect(nc)[0] AS best_chunk, min(dist) AS dist,
     collect(DISTINCT rel_type)[0] AS rel_type
RETURN
  best_chunk.chunk_id         AS chunk_id,
  best_chunk.text             AS text,
  best_chunk.title            AS title,
  best_chunk.source           AS source,
  best_chunk.source_type      AS source_type,
  best_chunk.project_name     AS project_name,
  best_chunk.url              AS url,
  best_chunk.status           AS status,
  best_chunk.created_at       AS created_at,
  best_chunk.parent_document_id AS parent_document_id,
  neighbor.issue_key          AS issue_key,
  neighbor.priority           AS priority,
  neighbor.status             AS neighbor_status,
  rel_type                    AS relationship,
  dist                        AS graph_distance
LIMIT $max_expand
"""

DIRECT_TICKET_FETCH = """
MATCH (t:JiraTicket {issue_key: $issue_key})
OPTIONAL MATCH (tc:Chunk)-[:PART_OF]->(t)
OPTIONAL MATCH (t)-[:ASSIGNED_TO]->(u:User)
OPTIONAL MATCH (t)-[:FIX_IN]->(v:Version)
OPTIONAL MATCH (t)-[:HAS_COMPONENT]->(comp:Component)
OPTIONAL MATCH (t)-[:CHILD_OF]->(parent:JiraTicket)
OPTIONAL MATCH (t)-[link]->(linked:JiraTicket)
  WHERE type(link) IN [
    'BLOCKS','BLOCKED_BY','RELATES_TO','DUPLICATES','DUPLICATED_BY',
    'IS_TESTED_BY','TESTS','CAUSES','CAUSED_BY','CLONES','CLONED_BY',
    'IMPLEMENTS','IMPLEMENTED_BY','SPLIT_FROM','SPLIT_TO',
    'IS_AUTOMATED_BY','AUTOMATES','REVIEWED_BY'
  ]
RETURN
  t.issue_key    AS issue_key,
  t.issue_id     AS issue_id,
  t.title        AS title,
  t.status       AS status,
  t.priority     AS priority,
  t.issue_type   AS issue_type,
  t.url          AS url,
  u.name         AS assignee,
  collect(DISTINCT tc.text)[0..2] AS chunk_texts,
  collect(DISTINCT v.name)        AS fix_versions,
  collect(DISTINCT comp.name)     AS components,
  parent.issue_key                AS parent_key,
  collect(DISTINCT {
    rel: type(link),
    key: linked.issue_key,
    title: linked.title,
    status: linked.status,
    priority: linked.priority
  }) AS linked_tickets
"""

CONF_EXPAND = """
UNWIND $page_ids AS pid
MATCH (seed:ConfluencePage {page_id: pid})
MATCH (seed)-[:CHILD_OF*1..$hops]-(neighbor:ConfluencePage)
MATCH (nc:Chunk)-[:PART_OF]->(neighbor)
WITH nc, neighbor, length(shortestPath((seed)-[:CHILD_OF*]-(neighbor))) AS dist
ORDER BY dist ASC, nc.chunk_index ASC
WITH neighbor, collect(nc)[0] AS best_chunk, min(dist) AS dist
RETURN
  best_chunk.chunk_id        AS chunk_id,
  best_chunk.text            AS text,
  best_chunk.title           AS title,
  best_chunk.source          AS source,
  best_chunk.source_type     AS source_type,
  best_chunk.project_name    AS project_name,
  best_chunk.url             AS url,
  best_chunk.status          AS status,
  best_chunk.created_at      AS created_at,
  best_chunk.parent_document_id AS parent_document_id,
  dist AS graph_distance
LIMIT $max_expand
"""


# ── RAG system ─────────────────────────────────────────────────────────────────

class Neo4jRAG:

    def __init__(self):
        if not all([REQUESTS_AVAILABLE, NEO4J_AVAILABLE, ST_AVAILABLE]):
            raise ImportError("Missing dependencies — pip install neo4j sentence-transformers requests")

        log("=" * 60)
        log("Initializing Neo4j RAG System")
        log("=" * 60)

        log(f"Connecting to Neo4j at {NEO4J_URI} …")
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            notifications_min_severity="WARNING",
            notifications_disabled_categories=["UNRECOGNIZED"],
        )
        self.driver.verify_connectivity()
        log("✓ Neo4j connected")

        log(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.model.max_seq_length = 512
        log("✓ Embedding model loaded")

        log(f"Connecting to Ollama at {OLLAMA_HOST}")
        if not self._check_ollama():
            raise ConnectionError(f"Cannot reach Ollama at {OLLAMA_HOST} — run: ollama serve")
        log("✓ Ollama connected")

        global OLLAMA_MODEL
        resolved = self._resolve_model(OLLAMA_MODEL)
        if resolved != OLLAMA_MODEL:
            log(f"Model resolved: '{OLLAMA_MODEL}' → '{resolved}'")
            OLLAMA_MODEL = resolved

        if not self._check_model():
            available = self._list_models()
            raise ValueError(
                f"Model '{OLLAMA_MODEL}' not found in Ollama.\n"
                f"Available: {available}\n"
                f"Pull it: ollama pull {OLLAMA_MODEL}"
            )

        log(f"✓ Model '{OLLAMA_MODEL}' ready")
        log(f"  num_ctx={NUM_CTX}  max_tokens={MAX_TOKENS}  top_k={TOP_K}  hops={GRAPH_HOPS}")
        log("=" * 60)
        log("✓ Neo4j RAG Ready!")
        log("=" * 60)
        print()

    def close(self):
        self.driver.close()

    # ── Ollama helpers ─────────────────────────────────────────────────────────

    def _check_ollama(self) -> bool:
        try:
            return requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5).status_code == 200
        except Exception:
            return False

    def _check_model(self) -> bool:
        try:
            r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            return OLLAMA_MODEL in [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return False

    def _list_models(self) -> List[str]:
        try:
            r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def _resolve_model(self, requested: str) -> str:
        try:
            r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            available = [m["name"] for m in r.json().get("models", [])]
            if requested in available:
                return requested
            base = requested.split(":")[0]
            for name in available:
                if name.split(":")[0] == base:
                    return name
            return requested
        except Exception:
            return requested

    # ── Embedding ──────────────────────────────────────────────────────────────

    def embed_query(self, query: str) -> List[float]:
        emb = self.model.encode(
            ARCTIC_QUERY_PREFIX + query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.tolist()

    # ── Graph retrieval ────────────────────────────────────────────────────────

    def _run(self, cypher: str, params: dict = None) -> List[Dict]:
        with self.driver.session(database=NEO4J_DATABASE) as s:
            return [dict(r) for r in s.run(cypher, params or {})]

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        filters: Dict[str, Any] = None,
        hops: int = GRAPH_HOPS,
    ) -> List[Dict]:
        """
        Full graph-augmented retrieval.

        1. Detect explicit ticket keys in query → direct graph fetch
        2. Vector search → top_k semantically similar chunks
        3. Graph expansion → follow Jira links and Confluence ancestry
        4. Merge + deduplicate
        """
        explicit_keys = ISSUE_KEY_PATTERN.findall(query)
        direct_results = []

        # Step 1: direct ticket lookups for any mentioned keys
        for key in explicit_keys:
            rows = self._run(DIRECT_TICKET_FETCH, {"issue_key": key})
            if rows:
                r = rows[0]
                for txt in (r.get("chunk_texts") or []):
                    direct_results.append({
                        "chunk_id":     f"direct_{key}",
                        "text":         txt,
                        "title":        r.get("title", ""),
                        "source":       "jira",
                        "source_type":  r.get("issue_type", "issue"),
                        "project_name": "",
                        "url":          r.get("url", ""),
                        "status":       r.get("status", ""),
                        "author":       r.get("assignee", ""),
                        "score":        1.0,
                        "issue_key":    key,
                        "issue_id":     r.get("issue_id", ""),
                        "from_direct":  True,
                        "graph_context": {
                            "priority":    r.get("priority"),
                            "fix_versions":r.get("fix_versions"),
                            "components":  r.get("components"),
                            "parent_key":  r.get("parent_key"),
                            "linked":      r.get("linked_tickets", [])[:5],
                        },
                    })

        # Step 2: vector search
        embedding   = self.embed_query(query)
        vector_hits = self._run(VECTOR_SEARCH, {"embedding": embedding, "k": top_k})

        # Step 3: graph expansion from vector results
        # Strip _part_N suffix from parent_document_id before extracting keys
        # e.g. "jira_issue_ML-1235_part_2" → "ML-1235"
        #      "confluence_page_4057_part_1" → "4057"
        _part_re = re.compile(r'_part_\d+$')
        jira_keys  = []
        conf_pages = []
        for r in vector_hits:
            pdid = _part_re.sub('', r.get("parent_document_id", ""))
            if r.get("source") == "jira" and "jira_issue_" in pdid:
                jira_keys.append(pdid.split("jira_issue_")[-1])
            elif r.get("source") == "confluence" and "confluence_page_" in pdid:
                conf_pages.append(pdid.split("confluence_page_")[-1])

        graph_hits = []
        if jira_keys:
            graph_hits += self._run(GRAPH_EXPAND, {
                "issue_keys": list(set(jira_keys)),
                "hops":       hops,
                "max_expand": MAX_GRAPH_EXPAND,
            })
        if conf_pages:
            graph_hits += self._run(CONF_EXPAND, {
                "page_ids":   list(set(conf_pages)),
                "hops":       hops,
                "max_expand": MAX_GRAPH_EXPAND,
            })

        for h in graph_hits:
            h["from_graph"] = True
            h.setdefault("score", 0.6)  # FIX: was 0.0 — graph hits ranked last vs any vector hit

        # Merge and deduplicate
        seen     = {r["chunk_id"] for r in direct_results + vector_hits}
        combined = direct_results + vector_hits
        for h in graph_hits:
            if h.get("chunk_id") not in seen:
                combined.append(h)
                seen.add(h["chunk_id"])

        return combined

    # ── Context builder ────────────────────────────────────────────────────────

    def _build_context(self, docs: List[Dict], max_chars_per_doc: int = 700) -> str:
        """
        Build a rich, structured context string for the LLM.
        Includes graph relationship context where available.
        """
        parts = []
        for i, doc in enumerate(docs, 1):
            src_tag   = f"[{doc.get('source','?').upper()} / {doc.get('source_type','')}]"
            title     = doc.get("title", "Untitled")
            status    = doc.get("status", "")
            url       = doc.get("url", "")
            text      = (doc.get("text") or "")[:max_chars_per_doc]

            header = f"--- Source {i}: {src_tag} {title}"
            if status:
                header += f"  [{status}]"
            if url:
                header += f"\n    URL: {url}"

            # For directly-fetched tickets, show structured graph context
            gctx = doc.get("graph_context")
            if gctx:
                if gctx.get("priority"):
                    header += f"\n    Priority: {gctx['priority']}"
                if gctx.get("fix_versions"):
                    header += f"\n    Fix Versions: {', '.join(gctx['fix_versions'])}"
                if gctx.get("components"):
                    header += f"\n    Components: {', '.join(gctx['components'])}"
                if gctx.get("parent_key"):
                    header += f"\n    Parent Ticket: {gctx['parent_key']}"
                linked = gctx.get("linked", [])
                if linked:
                    link_strs = [
                        f"{lnk.get('rel','')} {lnk.get('key','')} "
                        f"({lnk.get('title','')[:40]}) [{lnk.get('status','')}]"
                        for lnk in linked if lnk.get("key")
                    ]
                    if link_strs:
                        header += f"\n    Links:\n      " + "\n      ".join(link_strs)

            # For graph-expanded items, annotate the relationship
            if doc.get("from_graph") and doc.get("relationship"):
                header += f"\n    [Graph: {doc['relationship']} from seed, dist={doc.get('graph_distance',1)}]"

            parts.append(f"{header}\n\n{text}")

        return "\n\n".join(parts)

    # ── Generation ─────────────────────────────────────────────────────────────

    def generate(self, query: str, context: str, stream: bool = True) -> str:
        prompt = (
            "You are a knowledgeable AI assistant with access to the organisation's "
            "full knowledge base: Jira tickets, Confluence documentation, and GitLab "
            "code and issues. The context below was retrieved using both semantic "
            "similarity AND graph traversal (following ticket links, parent/child "
            "relationships, and documentation hierarchy).\n\n"
            "Instructions:\n"
            "- Answer using ONLY the information in the context.\n"
            "- If a ticket is mentioned as BLOCKING another, explain that relationship.\n"
            "- If the context includes linked tickets or parent epics, reference them.\n"
            "- Cite the source ticket key or page title for every claim.\n"
            "- If the context doesn't contain relevant information, say so clearly.\n"
            "- Never invent ticket keys, statuses, or assignees.\n\n"
            f"Context (from graph-augmented retrieval):\n"
            f"{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model":  OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": stream,
                    "options": {
                        "temperature": TEMPERATURE,
                        "num_predict": MAX_TOKENS,
                        "num_ctx":     NUM_CTX,
                    },
                },
                timeout=300,
                stream=stream,
            )

            if response.status_code != 200:
                return f"Error: Ollama returned HTTP {response.status_code}"

            if stream:
                full = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        full += chunk
                        print(chunk, end="", flush=True)
                        if data.get("done", False):
                            print()
                            break
                return full
            else:
                return response.json().get("response", "Error: no response from model")

        except requests.exceptions.Timeout:
            return "Error: Timeout — increase timeout or use a smaller model."
        except Exception as exc:
            return f"Error: {exc}"

    # ── Full pipeline ──────────────────────────────────────────────────────────

    def answer(
        self,
        query: str,
        top_k: int = TOP_K,
        filters: Dict[str, Any] = None,
        hops: int = GRAPH_HOPS,
        stream: bool = True,
    ) -> Dict[str, Any]:
        """
        Full graph-augmented RAG pipeline:
          detect ticket keys → vector search → graph expansion → generate
        """
        log(f"\n🔍 Query: {query}")

        # Check for explicit ticket mentions
        explicit_keys = ISSUE_KEY_PATTERN.findall(query)
        if explicit_keys:
            log(f"  Detected ticket keys: {explicit_keys}")

        docs = self.retrieve(query, top_k=top_k, filters=filters, hops=hops)

        if not docs:
            return {
                "answer":           "I couldn't find any relevant information.",
                "sources":          [],
                "graph_expanded":   0,
                "query":            query,
            }

        vector_count = sum(1 for d in docs if not d.get("from_graph") and not d.get("from_direct"))
        graph_count  = sum(1 for d in docs if d.get("from_graph"))
        direct_count = sum(1 for d in docs if d.get("from_direct"))

        log(f"  Retrieved: {vector_count} vector  +  {graph_count} graph  +  {direct_count} direct")

        context = self._build_context(docs)

        log(f"  Generating answer (model={OLLAMA_MODEL}, num_ctx={NUM_CTX}) …\n")
        answer = self.generate(query, context, stream=stream)

        return {
            "answer":         answer,
            "sources":        docs,
            "vector_count":   vector_count,
            "graph_count":    graph_count,
            "direct_count":   direct_count,
            "query":          query,
        }


# ── Interactive chat loop ──────────────────────────────────────────────────────

def chat_loop(rag: Neo4jRAG):
    print("\n" + "=" * 60)
    print("  Neo4j RAG Chat  (type 'exit' to quit, 'help' for options)")
    print("=" * 60)
    print("  Examples:")
    print("    What is blocking the payment service deployment?")
    print("    Tell me about ticket PAY-456")
    print("    Which bugs are critical priority in the auth component?")
    print("    What Confluence pages document the onboarding process?")
    print("=" * 60 + "\n")

    history = []  # for multi-turn context if needed

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        if query.lower() == "help":
            print("Commands: exit | help | stats")
            print("Flags:    --hops 3 | --top 15 | --source jira|confluence|gitlab")
            continue
        if query.lower() == "stats":
            with rag.driver.session(database=NEO4J_DATABASE) as s:
                n = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
                log(f"Total chunks in graph: {n:,}")
            continue

        # Parse inline flags from query
        hops    = GRAPH_HOPS
        top_k   = TOP_K
        filters = {}

        for flag, val in re.findall(r'--(\w+)\s+(\S+)', query):
            if flag == "hops":   hops  = int(val)
            if flag == "top":    top_k = int(val)
            if flag == "source": filters["source"] = val
        # Strip flags from query
        clean_query = re.sub(r'--\w+\s+\S+', '', query).strip()

        print(f"\nAssistant: ", end="", flush=True)
        result = rag.answer(
            clean_query,
            top_k=top_k,
            hops=hops,
            filters=filters if filters else None,
        )

        print(f"\n\n📚 Sources ({len(result['sources'])}):")
        for i, src in enumerate(result["sources"][:8], 1):
            tag = ""
            if src.get("from_direct"): tag = " [DIRECT]"
            if src.get("from_graph"):  tag = f" [GRAPH +{src.get('graph_distance',1)}hop]"
            print(f"  {i}. [{src.get('source','?').upper()}] {str(src.get('title',''))[:60]}{tag}")
            if src.get("url"):
                print(f"     {src['url']}")
        print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not all([REQUESTS_AVAILABLE, NEO4J_AVAILABLE, ST_AVAILABLE]):
        log("Missing dependencies — pip install neo4j sentence-transformers requests", "ERROR")
        return

    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "chat"

    try:
        rag = Neo4jRAG()
    except Exception as exc:
        log(f"Failed to initialize: {exc}", "ERROR")
        log("Troubleshooting:", "INFO")
        log("  1. Neo4j running?   docker ps | grep neo4j", "INFO")
        log("  2. Ollama running?  ollama serve", "INFO")
        log("  3. Model pulled?    ollama list", "INFO")
        return

    try:
        if mode == "chat":
            chat_loop(rag)

        elif mode == "ask":
            query = " ".join(sys.argv[2:])
            if not query:
                print("Usage: python neo4j_rag.py ask 'your question'")
                return
            result = rag.answer(query)
            if not result.get("answer", "").strip():
                # answer was printed via streaming, just show sources
                pass
            print(f"\n📚 Sources ({len(result['sources'])}):")
            for i, src in enumerate(result["sources"][:8], 1):
                tag = "[GRAPH]" if src.get("from_graph") else ""
                print(f"  {i}. [{src.get('source','?').upper()}] {src.get('title','')[:60]} {tag}")
                if src.get("url"):
                    print(f"     {src['url']}")

        elif mode == "test":
            test_queries = [
                "What is blocking the main deployment?",
                "Which tickets are critical priority and still open?",
                "Explain the authentication architecture",
                "What Confluence pages relate to the onboarding process?",
            ]
            for q in test_queries:
                print(f"\n{'='*60}")
                result = rag.answer(q, stream=False)
                print(f"Q: {q}")
                print(f"A: {result['answer'][:500]}")
                print(f"Sources: {len(result['sources'])}  "
                      f"(vector={result['vector_count']} graph={result['graph_count']})")
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python neo4j_rag.py [chat|ask|test]")
            print("  chat                    — interactive chat loop")
            print("  ask 'your question'     — single question")
            print("  test                    — run predefined test queries")
    finally:
        rag.close()


if __name__ == "__main__":
    main()

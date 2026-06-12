#!/usr/bin/env python3
"""
rag_generator_test.py  —  OPTIMIZED RAG generator  (A/B test variant)
=====================================================================
Drop-in variant of rag_generator.py. Same public class `RAGGenerator`
and the SAME method signatures, so intent_agent.py works if you point it
at this module instead:
    retrieve(query, top_k, filters, hops)          → List[Dict]
    generate(query, context, stream)               → str
    answer_question(query, top_k, filters, hops, stream) → Dict

It uses the optimized retrieval engine (neo4j_search_test.Neo4jSearch).

WHAT CHANGED vs rag_generator.py  (see DIAGNOSIS.md)
─────────────────────────────────────────────────────
1. ISSUE-KEY FAST PATH (quality + latency).  If the query names a Jira key
   (e.g. "PROJ-18399"), that ticket is fetched DIRECTLY from the graph by its
   unique-constrained issue_key — no embedding, no ANN, no graph walk needed
   for that part. The authoritative ticket is pinned to the top of context,
   with its status/priority/assignee and linked tickets. The old generator
   had no such routing, so a question naming a ticket relied entirely on the
   embedding happening to surface it.

2. SINGLE OLLAMA HEALTH CHECK (latency).  Startup made 3–4 separate HTTP
   round-trips to /api/tags (check + resolve + check-model + list). Collapsed
   into ONE call whose result is reused.

3. ADAPTIVE CONTEXT BUDGET (quality).  Per-source text was hard-truncated at
   700 chars regardless of NUM_CTX. Now the budget is computed from NUM_CTX
   (minus the prompt scaffold and answer reservation) and split across
   sources, so a roomy context window is actually used (capped to stay safe).

4. SOURCE DEDUP (quality).  Identical chunks (and near-duplicate direct/vector
   hits) are collapsed before context is built.

5. TIGHTER CITATION PROMPT.  Asks for [Source N] citations, explicit ticket
   key + Status + Priority, and explanation of BLOCKS/relationship links.

6. self.last_timings exposes retrieve/generate latency for the A/B harness.

.env keys
─────────
  (all of rag_generator.py's keys, plus the neo4j_search_test reranker keys)
  CONTEXT_CHARS_PER_SOURCE   0   0 = auto (derive from NUM_CTX); else fixed cap
"""

import os
import re
import json
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    print("⚠️  requests not installed. Run: pip install requests")

# SWAPPED-IN BUILD: canonical names now hold the optimized code.
# neo4j_search = optimized engine (originals are parked under *_test names).
try:
    from neo4j_search import Neo4jSearch
    NEO4J_SEARCH_AVAILABLE = True
except ImportError:
    try:
        from rag.neo4j_search import Neo4jSearch
        NEO4J_SEARCH_AVAILABLE = True
    except ImportError:
        NEO4J_SEARCH_AVAILABLE = False
        print("⚠️  neo4j_search not importable — check sys.path")

# ── Config ──────────────────────────────────────────────────────────────────────
OLLAMA_HOST      = os.getenv("OLLAMA_HOST",      "http://localhost:11434").strip().rstrip("/")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL",     "qwen3").strip()
MAX_TOKENS       = int(os.getenv("MAX_TOKENS",      "1024"))
TEMPERATURE      = float(os.getenv("TEMPERATURE",   "0.2"))
NUM_CTX          = int(os.getenv("NUM_CTX",         "8192"))
GRAPH_HOPS       = int(os.getenv("GRAPH_HOPS",      "2"))
MAX_GRAPH_EXPAND = int(os.getenv("MAX_GRAPH_EXPAND", "20"))
CTX_CHARS_PER_SRC = int(os.getenv("CONTEXT_CHARS_PER_SOURCE", "0"))  # 0 = auto

# Jira issue keys, e.g. PROJ-18399, PAY-456
ISSUE_KEY_PATTERN = re.compile(r'\b([A-Z][A-Z0-9]{1,9}-\d+)\b')

logger = logging.getLogger("rag_generator_test")


def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


# Direct, indexed ticket fetch — bypasses embedding/ANN for explicitly named keys.
DIRECT_TICKET_CQL = """
MATCH (t:JiraTicket {issue_key: $key})
OPTIONAL MATCH (c:Chunk)-[:PART_OF]->(t)
WITH t, c ORDER BY c.chunk_index ASC
WITH t, [x IN collect(c) WHERE x IS NOT NULL | x.text][0..3] AS texts
OPTIONAL MATCH (t)-[link]->(rel:JiraTicket)
  WHERE type(link) IN [
    'BLOCKS','BLOCKED_BY','RELATES_TO','DUPLICATES','DUPLICATED_BY',
    'IS_TESTED_BY','TESTS','CAUSES','CAUSED_BY','CLONES','CLONED_BY',
    'IMPLEMENTS','IMPLEMENTED_BY','SPLIT_FROM','SPLIT_TO',
    'IS_AUTOMATED_BY','AUTOMATES','REVIEWED_BY','CHILD_OF'
  ]
RETURN t.issue_key    AS issue_key,
       t.title        AS title,
       t.status       AS status,
       t.priority     AS priority,
       t.issue_type   AS issue_type,
       t.assignee     AS assignee,
       t.url          AS url,
       t.project_name AS project_name,
       texts,
       collect(DISTINCT {rel: type(link), key: rel.issue_key,
                         title: rel.title, status: rel.status})[0..8] AS links
"""


# ── Result normaliser ────────────────────────────────────────────────────────────

def _normalise_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text":           hit.get("text") or "",
        "score":          float(hit.get("score") or 0.0),
        "rerank_score":   float(hit.get("rerank_score") or hit.get("score") or 0.0),
        "source":         hit.get("source") or "",
        "source_type":    hit.get("source_type") or "",
        "title":          hit.get("title") or "",
        "url":            hit.get("url") or "N/A",
        "project":        hit.get("project_name") or hit.get("project") or "N/A",
        "project_name":   hit.get("project_name") or hit.get("project") or "N/A",
        "status":         hit.get("status") or hit.get("ticket_status") or "",
        "priority":       hit.get("priority") or "",
        "assignee":       hit.get("assignee") or "",
        "issue_key":      hit.get("issue_key") or "",
        "from_graph":     hit.get("from_graph", False),
        "from_direct":    hit.get("from_direct", False),
        "relationship":   hit.get("relationship") or "",
        "graph_distance": hit.get("graph_distance"),
        "graph_context":  hit.get("graph_context"),
    }


# ── Context builder ──────────────────────────────────────────────────────────────

def _context_budget(n_sources: int) -> int:
    """Per-source char budget derived from the model's context window."""
    if CTX_CHARS_PER_SRC > 0:
        return CTX_CHARS_PER_SRC
    # ~3.5 chars/token. Reserve room for the prompt scaffold + the answer.
    usable_tokens = max(NUM_CTX - MAX_TOKENS - 600, 1200)
    usable_chars  = usable_tokens * 3.5
    per = int(usable_chars / max(n_sources, 1))
    return max(400, min(per, 2000))


def _build_context(sources: List[Dict[str, Any]]) -> str:
    budget = _context_budget(len(sources))
    parts = []
    for i, doc in enumerate(sources, 1):
        src = (doc.get("source") or "").upper()
        if doc.get("from_direct"):
            tag = " [direct lookup]"
        elif doc.get("from_graph") and doc.get("relationship"):
            tag = f" [graph: {doc['relationship']} dist={doc.get('graph_distance', 1)}]"
        elif doc.get("from_graph"):
            tag = " [graph-expanded]"
        else:
            tag = ""

        title = doc.get("title", "")
        text  = (doc.get("text") or "")[:budget]

        meta_items = []
        if doc.get("issue_key"):    meta_items.append(f"Key: {doc['issue_key']}")
        if doc.get("status"):       meta_items.append(f"Status: {doc['status']}")
        if doc.get("priority"):     meta_items.append(f"Priority: {doc['priority']}")
        if doc.get("assignee"):     meta_items.append(f"Assignee: {doc['assignee']}")
        if doc.get("project_name") and doc["project_name"] != "N/A":
            meta_items.append(f"Project: {doc['project_name']}")
        if doc.get("url") and doc["url"] != "N/A":
            meta_items.append(f"URL: {doc['url']}")

        header = f"[Source {i}] [{src}] {title}{tag}"

        # Structured link context from a direct ticket fetch.
        gctx = doc.get("graph_context") or {}
        link_lines = []
        for lnk in (gctx.get("links") or []):
            if lnk.get("key"):
                link_lines.append(
                    f"{lnk.get('rel','')} {lnk['key']} "
                    f"({(lnk.get('title') or '')[:40]}) [{lnk.get('status','')}]"
                )

        block = header
        if meta_items:
            block += "\n" + "  |  ".join(meta_items)
        if link_lines:
            block += "\n  Links: " + "; ".join(link_lines)
        block += f"\n{text}"
        parts.append(block)

    return "\n\n---\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# RAGGenerator
# ══════════════════════════════════════════════════════════════════════════════

class RAGGenerator:

    def __init__(self):
        global OLLAMA_MODEL

        if not REQUESTS_AVAILABLE:
            raise ImportError("pip install requests")
        if not NEO4J_SEARCH_AVAILABLE:
            raise ImportError("neo4j_search_test not importable")

        log("=" * 60)
        log("Initializing RAG System (Neo4j + Ollama)  [OPTIMIZED test build]")
        log("=" * 60)

        self._search = Neo4jSearch()
        log("✓ Neo4j search ready (vector + multi-hop graph + cross-encoder rerank)")
        log(f"  graph_hops={GRAPH_HOPS}  max_graph_expand={MAX_GRAPH_EXPAND}")

        # Single /api/tags round-trip: connectivity + model resolution + presence.
        log(f"Connecting to Ollama at {OLLAMA_HOST}")
        models = self._ollama_tags()
        if models is None:
            raise ConnectionError(
                f"Cannot connect to Ollama at {OLLAMA_HOST}\n"
                "Make sure Ollama is running: ollama serve"
            )
        log("✓ Connected to Ollama")

        resolved = self._resolve_model(OLLAMA_MODEL, models)
        if resolved != OLLAMA_MODEL:
            log(f"Model name resolved: '{OLLAMA_MODEL}' → '{resolved}'")
            OLLAMA_MODEL = resolved

        if OLLAMA_MODEL not in models:
            log(f"Model '{OLLAMA_MODEL}' not found. Available:", "WARN")
            for m in models:
                log(f"  - {m}", "WARN")
            raise ValueError(
                f"Model '{OLLAMA_MODEL}' not found.\nPull it with: ollama pull {OLLAMA_MODEL}"
            )
        log(f"✓ Model '{OLLAMA_MODEL}' ready")
        log(f"  temperature: {TEMPERATURE}  |  num_ctx: {NUM_CTX}  |  max_tokens: {MAX_TOKENS}")
        log("=" * 60)
        log("✓ RAG System Ready!")
        log("=" * 60)

        self.last_timings: Dict[str, float] = {}

    def close(self):
        try:
            self._search.close()
        except Exception:
            pass

    # ── Ollama helpers (one HTTP call) ──────────────────────────────────────────

    def _ollama_tags(self) -> Optional[List[str]]:
        """Return the list of installed model names, or None if unreachable."""
        try:
            r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if r.status_code != 200:
                return None
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return None

    @staticmethod
    def _resolve_model(requested: str, available: List[str]) -> str:
        if requested in available:
            return requested
        base = requested.split(":")[0]
        for name in available:
            if name.split(":")[0] == base:
                return name
        return requested

    # ── Direct ticket fetch ─────────────────────────────────────────────────────

    def _direct_tickets(self, keys: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for key in dict.fromkeys(keys):  # de-dupe, keep order
            try:
                rows = self._search._run(DIRECT_TICKET_CQL, {"key": key})
            except Exception as exc:
                log(f"  direct fetch failed for {key}: {exc}", "WARN")
                continue
            if not rows:
                continue
            r = rows[0]
            texts = r.get("texts") or [""]
            body = "\n".join(t for t in texts if t)
            out.append({
                "chunk_id":     f"direct_{key}",
                "text":         body or (r.get("title") or ""),
                "title":        r.get("title") or key,
                "source":       "jira",
                "source_type":  r.get("issue_type") or "issue",
                "project_name": r.get("project_name") or "",
                "url":          r.get("url") or "",
                "status":       r.get("status") or "",
                "priority":     r.get("priority") or "",
                "assignee":     r.get("assignee") or "",
                "issue_key":    key,
                "score":        1.0,
                "rerank_score": 1.0,
                "from_direct":  True,
                "from_graph":   False,
                "graph_context": {"links": [l for l in (r.get("links") or []) if l.get("key")]},
            })
        return out

    # ── Retrieval ────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        hops: int = GRAPH_HOPS,
        max_expand: int = MAX_GRAPH_EXPAND,
    ) -> List[Dict[str, Any]]:
        source  = filters.get("source")  if filters else None
        project = filters.get("project") if filters else None

        # Fast path: explicitly named tickets fetched directly.
        keys = ISSUE_KEY_PATTERN.findall(query)
        direct = self._direct_tickets(keys) if keys else []
        if direct:
            log(f"  Direct ticket fetch: {[d['issue_key'] for d in direct]}")

        # Hybrid retrieval (push project filter into the engine; no Python over-fetch).
        hits = self._search.hybrid_search(
            query=query,
            top_k=top_k,
            source_filter=source,
            project_filter=project,
            hops=hops,
            max_expand=max_expand,
        )
        self.last_timings = dict(getattr(self._search, "last_timings", {}) or {})

        # Merge direct + hybrid, dedup by issue_key / chunk_id, direct wins.
        seen_keys   = {d["issue_key"] for d in direct}
        seen_chunks = {d["chunk_id"] for d in direct}
        merged = list(direct)
        for h in hits:
            if h.get("issue_key") and h["issue_key"] in seen_keys:
                continue
            if h.get("chunk_id") in seen_chunks:
                continue
            seen_chunks.add(h.get("chunk_id"))
            merged.append(h)

        merged = merged[:max(top_k, len(direct))]
        return [_normalise_hit(h) for h in merged]

    # ── Generation ───────────────────────────────────────────────────────────────

    def generate(self, query: str, context: str, stream: bool = True) -> str:
        prompt = (
            "You are a precise AI assistant with access to the organisation's "
            "knowledge base (Jira, Confluence, GitLab). The context below was "
            "retrieved by semantic search, graph traversal of ticket links, and "
            "cross-encoder reranking.\n\n"
            "Rules:\n"
            "- Answer using ONLY the information in the numbered context blocks.\n"
            "- Cite every claim with its [Source N] label or ticket key.\n"
            "- For Jira tickets, state the issue key, Status, and Priority when present.\n"
            "- If a ticket is linked to another (e.g. BLOCKS, RELATES_TO, CHILD_OF), "
            "explain that relationship using the Links provided.\n"
            "- If the context is insufficient, say exactly: "
            "'I do not have enough information in the knowledge base to answer this.'\n"
            "- Never invent ticket keys, names, dates, statuses, or assignees.\n"
            "- Be concise; use bullet points for lists.\n\n"
            f"Context:\n{context}\n\n"
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
                timeout=180,
                stream=stream,
            )
            if response.status_code != 200:
                return f"Error: Ollama returned HTTP {response.status_code}"

            if stream:
                full = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        full += data.get("response", "")
                        if data.get("done", False):
                            break
                return full
            return response.json().get("response", "Error: no response from model")

        except requests.exceptions.Timeout:
            return "Error: Request timed out."
        except Exception as exc:
            return f"Error generating response: {exc}"

    # ── Full RAG pipeline ─────────────────────────────────────────────────────────

    def answer_question(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        hops: int = GRAPH_HOPS,
        stream: bool = True,
    ) -> Dict[str, Any]:
        t_retrieve = time.perf_counter()
        sources = self.retrieve(query, top_k=top_k, filters=filters, hops=hops)
        retrieve_ms = (time.perf_counter() - t_retrieve) * 1000

        if not sources:
            return {
                "answer":      "I do not have enough information in the knowledge base to answer this.",
                "sources":     [],
                "query":       query,
                "num_sources": 0,
                "timings":     {"retrieve_ms": retrieve_ms, "generate_ms": 0.0},
            }

        vector_count = sum(1 for s in sources if not s.get("from_graph") and not s.get("from_direct"))
        graph_count  = sum(1 for s in sources if s.get("from_graph"))
        direct_count = sum(1 for s in sources if s.get("from_direct"))
        log(f"  Retrieved: {vector_count} vector + {graph_count} graph + {direct_count} direct (hops={hops})")

        context = _build_context(sources)

        t_gen = time.perf_counter()
        answer = self.generate(query, context, stream=stream)
        generate_ms = (time.perf_counter() - t_gen) * 1000

        timings = dict(self.last_timings)
        timings.update({"retrieve_ms": retrieve_ms, "generate_ms": generate_ms})

        return {
            "answer":       answer,
            "sources":      sources,
            "query":        query,
            "num_sources":  len(sources),
            "vector_count": vector_count,
            "graph_count":  graph_count,
            "direct_count": direct_count,
            "timings":      timings,
        }


# ── Smoke test / CLI ────────────────────────────────────────────────────────────

def _print_result(result: Dict[str, Any]):
    print(f"\n💡 Answer:\n{result['answer']}")
    t = result.get("timings", {})
    if t:
        print("\n⏱  timings(ms): " + "  ".join(f"{k}={v:.0f}" for k, v in t.items()))
    print(f"\n📚 Sources ({result['num_sources']})  "
          f"[vector={result.get('vector_count',0)} graph={result.get('graph_count',0)} "
          f"direct={result.get('direct_count',0)}]:")
    for i, src in enumerate(result["sources"], 1):
        if src.get("from_direct"):
            tag = " [DIRECT]"
        elif src.get("from_graph"):
            tag = f" [GRAPH +{src.get('graph_distance','?')}hop]"
        else:
            tag = ""
        print(f"\n{i}. [{src['source'].upper()}] {src['title']}{tag}")
        print(f"   Score:    {src['score']:.3f}")
        if src.get("issue_key"): print(f"   Key:      {src['issue_key']}")
        if src.get("status"):    print(f"   Status:   {src['status']}")
        if src.get("priority"):  print(f"   Priority: {src['priority']}")
        if src.get("assignee"):  print(f"   Assignee: {src['assignee']}")
        print(f"   URL:      {src['url']}")
    print("\n" + "=" * 60)


def test_rag():
    log("Testing OPTIMIZED RAG System (Neo4j + Ollama)")
    try:
        rag = RAGGenerator()
    except Exception as exc:
        log(f"Failed to initialize: {exc}", "ERROR")
        return
    for query in [
        "What is the architecture of ExampleProduct?",
        "What is PROJ-18399 about and what does the ExampleProduct documentation say about CC errors?",
        "What are the open PROJ bugs related to alarms and what does alarm documentation say?",
    ]:
        log(f"\n🔍 Question: {query}")
        _print_result(rag.answer_question(query, top_k=5, hops=GRAPH_HOPS))
    rag.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        try:
            rag = RAGGenerator()
        except Exception as exc:
            log(f"Failed to initialize: {exc}", "ERROR")
            sys.exit(1)
        log(f"\n🔍 Question: {query}")
        _print_result(rag.answer_question(query, top_k=5, hops=GRAPH_HOPS))
        rag.close()
    else:
        test_rag()

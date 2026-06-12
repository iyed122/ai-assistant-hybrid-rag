#!/usr/bin/env python3
"""
rag_generator.py  —  Neo4j edition (replaces Qdrant)
======================================================
RAG Generator — Neo4j hybrid retrieval + Ollama generation.

Fixes applied in this version
──────────────────────────────
1. project filter was silently dropped — hybrid_search() in neo4j_search.py
   has no project parameter. Filters from intent_agent.py carrying
   {"source": "jira", "project": "AUTH"} were passed but ignored, meaning
   the assistant would answer questions scoped to a project using results
   from the entire knowledge base.
   FIX: project filtering is applied as a post-retrieval step on the results
   returned by hybrid_search, which already have project_name populated.

2. Context was metadata-stripped — the original build was:
       f"[Source {i}] {doc['title']}\n{doc['text'][:600]}"
   No status, priority, assignee, url, or issue_key. The LLM could not
   answer "who is assigned to this blocker?" without hallucinating.
   FIX: _build_context() now includes all structured metadata fields.

3. TEMPERATURE left at 0.7 (default passthrough) — enterprise factual RAG
   should minimise variance. 0.7 is appropriate for creative tasks, not for
   structured data retrieval where accuracy and repeatability matter.
   FIX: default changed to 0.2. Still overridable via .env TEMPERATURE.

Public API is UNCHANGED — intent_agent.py calls:
    rag.retrieve(query, top_k, filters)        → List[Dict]
    rag.answer_question(query, top_k, filters) → {answer, sources, query, num_sources}
    rag.generate(query, context)               → str

.env keys used
──────────────
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
  EMBEDDING_MODEL   Snowflake/snowflake-arctic-embed-m
  SEARCH_TOP_K      10
  OLLAMA_HOST       http://localhost:11434
  OLLAMA_MODEL      qwen3
  MAX_TOKENS        1024
  TEMPERATURE       0.2   ← changed from 0.7
  NUM_CTX           8192
"""

import os
import json
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

try:
    from neo4j_search import Neo4jSearch
    NEO4J_SEARCH_AVAILABLE = True
except ImportError:
    try:
        from rag.neo4j_search import Neo4jSearch
        NEO4J_SEARCH_AVAILABLE = True
    except ImportError:
        NEO4J_SEARCH_AVAILABLE = False
        print("⚠️  neo4j_search not importable — check sys.path / pip install neo4j sentence-transformers")

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434").strip().rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3").strip()
MAX_TOKENS   = int(os.getenv("MAX_TOKENS",   "1024"))
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.2"))   # FIX: was 0.7
NUM_CTX      = int(os.getenv("NUM_CTX",      "8192"))

logger = logging.getLogger("rag_generator")


def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


# ── Ollama helpers ──────────────────────────────────────────────────────────────

def _resolve_model_name(requested: str, host: str) -> str:
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        if r.status_code != 200:
            return requested
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


def _check_ollama(host: str) -> bool:
    try:
        return requests.get(f"{host}/api/tags", timeout=5).status_code == 200
    except Exception:
        return False


def _check_model(host: str, model: str) -> bool:
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        if r.status_code == 200:
            return model in [m["name"] for m in r.json().get("models", [])]
        return False
    except Exception:
        return False


def _list_models(host: str) -> List[str]:
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
        return []
    except Exception:
        return []


# ── Result normaliser ───────────────────────────────────────────────────────────

def _normalise_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a Neo4jSearch hit to the flat dict that intent_agent.py expects.
    project_name → project (matches original Qdrant payload key).
    """
    return {
        "text":         hit.get("text") or "",
        "score":        float(hit.get("score") or 0.0),
        "source":       hit.get("source") or "",
        "source_type":  hit.get("source_type") or "",
        "title":        hit.get("title") or "",
        "url":          hit.get("url") or "N/A",
        "project":      hit.get("project_name") or hit.get("project") or "N/A",
        "project_name": hit.get("project_name") or hit.get("project") or "N/A",
        "status":       hit.get("status") or "",
        "priority":     hit.get("priority") or "",
        "assignee":     hit.get("assignee") or "",
        "issue_key":    hit.get("issue_key") or "",
        "from_graph":   hit.get("from_graph", False),
    }


# ── Context builder ─────────────────────────────────────────────────────────────

def _build_context(sources: List[Dict[str, Any]]) -> str:
    """
    Build a structured, metadata-rich context string for the LLM.

    FIX: the original built only f"[Source {i}] {doc['title']}\n{doc['text'][:600]}"
    with no status, priority, assignee, url, or key.  This version includes all
    structured fields so the LLM can answer ownership and triage questions.
    """
    parts = []
    for i, doc in enumerate(sources, 1):
        src   = (doc.get("source") or "").upper()
        tag   = " [graph-expanded]" if doc.get("from_graph") else ""
        title = doc.get("title", "")
        text  = (doc.get("text") or "")[:700]

        meta_items = []
        if doc.get("issue_key"):    meta_items.append(f"Key: {doc['issue_key']}")
        if doc.get("status"):       meta_items.append(f"Status: {doc['status']}")
        if doc.get("priority"):     meta_items.append(f"Priority: {doc['priority']}")
        if doc.get("assignee"):     meta_items.append(f"Assignee: {doc['assignee']}")
        if doc.get("project_name"): meta_items.append(f"Project: {doc['project_name']}")
        if doc.get("url"):          meta_items.append(f"URL: {doc['url']}")

        meta_str = "  |  ".join(meta_items)
        header   = f"[Source {i}] [{src}] {title}{tag}"
        block    = f"{header}\n{meta_str}\n{text}" if meta_str else f"{header}\n{text}"
        parts.append(block)

    return "\n\n---\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# RAGGenerator
# ══════════════════════════════════════════════════════════════════════════════

class RAGGenerator:
    """
    Neo4j-backed RAG system.

    Retrieval: Neo4jSearch.hybrid_search() — vector + graph + keyword re-rank
    Generation: Ollama (qwen3 or any local model)

    Public API (unchanged from Qdrant version):
        retrieve(query, top_k, filters) → List[Dict]
        generate(query, context, stream) → str
        answer_question(query, top_k, filters, stream) → Dict
    """

    def __init__(self):
        global OLLAMA_MODEL

        if not REQUESTS_AVAILABLE:
            raise ImportError("pip install requests")
        if not NEO4J_SEARCH_AVAILABLE:
            raise ImportError(
                "neo4j_search not importable — pip install neo4j sentence-transformers"
            )

        log("=" * 60)
        log("Initializing RAG System (Neo4j + Ollama)")
        log("=" * 60)

        log("Connecting to Neo4j …")
        self._search = Neo4jSearch()
        log("✓ Neo4j search ready (hybrid: vector + graph + keyword)")

        log(f"Connecting to Ollama at {OLLAMA_HOST}")
        if not _check_ollama(OLLAMA_HOST):
            raise ConnectionError(
                f"Cannot connect to Ollama at {OLLAMA_HOST}\n"
                "Make sure Ollama is running: ollama serve"
            )
        log("✓ Connected to Ollama")

        resolved = _resolve_model_name(OLLAMA_MODEL, OLLAMA_HOST)
        if resolved != OLLAMA_MODEL:
            log(f"Model name resolved: '{OLLAMA_MODEL}' → '{resolved}'")
            OLLAMA_MODEL = resolved

        if not _check_model(OLLAMA_HOST, OLLAMA_MODEL):
            log(f"Model '{OLLAMA_MODEL}' not found. Available:", "WARN")
            for m in _list_models(OLLAMA_HOST):
                log(f"  - {m}", "WARN")
            raise ValueError(
                f"Model '{OLLAMA_MODEL}' not found.\n"
                f"Pull it with: ollama pull {OLLAMA_MODEL}"
            )
        log(f"✓ Model '{OLLAMA_MODEL}' ready")
        log(f"  temperature: {TEMPERATURE}  |  num_ctx: {NUM_CTX}  |  max_tokens: {MAX_TOKENS}")
        log("=" * 60)
        log("✓ RAG System Ready!")
        log("=" * 60)

    def close(self):
        try:
            self._search.close()
        except Exception:
            pass

    # ── Retrieval ───────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid vector+graph retrieval from Neo4j.

        Args:
            query:   Natural language question.
            top_k:   Number of results to return.
            filters: Optional dict.  Supported keys:
                       "source"  → passed to hybrid_search as source_filter
                       "project" → applied as post-retrieval filter on project_name
                                   FIX: was passed as kwarg that hybrid_search ignores.

        Returns:
            List of normalised hit dicts.
        """
        source  = None
        project = None
        if filters:
            source  = filters.get("source")
            project = filters.get("project")

        hits = self._search.hybrid_search(
            query=query,
            top_k=top_k * 5 if project else top_k,  # over-fetch if filtering by project
            source_filter=source,
        )

        # FIX: project filter — neo4j_search.hybrid_search has no project param
        # so we apply it here on the returned results.
        if project:
            project_lower = project.lower()
            hits = [
                h for h in hits
                if project_lower in (h.get("project_name") or "").lower()
                or project_lower in (h.get("project_id") or "").lower()
            ]

        return [_normalise_hit(h) for h in hits[:top_k]]

    # ── Generation ──────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        context: str,
        stream: bool = True,
    ) -> str:
        prompt = (
            "You are a precise AI assistant with access to the organisation's "
            "knowledge base (Jira, Confluence, GitLab).\n\n"
            "Rules:\n"
            "- Answer using ONLY the information in the numbered context blocks below.\n"
            "- Cite sources by their [Source N] label or ticket key.\n"
            "- If context is insufficient, say: "
            "'I do not have enough information in the knowledge base to answer this.'\n"
            "- Never invent ticket keys, names, dates, or statuses.\n"
            "- For Jira tickets always state Status and Priority when present.\n"
            "- Be concise. Use bullet points for lists.\n\n"
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
            else:
                return response.json().get("response", "Error: no response from model")

        except requests.exceptions.Timeout:
            return "Error: Request timed out."
        except Exception as exc:
            return f"Error generating response: {exc}"

    # ── Full RAG pipeline ───────────────────────────────────────────────────

    def answer_question(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        stream: bool = True,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieve → build rich context → generate.
        """
        sources = self.retrieve(query, top_k=top_k, filters=filters)

        if not sources:
            return {
                "answer":      "I do not have enough information in the knowledge base to answer this.",
                "sources":     [],
                "query":       query,
                "num_sources": 0,
            }

        context = _build_context(sources)
        answer  = self.generate(query, context, stream=stream)

        return {
            "answer":      answer,
            "sources":     sources,
            "query":       query,
            "num_sources": len(sources),
        }


# ── Smoke test ──────────────────────────────────────────────────────────────────

def test_rag():
    log("=" * 60)
    log("Testing RAG System (Neo4j + Ollama)")
    log("=" * 60)

    try:
        rag = RAGGenerator()
    except Exception as exc:
        log(f"Failed to initialize: {exc}", "ERROR")
        log("  1. Start Neo4j:    docker compose up -d", "INFO")
        log("  2. Run import:     python -m rag.neo4j_import", "INFO")
        log("  3. Start Ollama:   ollama serve", "INFO")
        log("  4. Check model:    ollama list", "INFO")
        return

    test_queries = [
        "How do I authenticate with OAuth?",
        "What are the database connection issues?",
        "How do I submit a merge request?",
    ]

    for query in test_queries:
        log(f"\n🔍 Question: {query}")
        log("-" * 60)
        result = rag.answer_question(query, top_k=5)
        print(f"\n💡 Answer:\n{result['answer']}")
        print(f"\n📚 Sources ({result['num_sources']}):")
        for i, src in enumerate(result["sources"], 1):
            print(f"\n{i}. [{src['source'].upper()}] {src['title']}")
            print(f"   Score:    {src['score']:.3f}")
            if src.get("issue_key"): print(f"   Key:      {src['issue_key']}")
            if src.get("status"):    print(f"   Status:   {src['status']}")
            if src.get("priority"):  print(f"   Priority: {src['priority']}")
            if src.get("assignee"):  print(f"   Assignee: {src['assignee']}")
            print(f"   URL:      {src['url']}")
            print(f"   Project:  {src['project']}")
        print("\n" + "=" * 60)

    rag.close()


if __name__ == "__main__":
    test_rag()
"""
FastAPI backend for the AI Assistant UI.
Wraps intent_agent.py and exposes SSE streaming + conversation endpoints.

v4.1 changes:
  - Parallel sentry dispatch automatic (sentries_node updated in intent_agent)
  - Checkpointing: passes conversation_id as LangGraph thread_id
  - Uses shared _get_mongo_client() from intent_agent instead of own client
"""
import re
import asyncio
import json
import sys
import os
import uuid
from datetime import datetime
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Make sure project root is on the path ────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Import project modules ────────────────────────────────────────────────────
from agent.intent_agent import (
    intent_node,
    rag_node,
    sentries_node,
    rag_fallback_node,
    gpu_clear_node,
    AgentState,
    _get_mongo_collection,
    _conversation_history,
    _sentries_have_data,
    HISTORY_MAX_TURNS,
)

try:
    from agent.weaver_node import weave_stream
    WEAVER_AVAILABLE = True
except ImportError:
    WEAVER_AVAILABLE = False

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# ── Training / fine-tuning data router ───────────────────────────────────────
try:
    from api.training_router import router as training_router
    TRAINING_ROUTER_AVAILABLE = True
except ImportError:
    try:
        from training_router import router as training_router
        TRAINING_ROUTER_AVAILABLE = True
    except ImportError:
        TRAINING_ROUTER_AVAILABLE = False

# ── Training pipeline router (MLflow + QLoRA/DPO) ──────────────────────────
try:
    from api.training_pipeline_router import router as pipeline_router
    PIPELINE_ROUTER_AVAILABLE = True
except ImportError:
    try:
        from training_pipeline_router import router as pipeline_router
        PIPELINE_ROUTER_AVAILABLE = True
    except ImportError:
        PIPELINE_ROUTER_AVAILABLE = False

# ── SSE helper ────────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Assistant API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount training router ─────────────────────────────────────────────────────
if TRAINING_ROUTER_AVAILABLE:
    app.include_router(training_router)
else:
    import logging
    logging.getLogger("main").warning("training_router not found — /training/* endpoints unavailable")

if PIPELINE_ROUTER_AVAILABLE:
    app.include_router(pipeline_router)
else:
    import logging
    logging.getLogger("main").warning("training_pipeline_router not found — pipeline endpoints unavailable")

# ── Request / Response models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class NewConversationResponse(BaseModel):
    conversation_id: str


# ── SSE streaming endpoint ────────────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream a response to the client via SSE.

    Pipeline:
      1. intent_node     — classify query (keyword heuristic + Qwen3 /no_think fallback)
      2. rag_node        — Qdrant retrieval (if intent includes rag)
      3. sentries_node   — parallel API dispatch (if intent includes sentries)
      4. gpu_clear_node  — Python GC before Weaver
      5. weave_stream    — Qwen3 token streaming via asyncio Queue bridge

    SSE events sent:
      {type: "intent",   intent: "rag"|"sentries"|"both"}
      {type: "token",    content: "<chunk>"}           — one per Weaver token
      {type: "metadata", intent, sources, timing}      — after stream completes
      {type: "done"}                                   — signals completion
      {type: "error",    content: "<message>"}         — on any exception
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        query = request.query.strip()
        if not query:
            yield _sse({"type": "error", "content": "Empty query"})
            return

        # Ensure we have a conversation_id — used for checkpointing thread_id
        conversation_id = request.conversation_id or str(uuid.uuid4())

        try:
            # ── Build initial state ───────────────────────────────────────
            state: AgentState = {
                "query": query,
                "timing": {},
                "gpu_cleared": False,
                "final_answer": "",
                "error": None,
                "rag_result": None,
                "sentries_queries": None,
                "sentries_results": None,
                "conversation_history": list(_conversation_history[-HISTORY_MAX_TURNS:]),
            }

            loop = asyncio.get_event_loop()

            # ── Intent classification ─────────────────────────────────────
            state = await loop.run_in_executor(None, intent_node, state)
            intent = state.get("intent", "both")
            yield _sse({"type": "intent", "intent": intent})

            # ── RAG retrieval ─────────────────────────────────────────────
            if intent in ("rag", "both"):
                state = await loop.run_in_executor(None, rag_node, state)

            # ── Parallel sentry dispatch ──────────────────────────────────
            # sentries_node now uses ThreadPoolExecutor internally.
            # GitLab, Jira, Confluence fire simultaneously — no code change
            # needed here, the parallelism is inside sentries_node.
            if intent in ("sentries", "both"):
                state = await loop.run_in_executor(None, sentries_node, state)


            # ── RAG fallback: pure-sentries with no results → RAG ────────────
            # Mirrors route_after_sentries in intent_agent.py. Without this,
            # API "sentries" queries returning nothing reach Weaver with zero context.
            if intent == "sentries" and not _sentries_have_data(state):
                state = await loop.run_in_executor(None, rag_fallback_node, state)

            # ── GPU/memory clear ──────────────────────────────────────────
            state = await loop.run_in_executor(None, gpu_clear_node, state)

            # ── Stream Weaver tokens via asyncio Queue bridge ─────────────
            # weave_stream() is a synchronous generator.
            # We run it in a background thread and push each token to a
            # queue as it arrives — yielding immediately to the browser.
            full_answer = ""

            if WEAVER_AVAILABLE:
                rag_result       = state.get("rag_result")
                sentries_results = state.get("sentries_results")
                sentries_queries = state.get("sentries_queries")
                history          = state.get("conversation_history", [])

                queue = asyncio.Queue()
                _DONE = object()  # sentinel

                def _produce():
                    try:
                        for chunk in weave_stream(
                            query=query,
                            intent=intent,
                            rag_result=rag_result,
                            sentries_results=sentries_results,
                            sentries_queries=sentries_queries,
                            conversation_history=history,
                        ):
                            loop.call_soon_threadsafe(queue.put_nowait, chunk)
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, _DONE)

                loop.run_in_executor(None, _produce)

                while True:
                    chunk = await queue.get()
                    if chunk is _DONE:
                        break
                    full_answer += chunk
                    yield _sse({"type": "token", "content": chunk})
            else:
                msg = "Weaver not available."
                full_answer = msg
                yield _sse({"type": "token", "content": msg})

            # ── Build metadata ────────────────────────────────────────────
            rag      = state.get("rag_result") or {}
            sources  = rag.get("sources", [])
            timing   = state.get("timing", {})

            # ── Persist to MongoDB ────────────────────────────────────────
            if full_answer and not full_answer.startswith("ERROR"):
                _conversation_history.append({
                    "query":  query,
                    "answer": full_answer[:800],
                    "intent": intent,
                })
                collection = _get_mongo_collection()
                if collection is not None:
                    try:
                        # Helper to force-extract Jira keys if they are missing
                        def get_jira_key(url, current_key):
                            if current_key: return current_key
                            match = re.search(r'\b([A-Z]{2,8}-\d+)\b', str(url))
                            return match.group(1) if match else ""

                        collection.insert_one({
                            "conversation_id": conversation_id,
                            "query":           query,
                            "answer":          full_answer,
                            "intent":          intent,
                            "timing":          timing,
                            "sources": [
                                {
                                    "title":          s.get("title", ""),
                                    "source":         s.get("source", ""),
                                    "score":          s.get("score", 0),
                                    "url":            s.get("url", ""),
                                    "project":        s.get("project", ""),
                                    "issue_key":      get_jira_key(s.get("url", ""), s.get("issue_key", "")),
                                    "status":         s.get("status", ""),
                                    "priority":       s.get("priority", ""),
                                    "assignee":       s.get("assignee", ""),
                                    "from_graph":     s.get("from_graph", False),
                                    "relationship":   s.get("relationship", ""),
                                    "graph_distance": s.get("graph_distance"),
                                }
                                for s in sources  # <-- REMOVED [:5] limit
                            ],
                            "context_snippets": [
                                {
                                    "title":        s.get("title", ""),
                                    "source":       s.get("source", ""),
                                    "text":         (s.get("text") or "")[:500],
                                    "from_graph":   s.get("from_graph", False),
                                    "relationship": s.get("relationship", ""),
                                }
                                for s in sources  # <-- REMOVED [:5] limit
                            ],
                            "sentries_summary": [
                                {
                                    "source":    getattr(r, "source", ""),
                                    "operation": getattr(r, "operation", ""),
                                    "count":     len(r.data),
                                    "titles": [
                                        item.get("title") or item.get("key") or item.get("id", "")
                                        for item in r.data[:5]
                                    ],
                                }
                                for r in (state.get("sentries_results") or [])
                                if getattr(r, "success", False) and getattr(r, "data", None)
                            ],
                            "error":           state.get("error"),
                            "timestamp":       datetime.utcnow(),
                            "total_time_s":    round(sum(timing.values()), 3),
                            "sentry_fallback": state.get("sentry_fallback", False),
                        })
                    except Exception:
                        pass

            # ── Send metadata then done ───────────────────────────────────
            yield _sse({
                "type":    "metadata",
                "intent":  intent,
                "sources": sources,
                "timing":  timing,
            })
            yield _sse({"type": "done"})

        except Exception as e:
            yield _sse({"type": "error", "content": str(e)})
            yield _sse({"type": "done"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Conversation history endpoints ────────────────────────────────────────────
@app.get("/conversations")
async def list_conversations():
    """Return distinct conversations from MongoDB."""
    collection = _get_mongo_collection()
    if collection is None:
        return {"conversations": []}

    try:
        pipeline = [
            {"$sort": {"timestamp": -1}},
            {"$group": {
                "_id": "$conversation_id",
                "first_query": {"$last": "$query"},
                "last_query":  {"$first": "$query"},
                "last_time":   {"$first": "$timestamp"},
                "count":       {"$sum": 1},
            }},
            {"$sort": {"last_time": -1}},
            {"$limit": 50},
        ]
        results = list(collection.aggregate(pipeline))
        return {
            "conversations": [
                {
                    "id":         str(r["_id"]),
                    "title":      r["first_query"][:60],
                    "last_query": r["last_query"][:60],
                    "last_time":  r["last_time"].isoformat() if r.get("last_time") else "",
                    "count":      r["count"],
                }
                for r in results
                if r["_id"]
            ]
        }
    except Exception as e:
        return {"conversations": [], "error": str(e)}


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Return all messages in a conversation."""
    collection = _get_mongo_collection()
    if collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not available")

    try:
        messages = list(collection.find(
            {"conversation_id": conversation_id},
            {"_id": 0}
        ).sort("timestamp", 1))

        for m in messages:
            if "timestamp" in m and hasattr(m["timestamp"], "isoformat"):
                m["timestamp"] = m["timestamp"].isoformat()

        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete all messages in a conversation."""
    collection = _get_mongo_collection()
    if collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not available")

    try:
        result = collection.delete_many({"conversation_id": conversation_id})
        return {"deleted": result.deleted_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversations/new")
async def new_conversation():
    """Clear in-memory context and return a new conversation ID."""
    _conversation_history.clear()
    return {"conversation_id": str(uuid.uuid4())}


@app.get("/health")
async def health():
    """Health check — also reports checkpointing status."""
    checkpointer = None
    return {
        "status":        "ok",
        "checkpointing": checkpointer is not None,
        "checkpointer":  type(checkpointer).__name__ if checkpointer else "none",
    }


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
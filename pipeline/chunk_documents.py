#!/usr/bin/env python3
"""
Text Chunking Module
Splits normalized documents into optimal chunks for embeddings and RAG
"""

import os
import gc
from datetime import datetime
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv
from typing import List

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
MONGO_DB  = os.getenv('MONGO_DB', 'knowledge_base')

client = MongoClient(MONGO_URI)
db     = client[MONGO_DB]

CHUNK_SIZE        = int(os.getenv('CHUNK_SIZE',        '512'))
CHUNK_OVERLAP     = int(os.getenv('CHUNK_OVERLAP',     '128'))
CURSOR_BATCH_SIZE = int(os.getenv('CURSOR_BATCH_SIZE', '10'))
MAX_CONTENT_CHARS = int(os.getenv('MAX_CONTENT_CHARS', '50000'))
FLUSH_EVERY       = int(os.getenv('FLUSH_EVERY',       '500'))


def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def chunk_text(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    chunk_chars   = CHUNK_SIZE * 4
    overlap_chars = CHUNK_OVERLAP * 4
    chunks: List[str] = []
    start    = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_chars
        if end < text_len:
            for needle in ('\n\n', '. ', ' '):
                pos = text.rfind(needle, start, end)
                if pos > start:
                    end = pos + (1 if needle == '. ' else 0)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap_chars
        if start <= 0:
            start = end
    return chunks


def chunk_all_documents():
    log("=" * 60)
    log("Starting Text Chunking")
    log("=" * 60)
    log(f"Chunk size:    {CHUNK_SIZE} tokens (~{CHUNK_SIZE * 4} chars)")
    log(f"Chunk overlap: {CHUNK_OVERLAP} tokens (~{CHUNK_OVERLAP * 4} chars)")
    log(f"Max content:   {MAX_CONTENT_CHARS:,} chars")
    log(f"Cursor batch:  {CURSOR_BATCH_SIZE} docs per round-trip")
    log(f"Flush every:   {FLUSH_EVERY} chunk ops")
    log("")

    total     = db.normalized_documents.count_documents({})
    remaining = db.normalized_documents.count_documents({"chunked_at": {"$exists": False}})

    log(f"Total:      {total:,}")
    log(f"Done:       {total - remaining:,} — skipping")
    log(f"To process: {remaining:,}")
    log("")

    if remaining == 0:
        log("Nothing to do.")
        _print_stats(total)
        return

    cursor = (
        db.normalized_documents
        .find(
            {"chunked_at": {"$exists": False}},
            # Only pull fields we need — keeps wire transfer small
            {
                "document_id": 1, "source": 1,       "source_type": 1,
                "project_id":  1, "project_name": 1, "group_name": 1,
                "title":       1, "content": 1,      "author": 1,
                "created_at":  1, "url": 1,           "labels": 1,
                "status":      1, "metadata": 1,
            }
        )
        .batch_size(CURSOR_BATCH_SIZE)
    )

    total_chunks = 0
    processed    = 0
    batch_ops: List[UpdateOne] = []

    def flush():
        if batch_ops:
            db.document_chunks.bulk_write(batch_ops, ordered=False)
            batch_ops.clear()

    for doc in cursor:
        doc_id = doc.get("document_id")
        if not doc_id:
            del doc
            gc.collect()
            continue

        try:
            title   = doc.get("title")   or ""
            content = doc.get("content") or ""
            content = content[:MAX_CONTENT_CHARS]
            text    = f"{title}\n\n{content}".strip()

            meta = {
                "source":       doc.get("source"),
                "source_type":  doc.get("source_type"),
                "project_id":   doc.get("project_id"),
                "project_name": doc.get("project_name"),
                "group_name":   doc.get("group_name"),
                "title":        title,
                "author":       doc.get("author"),
                "created_at":   doc.get("created_at"),
                "url":          doc.get("url"),
                "labels":       doc.get("labels") or [],
                "status":       doc.get("status"),
                "metadata":     doc.get("metadata") or {},
            }
            del doc, title, content

            chunks = chunk_text(text)
            del text

            if not chunks:
                db.normalized_documents.update_one(
                    {"document_id": doc_id},
                    {"$set": {"chunked_at": datetime.utcnow()}},
                )
                del meta
                gc.collect()
                continue

            n = len(chunks)

            for idx, chunk_str in enumerate(chunks):
                batch_ops.append(UpdateOne(
                    {"chunk_id": f"{doc_id}_chunk_{idx}"},
                    {"$set": {
                        "chunk_id":           f"{doc_id}_chunk_{idx}",
                        "parent_document_id": doc_id,
                        "chunk_index":        idx,
                        "total_chunks":       n,
                        "text":               chunk_str,
                        "token_count":        estimate_tokens(chunk_str),
                        "chunked_at":         datetime.utcnow(),
                        **meta,
                    }},
                    upsert=True,
                ))
                if len(batch_ops) >= FLUSH_EVERY:
                    flush()

            del chunks, meta

            flush()  # ensure this doc's chunks are on disk before marking done

            db.normalized_documents.update_one(
                {"document_id": doc_id},
                {"$set": {"chunked_at": datetime.utcnow()}},
            )

            total_chunks += n
            processed    += 1

            if processed % 1000 == 0:
                log(f"  ✓ {processed:,}/{remaining:,} "
                    f"({processed / remaining * 100:.1f}%) — "
                    f"{total_chunks:,} chunks written")

        except MemoryError:
            batch_ops.clear()
            log(f"  ✗ MemoryError on {doc_id} — skipping", "ERROR")

        except Exception as e:
            batch_ops.clear()
            log(f"  ✗ {type(e).__name__} on {doc_id}: {e}", "ERROR")

        finally:
            # Force GC on every single document — keeps fragmented memory
            # from accumulating across hundreds of thousands of iterations
            gc.collect()

    flush()  # final flush for anything remaining

    log("\nCreating indexes on document_chunks...")
    db.document_chunks.create_index([("chunk_id", 1)],          unique=True)
    db.document_chunks.create_index([("parent_document_id", 1)])
    db.document_chunks.create_index([("source", 1)])
    db.document_chunks.create_index([("project_id", 1)])
    db.document_chunks.create_index([("chunked_at", -1)])

    log("=" * 60)
    log("Chunking Complete")
    log("=" * 60)
    _print_stats(total)


def _print_stats(total_documents: int):
    total_chunks = db.document_chunks.count_documents({})
    log("\nChunk Statistics:")
    log(f"  Total documents: {total_documents:,}")
    log(f"  Total chunks:    {total_chunks:,}")
    if total_documents > 0:
        log(f"  Avg chunks/doc:  {total_chunks / total_documents:.2f}")
    log("  By source:")
    for source in ["gitlab", "jira", "confluence"]:
        n = db.document_chunks.count_documents({"source": source})
        if n:
            log(f"    {source}: {n:,}")
    samples = list(db.document_chunks.find({}, {"token_count": 1}).limit(10))
    if samples:
        sizes = [s["token_count"] for s in samples]
        log("\nSample chunk sizes (first 10):")
        log(f"  Min: {min(sizes)} tokens")
        log(f"  Max: {max(sizes)} tokens")
        log(f"  Avg: {sum(sizes) / len(sizes):.1f} tokens")


if __name__ == "__main__":
    chunk_all_documents()

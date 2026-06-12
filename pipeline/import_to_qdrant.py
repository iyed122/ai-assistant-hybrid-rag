#!/usr/bin/env python3
"""
import_to_qdrant.py  (local machine)
=====================================
Reads embedded_jira.jsonl + embedded_confluence.jsonl produced by
embed_pipeline_aws.py and imports them directly into your local Qdrant.

This is the file-based counterpart to migrate_to_qdrant.py (which reads
from MongoDB).  Both produce the same Qdrant payload schema so
qdrant_search.py and rag_generator.py work identically regardless of
which import path was used.

The chunk_id → Qdrant integer ID mapping is identical to migrate_to_qdrant.py
(MD5 truncated to 60 bits) so the two pipelines can co-exist in one
collection without ID collisions.

Environment variables (.env or shell)
──────────────────────────────────────
  QDRANT_HOST        localhost
  QDRANT_PORT        6333
  QDRANT_COLLECTION  knowledge_base
  BATCH_SIZE         200   (points per upsert call)
  EMBEDDED_FILES     comma-separated list of jsonl files to import
                     default: embedded_jira.jsonl,embedded_confluence.jsonl

Dependencies
────────────
  pip install qdrant-client python-dotenv
"""

import os
import json
import hashlib
from datetime import datetime
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠  qdrant-client not installed.  pip install qdrant-client")

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
QDRANT_HOST       = os.getenv("QDRANT_HOST",       "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT",   "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge_base")
BATCH_SIZE        = int(os.getenv("IMPORT_BATCH_SIZE", "200"))

_default_files = "embedded_jira.jsonl,embedded_confluence.jsonl"
EMBEDDED_FILES: List[str] = [
    os.path.join(BASE_DIR, f.strip())
    for f in os.getenv("EMBEDDED_FILES", _default_files).split(",")
    if f.strip()
]


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


# ── ID conversion (must match migrate_to_qdrant.py exactly) ───────────────────

def chunk_id_to_qdrant_id(chunk_id: str) -> int:
    """
    Deterministic string → uint64 via MD5 (60-bit).
    Identical to the function in migrate_to_qdrant.py — guarantees that
    the same chunk always gets the same Qdrant point ID regardless of
    which import path was used (MongoDB or file-based).
    """
    return int(hashlib.md5(chunk_id.encode()).hexdigest()[:15], 16)


# ── Collection management ──────────────────────────────────────────────────────

def detect_embedding_dim(files: List[str]) -> int:
    """Peek at the first embedded file to discover embedding dimension."""
    for path in files:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    emb = obj.get("embedding")
                    if emb and isinstance(emb, list):
                        return len(emb)
                except Exception:
                    pass
    log("Could not detect embedding dimension — defaulting to 768", "WARN")
    return 768  # Arctic-M default


def ensure_collection(client: QdrantClient, dim: int, recreate: bool = False) -> None:
    collections = {c.name for c in client.get_collections().collections}

    if QDRANT_COLLECTION in collections:
        if recreate:
            log(f"Deleting existing collection: {QDRANT_COLLECTION}")
            client.delete_collection(QDRANT_COLLECTION)
        else:
            log(f"Collection '{QDRANT_COLLECTION}' already exists — will upsert into it.")
            return

    log(f"Creating collection '{QDRANT_COLLECTION}'  dim={dim}  metric=Cosine")
    # on_disk=True keeps raw vectors on SSD instead of RAM.
    # ScalarQuantization(int8) compresses the search index ~4x (~5 GB instead of ~20 GB)
    # and keeps the compressed index in RAM for fast searches.
    # Required on 16 GB machines — without this Qdrant will thrash the SSD.
    from qdrant_client.models import (
        ScalarQuantization, ScalarQuantizationConfig, ScalarType
    )
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE,
            on_disk=True,
        ),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                always_ram=True,
            )
        ),
    )
    log(f"  ✓ Collection created (int8 quantization + on_disk vectors).")


# ── Import ─────────────────────────────────────────────────────────────────────

def import_file(client: QdrantClient, path: str, label: str) -> int:
    """Stream one embedded jsonl file into Qdrant."""
    if not os.path.exists(path):
        log(f"  File not found: {path} — skipping", "WARN")
        return 0

    # Count lines for progress reporting
    total = 0
    with open(path, "rb") as fh:
        for _ in fh:
            total += 1

    log(f"  {label}: {total:,} chunks to import")
    imported = 0
    skipped  = 0
    errors   = 0
    batch:   List[PointStruct] = []

    def flush():
        nonlocal imported, errors
        if not batch:
            return
        try:
            client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
            imported += len(batch)
            log(
                f"  {label}: {imported:,}/{total:,} "
                f"({imported / total * 100:.1f}%)"
            )
        except Exception as exc:
            log(f"  ✗ Upsert error: {exc}", "ERROR")
            errors += 1
        batch.clear()

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            chunk_id = doc.get("chunk_id", "")
            embedding = doc.get("embedding")

            if not chunk_id or not embedding:
                skipped += 1
                continue

            meta = doc.get("metadata") or {}

            point = PointStruct(
                id     = chunk_id_to_qdrant_id(chunk_id),
                vector = embedding,
                payload = {
                    # ── Core identification ────────────────────────────────
                    "chunk_id":           chunk_id,
                    "parent_document_id": doc.get("parent_document_id", ""),
                    "text":               doc.get("text", ""),
                    "source":             doc.get("source", ""),
                    "source_type":        doc.get("source_type", ""),
                    "project_id":         doc.get("project_id", ""),
                    "project_name":       doc.get("project_name", ""),
                    "group_name":         doc.get("group_name", ""),
                    "title":              doc.get("title", ""),
                    "author":             doc.get("author", ""),
                    "url":                doc.get("url", ""),
                    "labels":             doc.get("labels", []),
                    "status":             doc.get("status", ""),
                    "token_count":        doc.get("token_count", 0),
                    "created_at":         str(doc.get("created_at", "")),
                    # ── Jira relationship fields ───────────────────────────
                    "issue_key":          meta.get("issue_key",    ""),
                    "issue_type":         meta.get("issue_type",   ""),
                    "priority":           meta.get("priority",     ""),
                    "assignee":           meta.get("assignee",     ""),
                    "parent_key":         meta.get("parent_key",   ""),
                    "resolution":         meta.get("resolution",   ""),
                    "fix_versions":       meta.get("fix_versions", []),
                    "components":         meta.get("components",   []),
                    "issue_links":        meta.get("issue_links",  []),
                    # ── Confluence context fields ──────────────────────────
                    "breadcrumb":         meta.get("breadcrumb",       ""),
                    "ancestors":          meta.get("ancestors",        []),
                    "last_modifier_id":   meta.get("last_modifier_id", ""),
                    "space_id":           meta.get("space_id",         ""),
                    # ── GitLab fields (empty for Jira/Confluence) ──────────
                    "language":           meta.get("language",   ""),
                    "file_path":          meta.get("file_path",  ""),
                    "ref":                meta.get("ref",        ""),
                },
            )
            batch.append(point)

            if len(batch) >= BATCH_SIZE:
                flush()

    flush()  # final partial batch

    log(f"  {label}: ✓ {imported:,} imported  |  {skipped:,} skipped  |  {errors} errors")
    return imported


# ── Stats ──────────────────────────────────────────────────────────────────────

def show_stats(client: QdrantClient) -> None:
    log("")
    log("=" * 60)
    log("Qdrant Collection Stats")
    log("=" * 60)
    try:
        info = client.get_collection(QDRANT_COLLECTION)
        # FIX: vectors_count is deprecated in newer qdrant-client — use points_count
        count = (
            getattr(info, "points_count", None)
            or getattr(info, "vectors_count", None)
            or 0
        )
        log(f"  Collection : {QDRANT_COLLECTION}")
        log(f"  Points     : {count:,}")
        log(f"  Status     : {info.status}")
    except Exception as exc:
        log(f"  Could not retrieve stats: {exc}", "WARN")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not QDRANT_AVAILABLE:
        log("qdrant-client not installed.  pip install qdrant-client", "ERROR")
        return

    log("=" * 60)
    log("Qdrant Import Tool (file-based)")
    log("=" * 60)
    log(f"  Host:       {QDRANT_HOST}:{QDRANT_PORT}")
    log(f"  Collection: {QDRANT_COLLECTION}")
    log(f"  Batch size: {BATCH_SIZE}")
    log(f"  Files:")
    for f in EMBEDDED_FILES:
        exists = "✓" if os.path.exists(f) else "✗ MISSING"
        log(f"    {exists}  {f}")
    log("")

    # Connect
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections()
        log(f"✓ Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    except Exception as exc:
        log(f"✗ Cannot connect to Qdrant: {exc}", "ERROR")
        log("  Make sure Qdrant is running:", "INFO")
        log("    docker run -p 6333:6333 qdrant/qdrant", "INFO")
        return

    # Detect dimension
    dim = detect_embedding_dim(EMBEDDED_FILES)
    log(f"  Embedding dim: {dim}")
    log("")

    # Ask about collection recreation
    print("Options:")
    print("  1. Use existing collection (upsert — safe to re-run)")
    print("  2. Recreate collection (deletes all existing vectors!)")
    choice = input("\nChoice (1/2) [default 1]: ").strip() or "1"

    recreate = (choice == "2")
    if recreate:
        confirm = input("⚠  This will DELETE the existing collection. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            log("Cancelled.", "WARN")
            return

    log("")
    ensure_collection(client, dim, recreate=recreate)
    log("")

    # Import files
    total = 0
    labels = ["Jira", "Confluence"]
    for path, label in zip(EMBEDDED_FILES, labels):
        total += import_file(client, path, label)

    show_stats(client)

    log("")
    log("=" * 60)
    log(f"✓ Import complete — {total:,} points upserted")
    log("=" * 60)
    log("")
    log("Next steps:")
    log("  python qdrant_search.py search 'your query'")
    log("  python rag_generator.py")


if __name__ == "__main__":
    main()

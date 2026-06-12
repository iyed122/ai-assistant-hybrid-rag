#!/usr/bin/env python3
"""
run_pipeline.py  —  Master Knowledge Base Pipeline
===================================================
Runs the complete workflow end-to-end:

  1. Ingest      pipeline/ingestion_newest.py   GitLab + Jira + Confluence → MongoDB
  2. Normalize   pipeline/normalize_data.py     raw docs → normalized_documents
  3. Chunk       pipeline/chunk_documents.py    normalized → document_chunks
  4. Embed       pipeline/embed.py              document_chunks → embeddings in MongoDB
  5. Graph       rag/neo4j_import.py            MongoDB → Neo4j property graph

After the pipeline finishes:
  Search:   python -m rag.neo4j_search hybrid 'your query'
  Chat:     python -m rag.neo4j_rag chat

Environment variables (.env at project root)
────────────────────────────────────────────
  MONGO_URI, MONGO_DB              MongoDB connection
  NEO4J_URI, NEO4J_USER,
  NEO4J_PASSWORD, NEO4J_DATABASE   Neo4j connection
  IMPORT_SOURCE=mongo              neo4j_import reads from MongoDB
  SKIP_INGEST=false                set true to skip ingestion step
  SKIP_EMBED=false                 set true if embeddings are already done
"""

import os
import sys
import time
import subprocess
from datetime import datetime


def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def run_step(module: str, description: str, env_override: dict = None) -> bool:
    """
    Run a pipeline step as a module (python -m module).
    module:        e.g. 'pipeline.normalize_data'
    description:   human-readable label for logging
    env_override:  extra env vars passed only to this step
    """
    log("=" * 70)
    log(f"STEP: {description}")
    log("=" * 70)

    start = time.time()

    env = os.environ.copy()
    if env_override:
        env.update(env_override)

    try:
        result = subprocess.run(
            [sys.executable, "-m", module],
            env=env,
            check=True,
        )
        elapsed = time.time() - start
        log(f"✓ {description} completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")
        log("")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        log(f"✗ {description} FAILED after {elapsed:.1f}s", "ERROR")
        log(f"  Exit code: {e.returncode}", "ERROR")
        log("")
        return False

    except FileNotFoundError:
        log(f"✗ Module not found: {module}", "ERROR")
        log("")
        return False


def main():
    print()
    log("=" * 70)
    log("KNOWLEDGE BASE PIPELINE  —  Ingest → Normalize → Chunk → Embed → Graph")
    log("=" * 70)
    print()

    # Read skip flags
    skip_ingest = os.getenv("SKIP_INGEST", "false").lower() == "true"
    skip_embed  = os.getenv("SKIP_EMBED",  "false").lower() == "true"

    overall_start = time.time()
    results = {}

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    if not skip_ingest:
        ok = run_step(
            "pipeline.ingestion_newest",
            "Data Ingestion  (GitLab + Jira + Confluence → MongoDB)",
        )
        results["Ingestion"] = ok
        if not ok:
            log("Pipeline stopped: ingestion is required.", "ERROR")
            sys.exit(1)
    else:
        log("Skipping ingestion (SKIP_INGEST=true)")
        results["Ingestion"] = "skipped"

    # ── Step 2: Normalize ─────────────────────────────────────────────────────
    ok = run_step(
        "pipeline.normalize_data",
        "Data Normalization  (raw → normalized_documents)",
    )
    results["Normalization"] = ok
    if not ok:
        log("Pipeline stopped: normalization is required.", "ERROR")
        sys.exit(1)

    # ── Step 3: Chunk ─────────────────────────────────────────────────────────
    ok = run_step(
        "pipeline.chunk_documents",
        "Text Chunking  (normalized_documents → document_chunks)",
    )
    results["Chunking"] = ok
    if not ok:
        log("Pipeline stopped: chunking is required.", "ERROR")
        sys.exit(1)

    # ── Step 4: Embed ─────────────────────────────────────────────────────────
    if not skip_embed:
        ok = run_step(
            "pipeline.embed",
            "Embedding Generation  (document_chunks → vectors in MongoDB)",
        )
        results["Embedding"] = ok
        if not ok:
            log("Pipeline stopped: embeddings are required for Neo4j import.", "ERROR")
            sys.exit(1)
    else:
        log("Skipping embedding (SKIP_EMBED=true)")
        results["Embedding"] = "skipped"

    # ── Step 5: Neo4j Graph Import ────────────────────────────────────────────
    ok = run_step(
        "rag.neo4j_import",
        "Neo4j Graph Import  (MongoDB document_chunks → Neo4j property graph)",
        env_override={"IMPORT_SOURCE": "mongo"},   # always read from MongoDB here
    )
    results["Neo4j Import"] = ok
    if not ok:
        log("Neo4j import failed — check Neo4j is running and .env is correct.", "ERROR")
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - overall_start
    print()
    log("=" * 70)
    log("PIPELINE SUMMARY")
    log("=" * 70)
    for step, status in results.items():
        if status == "skipped":
            icon = "⏭"
        elif status:
            icon = "✓"
        else:
            icon = "✗"
        log(f"  {icon}  {step}")

    log("")
    log(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} minutes)")
    log("")
    log("=" * 70)
    log("✓ PIPELINE COMPLETE — Knowledge base is ready in Neo4j")
    log("=" * 70)
    print()
    log("Next steps:")
    log("  Search:  python -m rag.neo4j_search hybrid 'your query'")
    log("  Chat:    python -m rag.neo4j_rag chat")
    log("  Stats:   python -m rag.neo4j_search stats")
    print()


if __name__ == "__main__":
    main()

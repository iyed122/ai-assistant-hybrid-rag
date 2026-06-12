#!/usr/bin/env python3
"""
Qdrant Vector Store Migration
Migrates embeddings from MongoDB to Qdrant for superior semantic search
"""

import os
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import List, Dict, Any
import numpy as np
import hashlib


def _chunk_id_to_qdrant_id(chunk_id: str) -> int:
    """
    Convert a string chunk_id to a deterministic Qdrant integer ID.
    Using MD5 truncated to 15 hex digits (60-bit) stays well within
    Qdrant's uint64 range and gives negligible collision probability
    across any realistic document collection.
    Same chunk_id always produces the same integer — guarantees upsert
    semantics on re-migration instead of creating duplicate vectors.
    """
    return int(hashlib.md5(chunk_id.encode()).hexdigest()[:15], 16)

# Try to import dependencies
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠️  qdrant-client not installed. Run: pip install qdrant-client")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed")

load_dotenv()

# Configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
MONGO_DB = os.getenv('MONGO_DB', 'knowledge_base')
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))
QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION', 'knowledge_base')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'Snowflake/snowflake-arctic-embed-m')
# Embedding dimension — read dynamically from a sample doc so it's model-agnostic.
# Falls back to 768 (Arctic-M default) if no embedded chunks exist yet.
def _detect_embedding_dim() -> int:
    try:
        sample = db.document_chunks.find_one({"embedding": {"$exists": True}}, {"embedding": 1})
        if sample and sample.get("embedding"):
            return len(sample["embedding"])
    except Exception:
        pass
    return 768  # Arctic-M default

EMBEDDING_DIM = _detect_embedding_dim()

# MongoDB connection
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB]


def log(message: str, level: str = "INFO"):
    """Simple logger"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


class QdrantMigrator:
    """Handles migration from MongoDB to Qdrant"""
    
    def __init__(self):
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client not installed")
        
        log(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = QDRANT_COLLECTION
        
    def create_collection(self, recreate: bool = False):
        """Create Qdrant collection with proper configuration"""
        log("=" * 60)
        log("Setting up Qdrant Collection")
        log("=" * 60)
        
        # Check if collection exists
        collections = self.qdrant.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if exists:
            if recreate:
                log(f"Deleting existing collection: {self.collection_name}")
                self.qdrant.delete_collection(self.collection_name)
            else:
                log(f"Collection '{self.collection_name}' already exists")
                return
        
        # Create new collection
        log(f"Creating collection: {self.collection_name}")
        log(f"Vector dimension: {EMBEDDING_DIM}")
        log(f"Distance metric: Cosine")
        
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
        
        log(f"✓ Collection '{self.collection_name}' created successfully")
    
    def migrate_from_mongodb(self, batch_size: int = 100):
        """Migrate all embeddings from MongoDB to Qdrant"""
        log("=" * 60)
        log("Migrating Data from MongoDB to Qdrant")
        log("=" * 60)
        
        # Get all chunks with embeddings
        total_chunks = db.document_chunks.count_documents({"embedding": {"$exists": True}})
        
        if total_chunks == 0:
            log("No embedded chunks found in MongoDB!", "ERROR")
            log("Run 'python embed.py' first to generate embeddings")
            return
        
        log(f"Found {total_chunks} chunks with embeddings")

        # Resume detection — count points already in Qdrant collection
        try:
            existing_info  = self.qdrant.get_collection(self.collection_name)
            already_in_qdrant = getattr(existing_info, 'vectors_count', None)                                or getattr(existing_info, 'points_count', 0) or 0
        except Exception:
            already_in_qdrant = 0

        if already_in_qdrant > 0:
            log(f"Qdrant already has {already_in_qdrant:,} points — resuming from where we left off")
            log(f"(upsert semantics: safe to re-run, no duplicates will be created)")
        else:
            log(f"Fresh migration starting")

        log(f"Migrating in batches of {batch_size}...")
        log("")

        migrated = 0
        batch_points = []

        for chunk in db.document_chunks.find({"embedding": {"$exists": True}}):
            try:
                # Create Qdrant point
                # Pull rich metadata from parent normalized_document for
                # Dispatcher filtering. These fields let the RAG layer answer
                # "show me critical issues blocking v2.3" without a re-fetch.
                meta = chunk.get('metadata', {}) or {}

                point = PointStruct(
                    id=_chunk_id_to_qdrant_id(chunk['chunk_id']),
                    vector=chunk['embedding'],
                    payload={
                        # ── Core identification ────────────────────────────────
                        "chunk_id":            chunk['chunk_id'],
                        "parent_document_id":  chunk['parent_document_id'],
                        "text":                chunk['text'],
                        "source":              chunk.get('source', ''),
                        "source_type":         chunk.get('source_type', ''),
                        "project_id":          chunk.get('project_id', ''),
                        "project_name":        chunk.get('project_name', ''),
                        "group_name":          chunk.get('group_name', ''),
                        "title":               chunk.get('title', ''),
                        "author":              chunk.get('author', ''),
                        "url":                 chunk.get('url', ''),
                        "labels":              chunk.get('labels', []),
                        "status":              chunk.get('status', ''),
                        "token_count":         chunk.get('token_count', 0),
                        "created_at":          str(chunk.get('created_at', '')),
                        # ── Jira relationship fields ───────────────────────────
                        "issue_key":           meta.get('issue_key', ''),
                        "issue_type":          meta.get('issue_type', ''),
                        "priority":            meta.get('priority', ''),
                        "assignee":            meta.get('assignee', ''),
                        "parent_key":          meta.get('parent_key', ''),
                        "resolution":          meta.get('resolution', ''),
                        "fix_versions":        meta.get('fix_versions', []),
                        "components":          meta.get('components', []),
                        "issue_links":         meta.get('issue_links', []),
                        # ── Confluence context fields ──────────────────────────
                        "breadcrumb":          meta.get('breadcrumb', ''),
                        "ancestors":           meta.get('ancestors', []),
                        "last_modifier_id":    meta.get('last_modifier_id', ''),
                        "space_id":            meta.get('space_id', ''),
                        # ── GitLab fields ──────────────────────────────────────
                        "language":            meta.get('language', ''),
                        "file_path":           meta.get('file_path', ''),
                        "ref":                 meta.get('ref', ''),
                    }
                )
                
                batch_points.append(point)
                migrated += 1
                
                # Upload batch when full
                if len(batch_points) >= batch_size:
                    self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=batch_points
                    )
                    log(f"  ✓ Migrated {migrated:,}/{total_chunks:,} chunks ({migrated/total_chunks*100:.1f}%)")
                    batch_points = []
            
            except Exception as e:
                log(f"  ✗ Error migrating chunk {chunk.get('chunk_id')}: {e}", "ERROR")
        
        # Upload remaining chunks
        if batch_points:
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=batch_points
            )
            log(f"  ✓ Migrated {migrated}/{total_chunks} chunks (100%)")
        
        log("=" * 60)
        log(f"Migration Complete: {migrated} chunks in Qdrant")
        log("=" * 60)
        
        # Show collection info
        self.show_collection_info()
    
    def show_collection_info(self):
        """Display Qdrant collection statistics"""
        log("\nQdrant Collection Info:")
        
        info = self.qdrant.get_collection(self.collection_name)
        log(f"  Collection: {self.collection_name}")
        
        # Handle different API versions
        vector_count = getattr(info, 'vectors_count', None) or getattr(info, 'points_count', 0)
        log(f"  Vectors: {vector_count}")
        log(f"  Status: {info.status}")


def test_qdrant_connection():
    """Test Qdrant connection"""
    log("=" * 60)
    log("Testing Qdrant Connection")
    log("=" * 60)
    
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = client.get_collections()
        
        log(f"✓ Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        log(f"✓ Existing collections: {len(collections.collections)}")
        
        for coll in collections.collections:
            log(f"  - {coll.name}")
        
        return True
    
    except Exception as e:
        log(f"✗ Connection failed: {e}", "ERROR")
        log("\nMake sure Qdrant is running:", "WARN")
        log("  Docker: docker run -p 6333:6333 qdrant/qdrant", "WARN")
        log("  Or install locally: https://qdrant.tech/documentation/quick-start/", "WARN")
        return False


def main():
    """Main migration workflow"""
    log("=" * 60)
    log("Qdrant Migration Tool")
    log("=" * 60)
    print()
    
    if not QDRANT_AVAILABLE:
        log("Please install qdrant-client:", "ERROR")
        log("  pip install qdrant-client", "INFO")
        return
    
    # Test connection
    if not test_qdrant_connection():
        return
    
    print()
    
    # Create migrator
    migrator = QdrantMigrator()
    
    # Ask user about recreation
    print("Options:")
    print("  1. Create new collection (fails if exists)")
    print("  2. Recreate collection (deletes existing)")
    print("  3. Use existing collection")
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice == "1":
        migrator.create_collection(recreate=False)
    elif choice == "2":
        migrator.create_collection(recreate=True)
    elif choice == "3":
        log("Using existing collection")
    else:
        log("Invalid choice", "ERROR")
        return
    
    print()
    
    # Migrate data
    migrator.migrate_from_mongodb()
    
    log("\n" + "=" * 60)
    log("✓ Setup Complete!")
    log("=" * 60)
    log("\nNext steps:")
    log("  1. Test search: python qdrant_search.py search 'your query'")
    log("  2. Use hybrid search for best results")
    log("  3. Monitor performance in Qdrant dashboard: http://localhost:6333/dashboard")


if __name__ == "__main__":
    main()

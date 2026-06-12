
#!/usr/bin/env python3
"""
Embedding Pipeline
Generates vector embeddings for document chunks using sentence-transformers
"""

import os
import sys
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import List, Dict, Any

# Try to import numpy first
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️  numpy not installed")

# Try to import sentence-transformers with better error handling
EMBEDDINGS_AVAILABLE = False
IMPORT_ERROR = None

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except Exception as e:
    IMPORT_ERROR = str(e)
    # Try to give helpful error message
    if "torch" in str(e).lower():
        print(f"⚠️  PyTorch compatibility issue: {e}")
        print("Try: pip uninstall torch torchvision -y && pip install torch torchvision")
    elif "transformers" in str(e).lower():
        print(f"⚠️  Transformers library issue: {e}")
        print("Try: pip install --upgrade transformers sentence-transformers")
    else:
        print(f"⚠️  sentence-transformers import error: {e}")
        print("Try: pip install --force-reinstall sentence-transformers")

load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
MONGO_DB = os.getenv('MONGO_DB', 'knowledge_base')

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

# Embedding configuration - Using Snowflake Arctic Embed (Best in class!)
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'Snowflake/snowflake-arctic-embed-m')
BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '64'))  # Arctic-M handles 64 comfortably

# Arctic embed specific settings
ARCTIC_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ARCTIC_MAX_LENGTH = 512  # Arctic's optimal sequence length


def log(message: str, level: str = "INFO"):
    """Simple logger"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


class EmbeddingGenerator:
    """Generates embeddings for text chunks using Snowflake Arctic Embed"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers not installed")
        
        log(f"Loading embedding model: {model_name}")
        log("Using Snowflake Arctic Embed - State-of-the-art retrieval model")
        
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.model_name = model_name
        
        # Set max sequence length for Arctic
        self.model.max_seq_length = ARCTIC_MAX_LENGTH
        
        log(f"✓ Model loaded successfully")
        log(f"  Embedding dimension: {self.embedding_dim}")
        log(f"  Max sequence length: {ARCTIC_MAX_LENGTH}")
    
    def generate_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            is_query: If True, adds Arctic's query prefix for better retrieval
        
        Returns:
            Embedding vector as list of floats
        """
        # For Arctic embed, queries should use a specific prefix
        if is_query and "arctic" in self.model_name.lower():
            text = ARCTIC_QUERY_PREFIX + text
        
        embedding = self.model.encode(
            text, 
            convert_to_numpy=True,
            normalize_embeddings=True  # Arctic performs better with normalized embeddings
        )
        return embedding.tolist()
    
    def generate_embeddings_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts to embed
            is_query: If True, treats as queries (adds prefix for Arctic)
        
        Returns:
            List of embedding vectors
        """
        # Add query prefix if needed
        if is_query and "arctic" in self.model_name.lower():
            texts = [ARCTIC_QUERY_PREFIX + text for text in texts]
        
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True, 
            show_progress_bar=False,
            normalize_embeddings=True,  # Arctic performs better with normalized embeddings
            batch_size=BATCH_SIZE
        )
        return embeddings.tolist()


def embed_all_chunks():
    """Generate embeddings for all document chunks"""
    log("=" * 60)
    log("Starting Embedding Generation")
    log("=" * 60)
    
    if not EMBEDDINGS_AVAILABLE:
        log("sentence-transformers not installed. Run: pip install sentence-transformers", "ERROR")
        log("\nFor now, showing what would be processed...\n", "INFO")
        show_embedding_preview()
        return
    
    # Initialize embedding generator
    try:
        generator = EmbeddingGenerator()
    except Exception as e:
        log(f"Failed to initialize embedding model: {e}", "ERROR")
        return
    
    # Get total chunks
    total_chunks = db.document_chunks.count_documents({})
    log(f"Total chunks to embed: {total_chunks}")
    
    # Count chunks already embedded
    embedded_count = db.document_chunks.count_documents({"embedding": {"$exists": True}})
    remaining_count = total_chunks - embedded_count
    
    if embedded_count > 0:
        log(f"Already embedded: {embedded_count}")
        log(f"Remaining: {remaining_count}")
    
    if remaining_count == 0:
        log("All chunks already have embeddings!", "INFO")
        show_embedding_statistics()
        return
    
    log(f"Processing in batches of {BATCH_SIZE}...")
    log("")
    
    # Process chunks in batches
    processed    = 0
    batch_texts  = []
    batch_ids    = []
    _embed_start = datetime.utcnow()
    
    # Query only chunks without embeddings
    query = {"embedding": {"$exists": False}}
    
    for chunk in db.document_chunks.find(query):
        batch_texts.append(chunk['text'])
        batch_ids.append(chunk['chunk_id'])
        
        # Process batch when full
        if len(batch_texts) >= BATCH_SIZE:
            try:
                embeddings = generator.generate_embeddings_batch(batch_texts)
                
                # Bulk write — one MongoDB roundtrip per batch instead of one per chunk
                from pymongo import UpdateOne as _UOne
                ops = [
                    _UOne(
                        {"chunk_id": cid},
                        {"$set": {
                            "embedding":       emb,
                            "embedding_model": EMBEDDING_MODEL,
                            "embedding_dim":   len(emb),
                            "embedded_at":     datetime.utcnow(),
                        }}
                    )
                    for cid, emb in zip(batch_ids, embeddings)
                ]
                db.document_chunks.bulk_write(ops, ordered=False)

                processed += len(batch_texts)
                elapsed   = (datetime.utcnow() - _embed_start).total_seconds()
                rate      = processed / elapsed if elapsed > 0 else 0
                eta_s     = (remaining_count - processed) / rate if rate > 0 else 0
                log(f"  ✓ {processed}/{remaining_count} ({processed/remaining_count*100:.1f}%) "
                    f"| {rate:.0f} chunks/s | ETA ~{eta_s/60:.0f}m")

                batch_texts = []
                batch_ids   = []

            except Exception as e:
                log(f"  ✗ Error processing batch: {e}", "ERROR")
                batch_texts = []
                batch_ids   = []

    # Process remaining chunks
    if batch_texts:
        try:
            embeddings = generator.generate_embeddings_batch(batch_texts)

            from pymongo import UpdateOne as _UOne
            ops = [
                _UOne(
                    {"chunk_id": cid},
                    {"$set": {
                        "embedding":       emb,
                        "embedding_model": EMBEDDING_MODEL,
                        "embedding_dim":   len(emb),
                        "embedded_at":     datetime.utcnow(),
                    }}
                )
                for cid, emb in zip(batch_ids, embeddings)
            ]
            db.document_chunks.bulk_write(ops, ordered=False)

            processed += len(batch_texts)
            log(f"  ✓ {processed}/{remaining_count} chunks (100%)")

        except Exception as e:
            log(f"  ✗ Error processing final batch: {e}", "ERROR")
    
    log("=" * 60)
    log("Embedding Generation Complete")
    log("=" * 60)
    
    show_embedding_statistics()


def show_embedding_statistics():
    """Show statistics about embeddings"""
    log("\nEmbedding Statistics:")
    
    total_chunks = db.document_chunks.count_documents({})
    embedded_chunks = db.document_chunks.count_documents({"embedding": {"$exists": True}})
    
    log(f"  Total chunks: {total_chunks}")
    log(f"  Embedded chunks: {embedded_chunks}")
    log(f"  Coverage: {(embedded_chunks/total_chunks)*100:.1f}%")
    
    # Get a sample embedding to show dimensions
    sample = db.document_chunks.find_one({"embedding": {"$exists": True}})
    if sample:
        log(f"  Embedding model: {sample.get('embedding_model', 'unknown')}")
        log(f"  Embedding dimension: {sample.get('embedding_dim', 'unknown')}")
    
    # Breakdown by source
    log("\nEmbeddings by source:")
    for source in ['gitlab', 'jira', 'confluence']:
        count = db.document_chunks.count_documents({
            'source': source,
            'embedding': {'$exists': True}
        })
        if count > 0:
            log(f"  {source}: {count}")


def show_embedding_preview():
    """Show what would be embedded (when sentence-transformers not installed)"""
    log("=" * 60)
    log("Embedding Preview (No Model Loaded)")
    log("=" * 60)
    
    total_chunks = db.document_chunks.count_documents({})
    log(f"Total chunks ready for embedding: {total_chunks}")
    
    log("\nBreakdown by source:")
    for source in ['gitlab', 'jira', 'confluence']:
        count = db.document_chunks.count_documents({'source': source})
        if count > 0:
            log(f"  {source}: {count} chunks")
    
    log("\nSample chunks to be embedded:")
    for i, chunk in enumerate(db.document_chunks.find({}).limit(3), 1):
        log(f"\n  Sample {i}:")
        log(f"    ID: {chunk['chunk_id']}")
        log(f"    Source: {chunk['source']} ({chunk['source_type']})")
        log(f"    Tokens: {chunk['token_count']}")
        log(f"    Preview: {chunk['text'][:100]}...")
    
    log("\n" + "=" * 60)
    log("Install sentence-transformers to generate embeddings:")
    log("  pip install sentence-transformers")
    log("=" * 60)


def test_embedding_search(query: str, top_k: int = 5):
    """
    Prototype semantic search — loads all embeddings into RAM.

    ⚠ WARNING: At 80k+ chunks with 768-dim vectors this loads ~1.2GB into
    Python memory and will be very slow or OOM. Use this for local dev /
    small datasets only. For production search, use Qdrant via migrate_to_qdrant.py.
    """
    if not EMBEDDINGS_AVAILABLE:
        log("sentence-transformers not installed", "ERROR")
        return

    if not NUMPY_AVAILABLE:
        log("numpy not installed", "ERROR")
        return

    total_embedded = db.document_chunks.count_documents({"embedding": {"$exists": True}})
    if total_embedded > 10_000:
        log(f"⚠ IN-MEMORY SEARCH on {total_embedded:,} chunks — this will be slow and memory-heavy.", "WARN")
        log("  Use Qdrant for production search (run migrate_to_qdrant.py first).", "WARN")

    log(f"\n🔍 Semantic Search Query: '{query}'")
    log("=" * 60)

    # Generate query embedding with Arctic query prefix
    generator = EmbeddingGenerator()
    query_embedding = generator.generate_embedding(query, is_query=True)

    # Get all chunks with embeddings
    chunks = list(db.document_chunks.find({"embedding": {"$exists": True}}))
    
    if not chunks:
        log("No embedded chunks found!", "ERROR")
        log("Run 'python embed_chunks.py' first to generate embeddings.")
        return
    
    log(f"Searching across {len(chunks)} embedded chunks...")
    
    # Calculate cosine similarity (embeddings are already normalized)
    similarities = []
    for chunk in chunks:
        chunk_embedding = np.array(chunk['embedding'])
        query_vec = np.array(query_embedding)
        
        # Cosine similarity (dot product for normalized vectors)
        similarity = np.dot(query_vec, chunk_embedding)
        
        similarities.append((chunk, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Show top results
    log(f"\n📊 Top {top_k} Results:")
    log("=" * 60)
    
    for i, (chunk, score) in enumerate(similarities[:top_k], 1):
        # Color coding for score
        if score > 0.7:
            relevance = "🟢 Highly Relevant"
        elif score > 0.5:
            relevance = "🟡 Relevant"
        else:
            relevance = "🔴 Somewhat Relevant"
        
        log(f"\n{i}. {relevance} (Score: {score:.4f})")
        log(f"   Source: [{chunk['source'].upper()}] {chunk['source_type']}")
        log(f"   Title: {chunk['title']}")
        log(f"   Preview: {chunk['text'][:250]}...")
        log(f"   URL: {chunk.get('url', 'N/A')}")
        log(f"   Project: {chunk.get('project_name', 'N/A')}")
    
    log("\n" + "=" * 60)
    log(f"✓ Search complete. Found {len(chunks)} total chunks.")


def re_embed_all_chunks():
    """
    Re-embed all chunks with the current model
    Useful when switching from one embedding model to another
    """
    log("=" * 60)
    log("Re-embedding All Chunks with New Model")
    log("=" * 60)
    log(f"Model: {EMBEDDING_MODEL}")
    log("")
    
    if not EMBEDDINGS_AVAILABLE:
        log("sentence-transformers not installed", "ERROR")
        return
    
    # Initialize embedding generator
    try:
        generator = EmbeddingGenerator()
    except Exception as e:
        log(f"Failed to initialize embedding model: {e}", "ERROR")
        return
    
    # Get all chunks (including already embedded ones)
    total_chunks = db.document_chunks.count_documents({})
    log(f"Total chunks to re-embed: {total_chunks}")
    log(f"Processing in batches of {BATCH_SIZE}...")
    log("")
    
    # Process all chunks
    processed = 0
    batch_texts = []
    batch_ids = []
    
    for chunk in db.document_chunks.find({}):
        batch_texts.append(chunk['text'])
        batch_ids.append(chunk['chunk_id'])
        
        # Process batch when full
        if len(batch_texts) >= BATCH_SIZE:
            try:
                embeddings = generator.generate_embeddings_batch(batch_texts)
                
                from pymongo import UpdateOne as _UOne
                ops = [
                    _UOne(
                        {"chunk_id": cid},
                        {"$set": {
                            "embedding":       emb,
                            "embedding_model": EMBEDDING_MODEL,
                            "embedding_dim":   len(emb),
                            "embedded_at":     datetime.utcnow(),
                        }}
                    )
                    for cid, emb in zip(batch_ids, embeddings)
                ]
                db.document_chunks.bulk_write(ops, ordered=False)

                processed += len(batch_texts)
                log(f"  ✓ Re-embedded {processed}/{total_chunks} ({processed/total_chunks*100:.1f}%)")

                batch_texts = []
                batch_ids   = []

            except Exception as e:
                log(f"  ✗ Error processing batch: {e}", "ERROR")
                batch_texts = []
                batch_ids   = []

    # Process remaining chunks
    if batch_texts:
        try:
            embeddings = generator.generate_embeddings_batch(batch_texts)

            from pymongo import UpdateOne as _UOne
            ops = [
                _UOne(
                    {"chunk_id": cid},
                    {"$set": {
                        "embedding":       emb,
                        "embedding_model": EMBEDDING_MODEL,
                        "embedding_dim":   len(emb),
                        "embedded_at":     datetime.utcnow(),
                    }}
                )
                for cid, emb in zip(batch_ids, embeddings)
            ]
            db.document_chunks.bulk_write(ops, ordered=False)

            processed += len(batch_texts)
            log(f"  ✓ Re-embedded {processed}/{total_chunks} chunks (100%)")

        except Exception as e:
            log(f"  ✗ Error processing final batch: {e}", "ERROR")
    
    log("=" * 60)
    log("Re-embedding Complete!")
    log("=" * 60)
    
    show_embedding_statistics()


def advanced_search(query: str, top_k: int = 10, filters: dict = None):
    """
    Prototype advanced search with MongoDB filters.

    ⚠ WARNING: Loads all matching embeddings into RAM — fine for small filtered
    subsets, dangerous on large ones. Prefer Qdrant with payload filters for
    production (migrate_to_qdrant.py populates all filter fields in payload).
    """
    if not EMBEDDINGS_AVAILABLE:
        log("sentence-transformers not installed", "ERROR")
        return []

    if not NUMPY_AVAILABLE:
        log("numpy not installed", "ERROR")
        return []

    log(f"\n🔍 Advanced Search: '{query}'")
    if filters:
        log(f"Filters: {filters}")
    log("=" * 60)
    
    # Build query
    mongo_query = {"embedding": {"$exists": True}}
    if filters:
        mongo_query.update(filters)
    
    # Generate query embedding
    generator = EmbeddingGenerator()
    query_embedding = generator.generate_embedding(query, is_query=True)
    
    # Get filtered chunks
    chunks = list(db.document_chunks.find(mongo_query))
    
    if not chunks:
        log("No chunks found matching filters!", "WARN")
        return []
    
    log(f"Searching across {len(chunks)} filtered chunks...")
    
    # Calculate similarities
    similarities = []
    for chunk in chunks:
        chunk_embedding = np.array(chunk['embedding'])
        query_vec = np.array(query_embedding)
        similarity = np.dot(query_vec, chunk_embedding)
        similarities.append((chunk, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Show results
    log(f"\n📊 Top {min(top_k, len(similarities))} Results:")
    log("=" * 60)
    
    results = []
    for i, (chunk, score) in enumerate(similarities[:top_k], 1):
        if score > 0.7:
            relevance = "🟢 Highly Relevant"
        elif score > 0.5:
            relevance = "🟡 Relevant"
        else:
            relevance = "🔴 Somewhat Relevant"
        
        log(f"\n{i}. {relevance} (Score: {score:.4f})")
        log(f"   Source: [{chunk['source'].upper()}] {chunk['source_type']}")
        log(f"   Title: {chunk['title'][:60]}...")
        log(f"   Project: {chunk.get('project_name', 'N/A')}")
        
        results.append({
            'chunk': chunk,
            'score': score,
            'rank': i
        })
    
    log("\n" + "=" * 60)
    return results


def compare_queries(queries: List[str], top_k: int = 3):
    """
    Compare multiple queries side by side
    Useful for testing different query formulations
    
    Args:
        queries: List of query strings to compare
        top_k: Number of results per query
    """
    if not EMBEDDINGS_AVAILABLE:
        log("sentence-transformers not installed", "ERROR")
        return
    
    log("\n" + "=" * 60)
    log("Query Comparison")
    log("=" * 60)
    
    generator = EmbeddingGenerator()
    
    for i, query in enumerate(queries, 1):
        log(f"\n{'='*60}")
        log(f"Query {i}: '{query}'")
        log(f"{'='*60}")
        test_embedding_search(query, top_k=top_k)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "search":
            # Semantic search
            if len(sys.argv) < 3:
                print("Usage: python embed_chunks.py search 'your query here'")
                print("Example: python embed_chunks.py search 'authentication bug'")
                sys.exit(1)
            
            query = " ".join(sys.argv[2:])
            test_embedding_search(query)
        
        elif command == "advanced":
            # Advanced search with filters
            if len(sys.argv) < 3:
                print("Usage: python embed_chunks.py advanced 'query' [--source gitlab] [--project myproject]")
                print("Example: python embed_chunks.py advanced 'security' --source gitlab")
                sys.exit(1)
            
            query = sys.argv[2]
            filters = {}
            
            # Parse filters
            i = 3
            while i < len(sys.argv):
                if sys.argv[i] == "--source" and i+1 < len(sys.argv):
                    filters['source'] = sys.argv[i+1]
                    i += 2
                elif sys.argv[i] == "--project" and i+1 < len(sys.argv):
                    filters['project_name'] = sys.argv[i+1]
                    i += 2
                elif sys.argv[i] == "--type" and i+1 < len(sys.argv):
                    filters['source_type'] = sys.argv[i+1]
                    i += 2
                else:
                    i += 1
            
            advanced_search(query, filters=filters if filters else None)
        
        elif command == "compare":
            # Compare multiple queries
            if len(sys.argv) < 4:
                print("Usage: python embed_chunks.py compare 'query1' 'query2' ['query3' ...]")
                print("Example: python embed_chunks.py compare 'auth bug' 'authentication error' 'login issue'")
                sys.exit(1)
            
            queries = sys.argv[2:]
            compare_queries(queries)
        
        elif command == "re-embed":
            # Re-embed all chunks with current model
            print("⚠️  This will re-embed ALL chunks with the current model.")
            print(f"Current model: {EMBEDDING_MODEL}")
            response = input("Continue? (yes/no): ")
            
            if response.lower() in ['yes', 'y']:
                re_embed_all_chunks()
            else:
                print("Cancelled.")
        
        elif command == "stats":
            # Show statistics only
            show_embedding_statistics()
        
        elif command == "help":
            print("\n🔍 Knowledge Base Embedding Tool - Snowflake Arctic Embed")
            print("=" * 60)
            print("\nCommands:")
            print("  python embed_chunks.py")
            print("    → Generate embeddings for all un-embedded chunks")
            print()
            print("  python embed_chunks.py search 'your query'")
            print("    → Semantic search across all chunks")
            print()
            print("  python embed_chunks.py advanced 'query' [--source X] [--project Y] [--type Z]")
            print("    → Advanced search with filters")
            print("    Example: python embed_chunks.py advanced 'security' --source gitlab")
            print()
            print("  python embed_chunks.py compare 'query1' 'query2' 'query3'")
            print("    → Compare multiple query formulations")
            print()
            print("  python embed_chunks.py re-embed")
            print("    → Re-embed all chunks (useful when changing models)")
            print()
            print("  python embed_chunks.py stats")
            print("    → Show embedding statistics")
            print()
            print("Current Model: " + EMBEDDING_MODEL)
            print("Model: Snowflake Arctic Embed-M (State-of-the-art retrieval)")
            print("=" * 60)
        
        else:
            print(f"Unknown command: {command}")
            print("Run 'python embed_chunks.py help' for usage information")
    
    else:
        # Normal embedding generation mode
        embed_all_chunks()
#!/usr/bin/env python3
"""
Qdrant Semantic Search
High-performance semantic search using Qdrant vector database
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any
import sys

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchRequest
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
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))
QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION', 'knowledge_base')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'Snowflake/snowflake-arctic-embed-m')
ARCTIC_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def log(message: str, level: str = "INFO"):
    """Simple logger"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


class QdrantSearch:
    """Qdrant-powered semantic search"""
    
    def __init__(self):
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client not installed")
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers not installed")
        
        # Connect to Qdrant
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = QDRANT_COLLECTION
        
        # Load embedding model
        log(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.model.max_seq_length = 512
        log("✓ Model loaded")
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        # Add Arctic query prefix for better retrieval
        prefixed_query = ARCTIC_QUERY_PREFIX + query
        
        embedding = self.model.encode(
            prefixed_query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.tolist()
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Dict[str, Any] = None,
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search with optional filtering
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Dict of filters (e.g., {'source': 'gitlab'})
            score_threshold: Minimum similarity score (0.0-1.0)
        
        Returns:
            List of search results
        """
        log(f"\n🔍 Searching: '{query}'")
        if filters:
            log(f"Filters: {filters}")
        log("=" * 60)
        
        # Generate query embedding
        query_vector = self.embed_query(query)
        
        # Build Qdrant filter
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)
        
        # Search in Qdrant
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
            with_payload=True
        ).points
        
        log(f"Found {len(results)} results")
        
        # Format results
        formatted_results = []
        for i, hit in enumerate(results, 1):
            # Determine relevance level
            score = hit.score
            if score > 0.7:
                relevance = "🟢 Highly Relevant"
            elif score > 0.5:
                relevance = "🟡 Relevant"
            else:
                relevance = "🔴 Somewhat Relevant"
            
            payload = hit.payload
            
            log(f"\n{i}. {relevance} (Score: {score:.4f})")
            log(f"   Source: [{payload['source'].upper()}] {payload['source_type']}")
            log(f"   Title: {payload['title'][:70]}...")
            log(f"   Preview: {payload['text'][:200]}...")
            log(f"   URL: {payload.get('url', 'N/A')}")
            log(f"   Project: {payload.get('project_name', 'N/A')}")
            
            formatted_results.append({
                'score': score,
                'payload': payload,
                'rank': i
            })
        
        log("\n" + "=" * 60)
        return formatted_results
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic and keyword matching
        
        Args:
            query: Search query
            top_k: Number of results
            keyword_weight: Weight for keyword matching (0.0-1.0)
            semantic_weight: Weight for semantic matching (0.0-1.0)
        
        Returns:
            List of reranked results
        """
        log(f"\n🔍 Hybrid Search: '{query}'")
        log(f"Semantic weight: {semantic_weight:.1f}, Keyword weight: {keyword_weight:.1f}")
        log("=" * 60)
        
        # Get semantic results
        query_vector = self.embed_query(query)
        semantic_results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k * 2,  # Get more for reranking
            with_payload=True
        ).points
        
        # Rerank with keyword boost
        query_terms = set(query.lower().split())
        reranked = []
        
        for hit in semantic_results:
            semantic_score = hit.score
            text = hit.payload['text'].lower()
            title = hit.payload['title'].lower()
            
            # Keyword scoring
            text_matches = sum(1 for term in query_terms if term in text)
            title_matches = sum(1 for term in query_terms if term in title) * 2  # Title matches worth more
            
            max_possible_matches = len(query_terms) * 3  # text + 2*title
            keyword_score = (text_matches + title_matches) / max_possible_matches if max_possible_matches > 0 else 0
            
            # Combined score
            combined_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)
            
            reranked.append({
                'score': combined_score,
                'semantic_score': semantic_score,
                'keyword_score': keyword_score,
                'payload': hit.payload
            })
        
        # Sort by combined score
        reranked.sort(key=lambda x: x['score'], reverse=True)
        
        # Display results
        for i, result in enumerate(reranked[:top_k], 1):
            score = result['score']
            
            if score > 0.7:
                relevance = "🟢 Highly Relevant"
            elif score > 0.5:
                relevance = "🟡 Relevant"
            else:
                relevance = "🔴 Somewhat Relevant"
            
            payload = result['payload']
            
            log(f"\n{i}. {relevance} (Score: {score:.4f})")
            log(f"   [Semantic: {result['semantic_score']:.3f} | Keyword: {result['keyword_score']:.3f}]")
            log(f"   Source: [{payload['source'].upper()}] {payload['source_type']}")
            log(f"   Title: {payload['title'][:70]}...")
            log(f"   Preview: {payload['text'][:200]}...")
            log(f"   URL: {payload.get('url', 'N/A')}")
        
        log("\n" + "=" * 60)
        return reranked[:top_k]
    
    def get_stats(self):
        """Get collection statistics"""
        log("=" * 60)
        log("Qdrant Collection Statistics")
        log("=" * 60)
        
        try:
            info = self.qdrant.get_collection(self.collection_name)
            log(f"Collection: {self.collection_name}")
            log(f"Total vectors: {info.vectors_count}")
            log(f"Status: {info.status}")
            
            # Breakdown by source
            log("\nBreakdown by source:")
            for source in ['gitlab', 'jira', 'confluence']:
                count = self.qdrant.count(
                    collection_name=self.collection_name,
                    count_filter=Filter(
                        must=[FieldCondition(key="source", match=MatchValue(value=source))]
                    )
                )
                if count.count > 0:
                    log(f"  {source}: {count.count}")
        
        except Exception as e:
            log(f"Error getting stats: {e}", "ERROR")


def main():
    """Main search interface"""
    if not QDRANT_AVAILABLE or not EMBEDDINGS_AVAILABLE:
        log("Missing dependencies", "ERROR")
        log("Install: pip install qdrant-client sentence-transformers", "INFO")
        return
    
    if len(sys.argv) < 2:
        print("\n🔍 Qdrant Semantic Search")
        print("=" * 60)
        print("\nUsage:")
        print("  python qdrant_search.py search 'your query'")
        print("  python qdrant_search.py hybrid 'your query'")
        print("  python qdrant_search.py filter 'query' --source gitlab")
        print("  python qdrant_search.py stats")
        print("\nExamples:")
        print("  python qdrant_search.py search 'authentication bug'")
        print("  python qdrant_search.py hybrid 'deployment process'")
        print("  python qdrant_search.py filter 'API' --source gitlab --project ecommerce")
        return
    
    command = sys.argv[1]
    
    try:
        searcher = QdrantSearch()
        
        if command == "search":
            if len(sys.argv) < 3:
                print("Usage: python qdrant_search.py search 'your query'")
                return
            
            query = " ".join(sys.argv[2:])
            searcher.search(query)
        
        elif command == "hybrid":
            if len(sys.argv) < 3:
                print("Usage: python qdrant_search.py hybrid 'your query'")
                return
            
            query = " ".join(sys.argv[2:])
            searcher.hybrid_search(query)
        
        elif command == "filter":
            if len(sys.argv) < 3:
                print("Usage: python qdrant_search.py filter 'query' [--source X] [--project Y]")
                return
            
            query = sys.argv[2]
            filters = {}
            
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
            
            searcher.search(query, filters=filters)
        
        elif command == "stats":
            searcher.get_stats()
        
        else:
            print(f"Unknown command: {command}")
            print("Run without arguments to see usage")
    
    except Exception as e:
        log(f"Error: {e}", "ERROR")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

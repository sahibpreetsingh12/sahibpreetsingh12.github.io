#!/usr/bin/env python3
"""
Vector Store
Local embeddings and similarity search for resume-based RAG system.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("sentence-transformers not found. Install with: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("numpy not found. Install with: pip install numpy")
    NUMPY_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    print("scikit-learn not found. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class VectorStore:
    """Local vector store for semantic search using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.chunks = []
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    async def initialize(self, chunks: List[Dict[str, str]]) -> None:
        """Initialize the vector store with resume chunks"""
        logger.info("🔍 Initializing vector store...")
        
        self.chunks = chunks
        
        # Check if we can use real embeddings
        if SENTENCE_TRANSFORMERS_AVAILABLE and NUMPY_AVAILABLE and SKLEARN_AVAILABLE:
            await self._initialize_real_embeddings()
        else:
            logger.warning("⚠️ Required libraries not available, using mock similarity search")
            await self._initialize_mock_embeddings()
        
        logger.info(f"✅ Vector store ready with {len(self.chunks)} chunks")
    
    async def _initialize_real_embeddings(self) -> None:
        """Initialize with real sentence transformer embeddings"""
        try:
            # Load the model
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Check if embeddings are cached
            cache_file = self.cache_dir / f"embeddings_{self.model_name.replace('/', '_')}.pkl"
            
            if cache_file.exists():
                logger.info("📂 Loading cached embeddings...")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if len(cached_data['chunks']) == len(self.chunks):
                        self.embeddings = cached_data['embeddings']
                        logger.info("✅ Cached embeddings loaded")
                        return
            
            # Generate new embeddings
            logger.info("🧮 Generating embeddings for resume chunks...")
            chunk_texts = [chunk['content'] for chunk in self.chunks]
            self.embeddings = self.model.encode(chunk_texts, convert_to_tensor=False)
            
            # Cache the embeddings
            logger.info("💾 Caching embeddings...")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'embeddings': self.embeddings,
                    'chunks': self.chunks,
                    'model_name': self.model_name
                }, f)
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize real embeddings: {e}")
            await self._initialize_mock_embeddings()
    
    async def _initialize_mock_embeddings(self) -> None:
        """Fallback mock similarity search based on keyword matching"""
        logger.info("🔄 Using mock keyword-based similarity search")
        self.embeddings = None
        self.model = None
    
    async def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, str], float]]:
        """
        Search for relevant chunks based on query
        
        Args:
            query: User's question
            top_k: Number of top results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.model and self.embeddings is not None:
            return await self._real_search(query, top_k)
        else:
            return await self._mock_search(query, top_k)
    
    async def _real_search(self, query: str, top_k: int) -> List[Tuple[Dict[str, str], float]]:
        """Real semantic search using embeddings"""
        try:
            # Encode the query
            query_embedding = self.model.encode([query], convert_to_tensor=False)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                chunk = self.chunks[idx]
                score = float(similarities[idx])
                results.append((chunk, score))
            
            logger.info(f"🔍 Found {len(results)} relevant chunks for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"❌ Real search failed: {e}")
            return await self._mock_search(query, top_k)
    
    async def _mock_search(self, query: str, top_k: int) -> List[Tuple[Dict[str, str], float]]:
        """Fallback keyword-based search"""
        query_lower = query.lower()
        results = []
        
        # Define keyword mappings for different types of queries
        keyword_mappings = {
            'experience': ['experience', 'work', 'job', 'role', 'position', 'career', 'employment'],
            'skills': ['skill', 'technology', 'technical', 'programming', 'framework', 'tool', 'language'],
            'education': ['education', 'degree', 'university', 'college', 'study', 'academic'],
            'projects': ['project', 'built', 'developed', 'created', 'implementation'],
            'research': ['research', 'paper', 'publication', 'study', 'investigation'],
            'achievements': ['achievement', 'award', 'recognition', 'accomplishment', 'success'],
            'contact': ['contact', 'email', 'phone', 'linkedin', 'github', 'reach']
        }
        
        # Score chunks based on keyword matches and type relevance
        for chunk in self.chunks:
            score = 0.0
            chunk_content = chunk['content'].lower()
            chunk_type = chunk.get('type', 'general')
            
            # Base score for content keyword matches
            for word in query_lower.split():
                if word in chunk_content:
                    score += 0.3
            
            # Bonus score for type-specific matches
            for chunk_type_key, keywords in keyword_mappings.items():
                if chunk_type == chunk_type_key:
                    for keyword in keywords:
                        if keyword in query_lower:
                            score += 0.5
            
            # Special bonus for exact matches
            if any(keyword in query_lower for keyword in ['sahibpreet', 'singh']):
                score += 0.2
            
            if score > 0:
                results.append((chunk, score))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_chunk_by_type(self, chunk_type: str) -> Optional[Dict[str, str]]:
        """Get a specific chunk by its type"""
        for chunk in self.chunks:
            if chunk.get('type') == chunk_type:
                return chunk
        return None
    
    def get_all_chunk_types(self) -> List[str]:
        """Get all available chunk types"""
        return list(set(chunk.get('type', 'general') for chunk in self.chunks))
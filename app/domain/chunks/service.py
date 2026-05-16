"""
Chunk service for business logic
"""
from typing import List, Optional, Dict, Any
from app.models.papers import DBPaperChunk
from app.extensions.logger import create_logger

from .repository import ChunkRepository
from .schemas import ChunkResponse, Chunk, ChunkRetrieved

logger = create_logger(__name__)


class ChunkService:
    """Service for chunk operations"""
    
    def __init__(self, repository: ChunkRepository, search_service=None):
        self.repository = repository
        self._search_service = search_service

    @property
    def search_service(self):
        """Lazy load local chunk search service."""
        if self._search_service is None:
            from app.search import ChunkSearchService

            self._search_service = ChunkSearchService(self.repository)
        return self._search_service
    
    async def create_chunk(
        self,
        chunk_id: str,
        paper_id: str,
        text: str,
        token_count: int,
        chunk_index: int,
        embedding: List[float],
        section_title: Optional[str] = None,
        page_number: Optional[int] = None,
        label: Optional[str] = None,
        level: Optional[int] = None,
        char_start: Optional[int] = None,
        char_end: Optional[int] = None,
        docling_metadata: Optional[Dict[str, Any]] = None,
    ) -> DBPaperChunk:
        """Create a paper chunk with full metadata"""
        return await self.repository.create_chunk(
            chunk_id=chunk_id,
            paper_id=paper_id,
            text=text,
            token_count=token_count,
            chunk_index=chunk_index,
            embedding=embedding,
            section_title=section_title,
            page_number=page_number,
            label=label,
            level=level,
            char_start=char_start,
            char_end=char_end,
            docling_metadata=docling_metadata,
        )
    
    async def get_paper_chunks(self, paper_id: str) -> List[ChunkResponse]:
        """Get all chunks for a paper"""
        chunks = await self.repository.get_chunks_by_paper_id(paper_id)
        return [ChunkResponse.model_validate(chunk) for chunk in chunks]
    
    async def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get a single chunk by ID"""
        chunk = await self.repository.get_chunk_by_id(chunk_id)
        if not chunk:
            return None
        return Chunk.model_validate(chunk)
    
    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        limit: int = 40,
        paper_ids: Optional[List[str]] = None
    ) -> List[ChunkRetrieved]:
        """
        Search for similar chunks using vector similarity.
        Returns tuples of (chunk, similarity_score).
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
            paper_ids: Optional list of paper IDs to restrict search
            
        Returns:
            List of ChunkRetrieved with relevance scores
        """
        rows = await self.repository.search_similar_chunks(
            query_embedding, limit, paper_ids
        )
        results = []
        for chunk, score in rows:
            chunk_dict = Chunk.model_validate(chunk).model_dump()
            chunk_dict['relevance_score'] = score
            chunk_retrieved = ChunkRetrieved.model_validate(chunk_dict)
            results.append(chunk_retrieved)
        return results
    
    async def delete_chunks_for_paper(self, paper_id: str) -> int:
        """
        Delete all chunks for a paper
        
        Args:
            paper_id: Paper ID to delete chunks for
            
        Returns:
            Number of chunks deleted
        """
        return await self.repository.delete_chunks_by_paper_id(paper_id)
    
    async def hybrid_search_chunks(
        self,
        query: str,
        limit: int = 40,
        paper_ids: Optional[List[str]] = None,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
    ) -> List[ChunkRetrieved]:
        """
        Hybrid BM25 + semantic search on chunks.
        
        Handles:
        - Query embedding generation
        - Weight normalization
        - Calls repository for data access
        - Converts results to ChunkRetrieved
        
        Args:
            query: Search query text
            limit: Maximum number of results
            paper_ids: Optional list of paper IDs to restrict search
            bm25_weight: Weight for BM25 score (0-1)
            semantic_weight: Weight for semantic score (0-1)
            
        Returns:
            List of ChunkRetrieved with relevance scores
        """
        return await self.search_service.hybrid_search(
            query=query,
            limit=limit,
            paper_ids=paper_ids,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
        )

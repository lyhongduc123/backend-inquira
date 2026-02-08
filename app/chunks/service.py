"""
Chunk service for business logic
"""
from typing import List, Optional
from app.chunks.repository import ChunkRepository
from app.chunks.schemas import ChunkResponse, Chunk, ChunkRetrieved
from app.models.papers import DBPaperChunk
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class ChunkService:
    """Service for chunk operations"""
    
    def __init__(self, repository: ChunkRepository):
        self.repository = repository
    
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
    ) -> DBPaperChunk:
        """Create a paper chunk"""
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
            List of tuples (chunk, similarity_score) ordered by similarity
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

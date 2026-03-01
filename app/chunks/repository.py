"""
Chunk repository for database operations
"""
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.papers import DBPaperChunk
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class ChunkRepository:
    """Repository for chunk database operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
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
        defer_commit: bool = False,
    ) -> DBPaperChunk:
        """Create a paper chunk
        
        Args:
            defer_commit: If True, don't commit immediately (for batch operations)
        """
        db_chunk = DBPaperChunk(
            chunk_id=chunk_id,
            paper_id=paper_id,
            text=text,
            token_count=token_count,
            chunk_index=chunk_index,
            section_title=section_title,
            page_number=page_number,
            label=label,
            level=level,
            embedding=embedding,
        )
        
        self.db.add(db_chunk)
        
        if not defer_commit:
            await self.db.commit()
            await self.db.refresh(db_chunk)
        
        return db_chunk
    
    async def get_chunks_by_paper_id(self, paper_id: str) -> List[DBPaperChunk]:
        """Get all chunks for a paper"""
        result = await self.db.execute(
            select(DBPaperChunk)
            .where(DBPaperChunk.paper_id == paper_id)
            .order_by(DBPaperChunk.chunk_index)
        )
        return list(result.scalars().all())
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[DBPaperChunk]:
        """Get a single chunk by chunk_id"""
        result = await self.db.execute(
            select(DBPaperChunk).where(DBPaperChunk.chunk_id == chunk_id)
        )
        return result.scalar_one_or_none()
    
    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        limit: int = 40,
        paper_ids: Optional[List[str]] = None
    ) -> List[tuple[DBPaperChunk, float]]:
        """
        Search for similar chunks using vector similarity
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
            paper_ids: Optional list of paper IDs to restrict search
            
        Returns:
            List of tuples (chunk, similarity_score) ordered by similarity descending
            Similarity scores range from 0 to 1, where 1 is most similar
        """
        # Calculate similarity as (1 - cosine_distance)
        similarity = 1 - DBPaperChunk.embedding.cosine_distance(query_embedding)
        
        query = select(DBPaperChunk, similarity.label('similarity_score')).order_by(
            DBPaperChunk.embedding.cosine_distance(query_embedding)
        ).limit(limit)
        
        # Filter by paper IDs if provided
        if paper_ids:
            query = query.where(DBPaperChunk.paper_id.in_(paper_ids))
        
        result = await self.db.execute(query)
        chunks_with_scores = result.all()
        
        # Return as list of tuples (chunk, score)
        return [(chunk, float(score)) for chunk, score in chunks_with_scores]
    
    async def delete_chunks_by_paper_id(self, paper_id: str) -> int:
        """
        Delete all chunks for a paper
        
        Args:
            paper_id: Paper ID to delete chunks for
            
        Returns:
            Number of chunks deleted
        """
        from sqlalchemy import delete as sql_delete
        
        result = await self.db.execute(
            sql_delete(DBPaperChunk).where(DBPaperChunk.paper_id == paper_id)
        )
        await self.db.commit()
        
        return result.rowcount # type: ignore

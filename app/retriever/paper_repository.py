from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from datetime import datetime
from app.models.papers import DBPaper, DBPaperChunk, DBResearchQuery
from app.retriever.paper_schemas import Paper, PaperChunk, Author
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class PaperRepository:
    """Repository for paper database operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_paper(self, paper: Paper) -> DBPaper:
        """
        Create a new paper in database
        
        Args:
            paper: Paper schema object
            
        Returns:
            Created DBPaper object
        """
        # Convert authors to JSONB format
        authors_json = [{"name": author.name, "author_id": author.author_id} for author in paper.authors]
        
        db_paper = DBPaper(
            paper_id=paper.paper_id,
            title=paper.title,
            authors=authors_json,
            abstract=paper.abstract,
            publication_date=paper.publication_date,
            venue=paper.venue,
            is_open_access=paper.is_open_access,
            open_access_pdf=paper.open_access_pdf,
            url=paper.url,
            pdf_url=paper.pdf_url,
            source=paper.source,
            external_id=paper.external_id,
            relevance_score=paper.relevance_score,
            citation_count=paper.citation_count,
            influential_citation_count=paper.influential_citation_count,
            reference_count=paper.reference_count,
            is_processed=paper.is_processed,
            processing_status=paper.processing_status
        )
        
        self.db.add(db_paper)
        await self.db.commit()
        await self.db.refresh(db_paper)
        
        logger.info(f"Created paper {paper.paper_id} in database")
        return db_paper
    
    async def get_paper_by_id(self, paper_id: str) -> Optional[DBPaper]:
        """Get paper by internal paper_id"""
        result = await self.db.execute(
            select(DBPaper).where(DBPaper.paper_id == paper_id)
        )
        return result.scalar_one_or_none()
    
    async def get_paper_by_external_id(self, external_id: str, source: str) -> Optional[DBPaper]:
        """Get paper by external ID and source"""
        result = await self.db.execute(
            select(DBPaper).where(
                and_(
                    DBPaper.external_id == external_id,
                    DBPaper.source == source
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def update_paper_processing_status(
        self,
        paper_id: str,
        status: str,
        error: Optional[str] = None
    ):
        """Update paper processing status"""
        update_data = {
            "processing_status": status,
            "updated_at": datetime.utcnow()
        }
        
        if status == "completed":
            update_data["is_processed"] = True
        
        if error:
            update_data["processing_error"] = error
        
        await self.db.execute(
            update(DBPaper)
            .where(DBPaper.paper_id == paper_id)
            .values(**update_data)
        )
        await self.db.commit()
    
    async def update_paper_summary(
        self,
        paper_id: str,
        summary: str,
        summary_embedding: List[float]
    ):
        """Update paper summary and embedding"""
        await self.db.execute(
            update(DBPaper)
            .where(DBPaper.paper_id == paper_id)
            .values(
                summary=summary,
                summary_embedding=summary_embedding,
                updated_at=datetime.now()
            )
        )
        await self.db.commit()
    
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
        embedding_dimension: Optional[int] = None
    ) -> DBPaperChunk:
        """Create a paper chunk"""
        db_chunk = DBPaperChunk(
            chunk_id=chunk_id,
            paper_id=paper_id,
            text=text,
            token_count=token_count,
            chunk_index=chunk_index,
            section_title=section_title,
            page_number=page_number,
            embedding=embedding,
            embedding_dimension=embedding_dimension
        )
        
        self.db.add(db_chunk)
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
    
    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        limit: int = 10,
        paper_ids: Optional[List[str]] = None
    ) -> List[DBPaperChunk]:
        """
        Search for similar chunks using vector similarity
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
            paper_ids: Optional list of paper IDs to restrict search
            
        Returns:
            List of similar chunks ordered by similarity
        """
        # Build query with cosine similarity
        query = select(DBPaperChunk).order_by(
            DBPaperChunk.embedding.cosine_distance(query_embedding)
        ).limit(limit)
        
        # Filter by paper IDs if provided
        if paper_ids:
            query = query.where(DBPaperChunk.paper_id.in_(paper_ids))
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def search_similar_papers(
        self,
        query_embedding: List[float],
        limit: int = 10
    ) -> List[DBPaper]:
        """
        Search for similar papers using summary embeddings
        
        Args:
            query_embedding: Query embedding vector
            limit: Number of results to return
            
        Returns:
            List of similar papers ordered by similarity
        """
        result = await self.db.execute(
            select(DBPaper)
            .where(DBPaper.summary_embedding.isnot(None))
            .order_by(DBPaper.summary_embedding.cosine_distance(query_embedding))
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def update_last_accessed(self, paper_id: str):
        """Update last accessed timestamp"""
        await self.db.execute(
            update(DBPaper)
            .where(DBPaper.paper_id == paper_id)
            .values(last_accessed_at=datetime.utcnow())
        )
        await self.db.commit()
    
    async def create_research_query(
        self,
        query_text: str,
        query_embedding: List[float],
        retrieved_paper_ids: List[str],
        answer: Optional[str] = None,
        confidence: Optional[float] = None,
        user_id: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> DBResearchQuery:
        """Create a research query record"""
        db_query = DBResearchQuery(
            query_text=query_text,
            query_embedding=query_embedding,
            retrieved_paper_ids=retrieved_paper_ids,
            answer=answer,
            confidence=confidence,
            user_id=user_id,
            session_id=session_id
        )
        
        self.db.add(db_query)
        await self.db.commit()
        await self.db.refresh(db_query)
        
        return db_query

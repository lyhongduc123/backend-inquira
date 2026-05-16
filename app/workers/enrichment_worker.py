"""
Worker functions for enrichment tasks.
These run in background without blocking API responses.
"""

from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.extensions.logger import create_logger
from app.core.db.database import async_session

logger = create_logger(__name__)


class EnrichmentWorker:
    """Worker class for enrichment operations"""
    
    @staticmethod
    async def enrich_author_background(author_id: str, limit: int = 500) -> Dict[str, Any]:
        """
        Background task for author enrichment.
        Creates its own database session to avoid conflicts.
        
        Args:
            author_id: Author ID to enrich
            limit: Maximum papers to fetch
        
        Returns:
            Dict with enrichment results
        """
        from app.domain.authors.service import AuthorService
        
        async with async_session() as db:
            try:
                logger.info(f"[WORKER] Starting author enrichment for {author_id}")
                
                author_service = AuthorService(db)
                result = await author_service.ingest_author_pipeline(
                    author_id=author_id,
                    limit=limit,
                    compute_relationships=True,
                )
                
                papers_count = len(result.papers) if result.papers else 0
                logger.info(
                    f"[WORKER] Completed author enrichment for {author_id}: "
                    f"{papers_count} papers retrieved"
                )
                
                return {
                    "author_id": author_id,
                    "papers_count": papers_count,
                    "success": True
                }
                
            except Exception as e:
                logger.error(
                    f"[WORKER] Author enrichment failed for {author_id}: {e}",
                    exc_info=True
                )
                return {
                    "author_id": author_id,
                    "success": False,
                    "error": str(e)
                }
    
    @staticmethod
    async def compute_author_relationships_background(author_id: str) -> Dict[str, Any]:
        """
        Background task for computing author relationships (collaborations, citing, referenced).
        
        Args:
            author_id: Author ID to compute relationships for
        
        Returns:
            Dict with computation results
        """
        from app.domain.authors.service import AuthorService
        
        async with async_session() as db:
            try:
                logger.info(f"[WORKER] Computing relationships for author {author_id}")
                
                author_service = AuthorService(db)
                result = await author_service.compute_author_relationships(author_id)
                
                logger.info(
                    f"[WORKER] Computed relationships for {author_id}: "
                    f"{result['collaborations']} collaborations"
                )
                
                return {
                    "author_id": author_id,
                    "success": True,
                    **result
                }
                
            except Exception as e:
                logger.error(
                    f"[WORKER] Relationship computation failed for {author_id}: {e}",
                    exc_info=True
                )
                return {
                    "author_id": author_id,
                    "success": False,
                    "error": str(e)
                }
    


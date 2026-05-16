"""
Single phase preprocessing endpoint for running specific preprocessing phases independently.

This module provides a simplified interface for running either embedding-only or 
content-processing-only phases based on the requested preprocessing_phase parameter.
"""

from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.extensions.logger import create_logger
from app.processor.preprocessing_service import PreprocessingService
from app.processor.preprocessing_repository import PreprocessingRepository
from app.domain.papers.types import PaperEnrichedDTO

logger = create_logger(__name__)


class PreprocessingSinglePhaseService:
    """
    Service for running specific preprocessing phases based on phase parameter.
    
    Supports two main phases:
    - embed: Run only embedding generation (Phase 3)
    - process_content: Run only PDF extraction/chunking/embedding (Phase 4)
    """

    def __init__(
        self,
        db_session: AsyncSession,
        preprocessing_service: Optional[PreprocessingService] = None,
        preprocessing_repo: Optional[PreprocessingRepository] = None,
    ):
        self.db_session = db_session
        self.preprocessing_service = preprocessing_service or PreprocessingService(db_session)
        self.preprocessing_repo = preprocessing_repo or PreprocessingRepository(db_session)

    async def run_preprocessing(
        self,
        run_embed: bool = False,
        run_process_content: bool = False,
        paper_ids: Optional[List[str]] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Run the preprocessing pipeline with conditional phase execution.
        
        Uses if-else logic to execute only the non-excluded preprocessing stage
        based on the provided boolean flags.

        Args:
            run_embed: If True, run embedding generation for papers missing embeddings
            run_process_content: If True, run PDF content processing (extract, chunk, embed)
            paper_ids: Optional list of specific paper IDs to process
            limit: Maximum number of papers to process per iteration

        Returns:
            Statistics dictionary with processing results
        """
        if run_embed and run_process_content:
            return await self._run_full_preprocessing(paper_ids=paper_ids, limit=limit)
        elif run_embed:
            return await self._run_embed_only(paper_ids=paper_ids, limit=limit)
        elif run_process_content:
            return await self._run_content_processing_only(limit=limit)
        else:
            return {
                "error": "No valid phase specified",
                "message": "Set either run_embed=True or run_process_content=True",
                "processed": 0,
                "failed": 0,
            }

    async def _run_embed_only(
        self,
        paper_ids: Optional[List[str]] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Run embedding generation for papers missing embeddings."""
        if paper_ids:
            await self.preprocessing_service._generate_missing_embeddings_for_papers(paper_ids)
            return {
                "phase": "embedding",
                "mode": "selected_papers",
                "paper_count": len(paper_ids),
                "message": "Embeddings generated for specified papers",
            }
        else:
            result = await self.preprocessing_service._generate_missing_embeddings(state=None)
            return {
                "phase": "embedding",
                "mode": "all_missing",
                "considered": result.get("considered", 0),
                "updated": result.get("updated", 0),
                "remaining_missing": result.get("remaining_missing", 0),
            }

    async def _run_content_processing_only(self, limit: int = 50) -> Dict[str, Any]:
        """Run content processing (PDF extract, chunk, embed) for pending papers."""
        result = await self.preprocessing_service.run_content_processing(limit=limit)
        return {
            "phase": "content_processing",
            "total": result.get("total", 0),
            "processed": result.get("processed", 0),
            "failed": result.get("failed", 0),
        }

    async def _run_full_preprocessing(
        self,
        paper_ids: Optional[List[str]] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Run both embedding and content processing phases."""
        embed_result = await self._run_embed_only(paper_ids=paper_ids, limit=limit)
        content_result = await self._run_content_processing_only(limit=limit)
        
        return {
            "phase": "full",
            "embedding": embed_result,
            "content_processing": content_result,
        }
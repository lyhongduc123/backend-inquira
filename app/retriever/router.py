"""
Debug router for retriever service endpoints - NO AUTH REQUIRED
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.extensions.logger import create_logger
from app.core.db.database import get_db_session as get_db
from app.domain.papers.types import PaperEnrichedDTO
from app.retriever.service import RetrievalService

logger = create_logger(__name__)

router = APIRouter(prefix="/api/retriever/debug", tags=["retriever_debug"])


def get_retrieval_service(db: AsyncSession = Depends(get_db)) -> RetrievalService:
    """Get retrieval service instance"""
    return RetrievalService(db=db)


@router.get("/get-multiple-papers", response_model=List[PaperEnrichedDTO])
async def debug_get_multiple_papers(
    paper_ids: str,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> List[PaperEnrichedDTO]:
    """
    Debug endpoint to fetch multiple papers by IDs.

    Args:
        paper_ids: Comma-separated list of paper IDs (e.g., "12345678,87654321")

    Returns:
        List of enriched paper objects
    """
    try:
        ids = [id.strip() for id in paper_ids.split(",")]
        logger.info(f"[DEBUG] Fetching {len(ids)} papers: {ids}")

        papers = await retrieval_service.get_multiple_papers(ids)
        logger.info(f"[DEBUG] Successfully fetched {len(papers)} papers")

        return papers

    except Exception as e:
        logger.error(f"[DEBUG] Error fetching multiple papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hybrid-search", response_model=Dict[str, Any])
async def debug_hybrid_search(
    query: str,
    semantic_limit: int = 50,
    final_limit: int = 100,
    enable_enrichment: bool = True,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> Dict[str, Any]:
    """
    Debug endpoint for hybrid search combining Semantic Scholar + OpenAlex.

    Args:
        query: Search query string
        semantic_limit: Max results from Semantic Scholar (default 50)
        final_limit: Max final results to return (default 100)
        enable_enrichment: Whether to enrich with OpenAlex metadata (default True)

    Returns:
        Dictionary with papers and metadata:
        {
            "papers": [...],
            "metadata": {
                "semantic_scholar_count": int,
                "openalex_enriched_count": int,
                "final_returned": int
            }
        }
    """
    try:
        logger.info(
            f"[DEBUG] Hybrid search query: {query[:50]}... (semantic_limit={semantic_limit}, final_limit={final_limit})"
        )

        papers, metadata = await retrieval_service.hybrid_search(
            query=query,
            s2_limit=semantic_limit,
            final_limit=final_limit,
            enable_enrichment=enable_enrichment,
        )

        logger.info(
            f"[DEBUG] Hybrid search returned {len(papers)} papers. Metadata: {metadata}"
        )

        return {"papers": papers, "metadata": metadata}

    except Exception as e:
        logger.error(f"[DEBUG] Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def debug_health() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "ok", "service": "retriever_debug"}





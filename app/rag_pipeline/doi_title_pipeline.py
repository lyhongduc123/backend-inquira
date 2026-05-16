# app/rag_pipeline/doi_title_pipeline.py
"""
DOI & Title-Only Lookup Pipeline

Specialized pipeline for retrieving papers by exact DOI or title match.
Skips hybrid search entirely; focuses on precision retrieval of specific papers.
Includes paper ranking but only on the narrow set of matched papers.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.container import ServiceContainer
from app.domain.papers.repository import LoadOptions
from app.domain.chunks.types import ChunkRetrieved
from app.models.papers import DBPaper
from app.search.types import RankedPaper
from app.llm.schemas import QueryIntent

from app.rag_pipeline.schemas import (
    RAGPipelineContext,
    RAGResult,
)
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class DoiTitlePipeline:
    """
    Specialized pipeline for DOI/Title lookup without hybrid search.
    
    Only retrieves papers by:
    1. Exact DOI match via external ID lookup
    2. Exact title match via BM25 top-1
    
    If papers found, retrieves chunks and applies reranking (no paper ranking).
    Returns early if no papers found (no fallback).
    """

    def __init__(
        self,
        db_session: AsyncSession,
        container: Optional[ServiceContainer] = None,
    ):
        """Initialize DOI/Title Pipeline."""
        self.db_session = db_session
        self.container = container or ServiceContainer(db_session)
        
        # Core services
        self.paper_service = self.container.paper_service
        self.chunk_service = self.container.chunk_service
        self.ranking_service = self.container.ranking_service


    async def run_doi_title_lookup(
        self,
        original_query: str,
        specific_papers: Optional[List[str]] = None,
        dois: Optional[List[str]] = None,
        rerank_query: Optional[str] = None,
        top_papers: int = 50,
        top_chunks: int = 40,
        enable_reranking: bool = True,
        relevance_threshold: float = 0.3,
    ) -> Optional[RAGResult]:
        """
        Lookup papers by DOI/specific title.
        
        Args:
            original_query: User's original query (for context)
            specific_papers: List of paper titles to match
            dois: List of DOIs to lookup
            rerank_query: Query for chunk reranking (defaults to original_query)
            top_papers: Max papers to return
            top_chunks: Max chunks to return
            enable_reranking: Whether to rerank chunks
            relevance_threshold: Minimum chunk relevance score
            
        Returns:
            RAGResult with papers and chunks, or empty RAGResult if no matches
        """
        
        ctx = RAGPipelineContext(query=original_query)
        db_papers_with_scores: List[tuple[DBPaper, float]] = []
        
        logger.info(f"DOI/Title Lookup: Searching for {len(specific_papers or [])} titles and {len(dois or [])} DOIs")
        
        if dois:
            for doi in dois:
                try:
                    paper = await self.paper_service.get_paper_by_external_ids(
                        external_ids={"doi": doi}
                    )
                    if paper:
                        db_papers_with_scores.append((paper, 1.0))
                        logger.info(f"Found paper by DOI: {doi}")
                    else:
                        logger.debug(f"No paper found for DOI: {doi}")
                except Exception as e:
                    logger.warning(f"Failed to lookup DOI {doi}: {e}")
        
        # Lookup by specific paper title (exact match via BM25 top-1)
        if specific_papers:
            found_paper_ids = {p.paper_id for p, _ in db_papers_with_scores}
            
            for title in specific_papers:
                try:
                    results = await self.paper_service.bm25_search(
                        query=title,
                        limit=1,
                        filter_options=None
                    )
                    if results:
                        paper, score = results[0]
                        if paper.paper_id not in found_paper_ids:
                            db_papers_with_scores.append((paper, score))
                            found_paper_ids.add(paper.paper_id)
                            logger.info(f"Found paper by title: '{title[:60]}...' (score: {score:.3f})")
                        else:
                            logger.debug(f"Paper already found for title: '{title[:60]}...'")
                    else:
                        logger.debug(f"No paper found for title: '{title[:60]}...'")
                except Exception as e:
                    logger.warning(f"Failed to lookup title '{title[:60]}...': {e}")
        
        if not db_papers_with_scores:
            logger.info("DOI/Title lookup found no papers. Returning empty result.")
            return RAGResult(papers=[], chunks=[])

        refreshed_papers_with_scores: List[tuple[DBPaper, float]] = []
        for paper, score in db_papers_with_scores:
            try:
                await self.db_session.refresh(paper)
                refreshed_papers_with_scores.append((paper, score))
            except Exception as exc:
                logger.debug(
                    "Failed to refresh DOI/title lookup paper before result build: %s",
                    exc,
                )
        db_papers_with_scores = refreshed_papers_with_scores
        if not db_papers_with_scores:
            logger.info("DOI/Title lookup papers expired before result build. Returning empty result.")
            return RAGResult(papers=[], chunks=[])
        
        db_papers_with_scores = db_papers_with_scores[:top_papers]
        logger.info(f"DOI/Title lookup found {len(db_papers_with_scores)} papers")
        
        ctx.papers_with_hybrid_scores = db_papers_with_scores
        paper_ids = [p.paper_id for p, _ in db_papers_with_scores]
        
        rerank_basis_query = (rerank_query or original_query).strip() or original_query
        
        if paper_ids:
            chunks = await self._chunk_search(
                query=rerank_basis_query,
                paper_ids=paper_ids,
                top_chunks=top_chunks,
            )
            
            papers_with_chunks = {chunk.paper_id for chunk in chunks}
            papers_without_chunks = [
                (paper, score) for paper, score in db_papers_with_scores
                if paper.paper_id not in papers_with_chunks and paper.abstract
            ]
            
            # Add abstract as virtual chunk for papers without extracted chunks
            if papers_without_chunks:
                for paper, score in papers_without_chunks:
                    virtual_chunk = ChunkRetrieved(
                        chunk_id=f"{paper.paper_id}_abstract",
                        paper_id=paper.paper_id,
                        text=paper.abstract,
                        token_count=len(paper.abstract.split()),
                        chunk_index=0,
                        section_title="Abstract",
                        page_number=None,
                        label="abstract",
                        level=0,
                        id=paper.id,
                        char_start=None,
                        char_end=None,
                        docling_metadata=None,
                        embedding=None,
                        created_at=datetime.now(),
                        relevance_score=score * 0.8,
                    )
                    chunks.append(virtual_chunk)
            
            ctx.chunks = chunks
        
        if enable_reranking and ctx.chunks:
            try:
                ctx.chunks = self.ranking_service.rerank_chunks(rerank_basis_query, ctx.chunks)
                ctx.chunks = [c for c in ctx.chunks if getattr(c, "relevance_score", 0.0) >= relevance_threshold]
                logger.info(f"Reranked chunks: {len(ctx.chunks)} after threshold filtering")
            except Exception as e:
                logger.error(f"Error reranking chunks: {e}")
        
        ctx.result_papers = [
            RankedPaper(
                id=p.id,
                paper_id=p.paper_id,
                paper=p,
                relevance_score=score,
                ranking_scores={"doi_title_score": score}
            )
            for p, score in db_papers_with_scores[:top_papers]
        ]
        
        # Filter chunks to matched papers only
        ctx.chunks = [chunk for chunk in ctx.chunks if chunk.paper_id in [rp.paper_id for rp in ctx.result_papers]]

        return RAGResult(
            papers=ctx.result_papers,
            chunks=ctx.chunks[:top_chunks],
        )

    async def _chunk_search(
        self,
        query: str,
        paper_ids: List[str],
        top_chunks: int,
    ) -> List[ChunkRetrieved]:
        """Retrieve chunks for given papers."""
        try:
            return await self.chunk_service.hybrid_search_chunks(
                query=query,
                paper_ids=paper_ids,
                limit=top_chunks,
            )
        except Exception as e:
            logger.error(f"Chunk search failed: {e}")
            return []

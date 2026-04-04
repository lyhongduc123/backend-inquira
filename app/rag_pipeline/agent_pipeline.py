# app/rag_pipeline/agent_pipeline.py
"""
Agent-Driven Database Search Pipeline

This pipeline is a simplified "dumb executor" for the Agentic State Machine.
Unlike DatabasePipeline, it does NOT perform its own LLM query decomposition.
Instead, it expects explicit `search_queries`, `filters`, and `intent` to be 
passed down directly from the ChatAgentService's orchestration layer.

Features:
- Pure executor for BM25 + Semantic Search
- Accepts explicit explicit search arrays from upstream Agent
- Fuses multi-query results via RRF
"""

import asyncio
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Dict, Any, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.container import ServiceContainer
from app.domain.papers import LoadOptions
from app.domain.chunks.schemas import ChunkRetrieved
from app.core.singletons import get_ranking_service
from app.models.papers import DBPaper
from app.processor.schemas import RankedPaper
from app.llm.schemas import QueryIntent

from app.rag_pipeline.schemas import (
    RAGPipelineContext,
    RAGPipelineEvent,
    RAGResult,
    RAGEventType,
)
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class AgentPipeline:
    """
    Agent-Driven Explicit Search Pipeline.
    
    Acts purely as an execution engine. Takes precise search intents, filters,
    and sub-queries derived from the upstream ChatAgentService.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        container: Optional[ServiceContainer] = None,
    ):
        """Initialize Agent Pipeline."""
        self.db_session = db_session
        self.container = container or ServiceContainer(db_session)
        
        # Core services from container
        self.repository = self.container.paper_repository
        self.chunk_service = self.container.chunk_service
        self.ranking_service = self.container.ranking_service


    async def run_explicit_workflow(
        self,
        original_query: str,
        search_queries: List[str],
        intent: QueryIntent,
        filters: Optional[Dict[str, Any]] = None,
        top_papers: int = 50,
        top_chunks: int = 40,
        enable_reranking: bool = True,
        enable_paper_ranking: bool = True,
        relevance_threshold: float = 0.3,
    ) -> AsyncGenerator[RAGPipelineEvent, None]:
        """
        Database search without internal decomposition step.
        """
        
        ctx = RAGPipelineContext(original_query)
        ctx.search_queries = search_queries
        
        yield RAGPipelineEvent(
            type=RAGEventType.SEARCHING,
            data={
                "queries": search_queries,
                "original": original_query,
                "intent": intent.value,
                "filters": filters or {},
            },
        )

        # Step 1: Database split retrieval with filters
        yield RAGPipelineEvent(
            type=RAGEventType.PROCESSING,
            data={"message": "Executing Agent's search sub-queries via Hybrid RAG"},
        )
        
        paper_rankings: List[List[tuple[DBPaper, float]]] = []
        
        for search_query in search_queries:
            bm25_results, semantic_results = await self._database_split_search(
                query=search_query,
                limit=top_papers * 2,
                filters=filters,
                intent=intent,
            )
            if bm25_results:
                paper_rankings.append(bm25_results)
            if semantic_results:
                paper_rankings.append(semantic_results)

            logger.info(
                f"Agent Query '{search_query[:50]}...' retrieved "
                f"bm25={len(bm25_results)}, semantic={len(semantic_results)}"
            )

        db_papers_with_scores = self._fuse_rankings_with_rrf(
            paper_rankings=paper_rankings,
            k=60,
            limit=top_papers * 2,
        )
        
        logger.info(f"After RRF deduplication: {len(db_papers_with_scores)} unique papers")
        
        if not db_papers_with_scores:
            logger.warning("No papers found in database explicitly")
            yield RAGPipelineEvent(
                type=RAGEventType.RESULT,
                data=RAGResult(papers=[], chunks=[])
            )
            return
        
        ctx.papers_with_hybrid_scores = db_papers_with_scores
        paper_ids = [p.paper_id for p, _ in db_papers_with_scores[:top_papers]]
        
        # Step 2: Fetch Chunks
        if paper_ids:
            yield RAGPipelineEvent(
                type=RAGEventType.PROCESSING,
                data={"message": "Searching chunks in filtered papers"},
            )
            
            # Using the primary search query for chunk search ranking
            primary_query = search_queries[0] if search_queries else original_query
            chunks = await self._chunk_search(
                query=primary_query,
                paper_ids=paper_ids,
                top_chunks=top_chunks,
                intent=intent,
            )
            
            # Add virtual abstract chunks if no real chunks found for a paper
            papers_with_chunks = {chunk.paper_id for chunk in chunks}
            papers_without_chunks = [
                (paper, score) for paper, score in db_papers_with_scores[:top_papers]
                if paper.paper_id not in papers_with_chunks and paper.abstract
            ]
            
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
                        docling_metadata=None, # Importantly, metadata is None here
                        embedding=None,
                        created_at=datetime.now(),
                        relevance_score=score * 0.8,
                    )
                    chunks.append(virtual_chunk)
            
            ctx.chunks = chunks
        
        # Step 3: Reranking
        if enable_reranking and ctx.chunks:
            primary_query = search_queries[0] if search_queries else original_query
            try:
                ctx.chunks = self.ranking_service.rerank_chunks(primary_query, ctx.chunks)
            except Exception as e:
                logger.error(f"Error reranking chunks: {e}")
        
        # Step 4: Paper ranking
        if enable_paper_ranking and ctx.papers_with_hybrid_scores:
            yield RAGPipelineEvent(
                type=RAGEventType.RANKING,
                data={
                    "total_papers": len(ctx.papers_with_hybrid_scores),
                    "total_chunks": len(ctx.chunks),
                },
            )
            
            try:
                paper_ids_to_rank = [p.paper_id for p, _ in ctx.papers_with_hybrid_scores[:top_papers]]
                enriched_papers, _ = await self.repository.get_papers(
                    paper_ids=paper_ids_to_rank,
                    load_options=LoadOptions(authors=True, journal=True, institutions=True),
                )
                
                paper_hybrid_scores = {p.paper_id: score for p, score in ctx.papers_with_hybrid_scores}
                
                ranked_papers = await self._rank_papers(
                    query=original_query,
                    papers=enriched_papers,
                    chunks=ctx.chunks,
                    paper_hybrid_scores=paper_hybrid_scores,
                    intent=intent,
                )
                
                ctx.result_papers = ranked_papers[:top_papers]
            except Exception as e:
                logger.error(f"Error during paper ranking: {e}")
                ctx.result_papers = [
                    RankedPaper(
                        id=p.id,
                        paper_id=p.paper_id,
                        paper=p,
                        relevance_score=score,
                        ranking_scores={"hybrid_score": score}
                    )
                    for p, score in ctx.papers_with_hybrid_scores[:top_papers]
                ]
        
        ctx.chunks = [chunk for chunk in ctx.chunks if chunk.paper_id in [rp.paper_id for rp in ctx.result_papers]]
        
        yield RAGPipelineEvent(
            type=RAGEventType.RESULT,
            data=RAGResult(
                papers=ctx.result_papers,
                chunks=ctx.chunks[:top_chunks],
            ),
        )

    async def _database_split_search(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
        intent: Optional[QueryIntent] = None,
    ) -> Tuple[List[tuple[DBPaper, float]], List[tuple[DBPaper, float]]]:
        try:
            from app.processor.services.embeddings import get_embedding_service

            author_name = filters.get("author") if filters else None
            year_min = filters.get("year_min") if filters else None
            year_max = filters.get("year_max") if filters else None
            venue = filters.get("venue") if filters else None
            min_citations = filters.get("min_citations") if filters else None
            max_citations = filters.get("max_citations") if filters else None

            bm25_results = await self.repository.bm25_search_papers_with_filters(
                query=query,
                limit=limit,
                author_name=author_name,
                year_min=year_min,
                year_max=year_max,
                venue=venue,
                min_citation_count=min_citations,
                max_citation_count=max_citations,
            )

            semantic_results: List[tuple[DBPaper, float]] = []
            embedding_service = get_embedding_service()
            query_embedding = await embedding_service.create_embedding(
                query, task="search_query"
            )
            
            if query_embedding:
                semantic_results = await self.repository.semantic_search_papers_with_filters(
                    query_embedding=query_embedding,
                    limit=limit,
                    author_name=author_name,
                    year_min=year_min,
                    year_max=year_max,
                    venue=venue,
                    min_citation_count=min_citations,
                    max_citation_count=max_citations,
                )

            return bm25_results, semantic_results
        except Exception as e:
            logger.error(f"Database split search failed: {e}")
            return [], []

    def _fuse_rankings_with_rrf(
        self,
        paper_rankings: List[List[tuple[DBPaper, float]]],
        k: int = 60,
        limit: int = 100,
    ) -> List[tuple[DBPaper, float]]:
        rrf_scores: Dict[str, float] = {}
        paper_map: Dict[str, DBPaper] = {}

        for ranking in paper_rankings:
            for rank, (paper, _) in enumerate(ranking, start=1):
                pid = paper.paper_id
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (k + rank))
                if pid not in paper_map:
                    paper_map[pid] = paper

        ranked_ids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)[:limit]
        return [(paper_map[pid], rrf_scores[pid]) for pid in ranked_ids]

    async def _chunk_search(
        self,
        query: str,
        paper_ids: List[str],
        top_chunks: int,
        intent: Optional[QueryIntent] = None,
    ) -> List[ChunkRetrieved]:
        try:
            bm25_weight = 0.4
            semantic_weight = 0.6
            
            if intent == QueryIntent.FOUNDATIONAL:
                bm25_weight = 0.6
                semantic_weight = 0.4
                
            return await self.chunk_service.hybrid_search_chunks(
                query=query,
                paper_ids=paper_ids,
                limit=top_chunks,
                bm25_weight=bm25_weight,
                semantic_weight=semantic_weight,
            )
        except Exception as e:
            logger.error(f"Chunk search failed: {e}")
            return []

    async def _rank_papers(
        self,
        query: str,
        papers: List[DBPaper],
        chunks: List[ChunkRetrieved],
        paper_hybrid_scores: Dict[str, float],
        intent: Optional[QueryIntent] = None,
    ) -> List[RankedPaper]:
        try:
            weights = {
                'relevance': 0.7,
                'authority': 0.3,
            }
            return self.ranking_service.rank_papers(
                query=query,
                papers=papers,
                chunks=chunks,
                weights=weights,
            )
        except Exception as e:
            logger.error(f"Paper ranking failed: {e}")
            return [
                RankedPaper(
                    id=p.id,
                    paper_id=p.paper_id,
                    paper=p,
                    relevance_score=paper_hybrid_scores.get(p.paper_id, 0.0),
                    ranking_scores={"hybrid_score": paper_hybrid_scores.get(p.paper_id, 0.0)}
                )
                for p in papers
            ]

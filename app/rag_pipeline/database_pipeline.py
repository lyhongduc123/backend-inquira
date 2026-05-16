# app/rag_pipeline/database_pipeline.py
"""
Database-Only Search Pipeline

This pipeline searches exclusively in the existing database without retrieving
new papers from external APIs (S2/OA). It's optimized for fast searches when
you already have papers cached.

Features:
- No external API calls
- Fast BM25 + semantic search in database
- Filter support (author, year, venue, citation count)
- Intent-based optimization
- Chunk-level search within filtered papers
"""

from typing import AsyncGenerator, List, Optional, Dict, Any, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm import get_llm_service
from app.core.container import ServiceContainer
from app.domain.papers.repository import LoadOptions
from app.domain.chunks.types import ChunkRetrieved
from app.core.singletons import get_ranking_service
from app.models.papers import DBPaper
from app.search.types import RankedPaper
from app.llm.schemas import QueryIntent

from app.rag_pipeline.schemas import (
    RAGPipelineContext,
    RAGPipelineEvent,
    RAGResult,
    RAGEventType,
    SearchWorkflowConfig,
)
from app.extensions.logger import create_logger
from app.rag_pipeline.data_collector import get_data_collector
from app.search import (
    append_missing_abstract_chunks,
    parse_search_filter_options,
    reciprocal_rank_fusion,
)

logger = create_logger(__name__)


class DatabasePipeline:
    """
    Database-Only Search Pipeline.

    Searches exclusively in the existing database without external API calls.
    Ideal for:
    - Fast searches in cached papers
    - Filtered searches (author, year, venue)
    - Internal knowledge base queries
    """

    def __init__(
        self,
        db_session: AsyncSession,
        container: Optional[ServiceContainer] = None,
        llm_service=None,
        enable_data_collection: bool = False,
    ):
        """Initialize Database Pipeline."""
        self.db_session = db_session
        self.container = container or ServiceContainer(db_session)

        # Core services from container
        self.repository = self.container.paper_repository
        self.paper_service = self.container.paper_service
        self.chunk_service = self.container.chunk_service
        self.ranking_service = self.container.ranking_service

        # Optional overrides
        self.llm = llm_service or get_llm_service()
        self.data_collector = get_data_collector(enabled=enable_data_collection)

    async def run_database_search_workflow(
        self, config: SearchWorkflowConfig
    ) -> AsyncGenerator[RAGPipelineEvent, None]:
        """
        Database-only search workflow.

        Args:
            config: SearchWorkflowConfig containing query, filters, and parameters

        Yields:
            RAGPipelineEvent with progress and results
        """
        # Start data collection
        self.data_collector.start_execution(
            query=config.query,
            pipeline_type="database",
            conversation_id=config.conversation_id,
            filters=config.filters,
            config={
                "top_papers": config.top_papers,
                "top_chunks": config.top_chunks,
                "enable_reranking": config.enable_reranking,
                "enable_paper_ranking": config.enable_paper_ranking,
                "relevance_threshold": config.relevance_threshold,
            },
        )

        ctx = RAGPipelineContext(
            query=config.query,
            search_queries=config.search_queries or [config.query],
        )

        intent = QueryIntent.COMPREHENSIVE_SEARCH
        if config and config.intent:
            intent = config.intent
            logger.info(f"Query intent: {intent.value}")

        self.data_collector.record_decomposition(
            queries=ctx.search_queries,
            intent=intent,
        )

        merged_filters = config.filters or {}
        if config.filters:
            merged_filters.update(config.filters)

        yield RAGPipelineEvent(
            type=RAGEventType.SEARCHING,
            data={
                "queries": ctx.search_queries or [config.query],
                "original": config.query,
                "intent": intent.value,
                "filters": merged_filters,
            },
        )

        # yield RAGPipelineEvent(
        #     type=RAGEventType.PROCESSING,
        #     data={"message": "Finding relevance papers and documents..."},
        # )

        paper_rankings: List[List[tuple[DBPaper, float]]] = []
        for search_query in (ctx.search_queries or [config.query]):
            hybrid_results = await self._run_hybrid_search(
                search_query,
                config.filters,
                intent=intent,
            )
            if hybrid_results:
                paper_rankings.append(hybrid_results)

        db_papers_with_scores = self._fuse_rankings_with_rrf(
            paper_rankings=paper_rankings,
            k=60,
            limit=config.top_papers * 2,
        )

        logger.info(
            f"After RRF deduplication: {len(db_papers_with_scores)} unique papers"
        )

        if not db_papers_with_scores:
            logger.warning("No papers found in database")
            yield RAGPipelineEvent(
                type=RAGEventType.RESULT, data=RAGResult(papers=[], chunks=[])
            )
            return

        ctx.papers_with_hybrid_scores = db_papers_with_scores
        logger.info(f"Database search returned {len(db_papers_with_scores)} papers")

        self.data_collector.record_papers(
            papers=[], papers_with_scores=db_papers_with_scores
        )

        paper_ids = [p.paper_id for p, _ in db_papers_with_scores]
        if paper_ids:
            yield RAGPipelineEvent(
                type=RAGEventType.PROCESSING,
                data={"message": "Searching chunks in filtered papers"},
            )

            chunks = await self._chunk_search(
                query=config.query,
                paper_ids=paper_ids,
                top_chunks=config.top_chunks,
                intent=intent,
            )

            chunks_before_abstracts = len(chunks)
            append_missing_abstract_chunks(
                chunks,
                db_papers_with_scores,
                score_multiplier=0.8,
            )
            abstract_count = len(chunks) - chunks_before_abstracts

            if abstract_count:
                logger.info(
                    f"Creating {abstract_count} virtual abstract chunks for papers without chunks"
                )

            ctx.chunks = chunks
            logger.info(
                f"Found {len(chunks)} total chunks ({chunks_before_abstracts} real + {abstract_count} abstract)"
            )
            self.data_collector.record_chunks(chunks)

        if config.enable_reranking and ctx.chunks:
            try:
                ctx.chunks = self.ranking_service.rerank_chunks(
                    config.query, ctx.chunks
                )
                logger.info(f"Reranked {len(ctx.chunks)} chunks")
            except Exception as e:
                logger.error(f"Error reranking chunks: {e}")

        if config.enable_paper_ranking and ctx.papers_with_hybrid_scores:
            yield RAGPipelineEvent(
                type=RAGEventType.RANKING,
                data={
                    "total_papers": len(ctx.papers_with_hybrid_scores),
                    "total_chunks": len(ctx.chunks),
                },
            )

            try:
                paper_ids_to_rank = [
                    p.paper_id
                    for p, _ in ctx.papers_with_hybrid_scores[: config.top_papers]
                ]
                enriched_papers, _ = await self.repository.get_papers(
                    paper_ids=paper_ids_to_rank,
                    load_options=LoadOptions(
                        authors=True, journal=True, institutions=True
                    ),
                )

                paper_hybrid_scores = {
                    p.paper_id: score for p, score in ctx.papers_with_hybrid_scores
                }

                ranked_papers = await self._rank_papers(
                    query=config.query,
                    papers=enriched_papers,
                    chunks=ctx.chunks,
                    paper_hybrid_scores=paper_hybrid_scores,
                    intent=intent,
                )

                ctx.result_papers = ranked_papers[:config.top_papers]
                logger.info(f"Ranked {len(ranked_papers)} papers")

                self.data_collector.record_ranking(
                    ranked_papers=ranked_papers,
                    weights={"relevance": 0.7, "authority": 0.3},
                )
            except Exception as e:
                logger.error(f"Error during paper ranking: {e}")
                self.data_collector.record_error(f"Ranking error: {str(e)}")
                # Fallback: convert to RankedPaper without ranking
                ctx.result_papers = [
                    RankedPaper(
                        id=p.id,
                        paper_id=p.paper_id,
                        paper=p,
                        relevance_score=score,
                        ranking_scores={"hybrid_score": score},
                    )
                    for p, score in ctx.papers_with_hybrid_scores[:config.top_papers]
                ]

        result_paper_ids = {
            str(ranked.paper_id)
            for ranked in (ctx.result_papers or [])
        }
        ctx.chunks = [
            chunk for chunk in ctx.chunks if str(chunk.paper_id) in result_paper_ids
        ]

        saved_path = self.data_collector.end_execution()
        if saved_path:
            logger.info(f"Pipeline execution data saved to: {saved_path}")

        yield RAGPipelineEvent(
            type=RAGEventType.RESULT,
            data=RAGResult(
                papers=ctx.result_papers,
                chunks=ctx.chunks[:config.top_chunks],
            ),
        )

    async def _run_bm25_search(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[tuple[DBPaper, float]]:
        """Run BM25 search in database with optional filters."""
        try:
            results = await self.paper_service.bm25_search(
                query=query,
                limit=100,
                filter_options=parse_search_filter_options(filters),
            )
            logger.debug(
                f"BM25 search for query '{query[:50]}...' returned {len(results)} papers"
            )
            return results
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            await self.db_session.rollback()
            return []

    async def _run_semantic_search(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[tuple[DBPaper, float]]:
        """Run semantic search in database with optional filters."""
        try:
            results = await self.paper_service.semantic_search(
                query=query,
                limit=100,
                filter_options=parse_search_filter_options(filters),
            )
            logger.debug(
                f"Semantic search for query '{query[:50]}...' returned {len(results)} papers"
            )
            return results
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            await self.db_session.rollback()
            return []
        
    async def _run_hybrid_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        intent: Optional[QueryIntent] = None,
    ) -> List[tuple[DBPaper, float]]:
        try:
            results = await self.paper_service.hybrid_search(
                query=query,
                limit=100,
                filter_options=parse_search_filter_options(filters),
            )
            return results
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []


    def _get_hybrid_score_weight(self, intent: Optional[QueryIntent] = None) -> float:
        if intent == QueryIntent.FOUNDATIONAL:
            return 0.3
        if intent == QueryIntent.COMPARISON:
            return 0.35
        if intent == QueryIntent.AUTHOR_PAPERS:
            return 0.15
        return 0.25


    def _fuse_rankings_with_rrf(
        self,
        paper_rankings: List[List[tuple[DBPaper, float]]],
        k: int = 60,
        limit: int = 100,
    ) -> List[tuple[DBPaper, float]]:
        """Fuse multiple ranked lists with Reciprocal Rank Fusion (RRF)."""
        return reciprocal_rank_fusion(
            paper_rankings,
            key=lambda paper: paper.paper_id,
            k=k,
            limit=limit,
        )

    async def _chunk_search(
        self,
        query: str,
        paper_ids: List[str],
        top_chunks: int,
        intent: Optional[QueryIntent] = None,
    ) -> List[ChunkRetrieved]:
        """Search for relevant chunks in specified papers."""
        try:
            bm25_weight = 0.4
            semantic_weight = 0.6

            if intent == QueryIntent.FOUNDATIONAL:
                bm25_weight = 0.6
                semantic_weight = 0.4

            chunks = await self.chunk_service.hybrid_search_chunks(
                query=query,
                paper_ids=paper_ids,
                limit=top_chunks,
                bm25_weight=bm25_weight,
                semantic_weight=semantic_weight,
            )

            return chunks

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
        """Rank papers using comprehensive scoring."""
        try:
            weights = {
                "relevance": 0.7,
                "authority": 0.3,
            }
            ranked_papers = self.ranking_service.rank_papers(
                query=query,
                papers=papers,
                chunks=chunks,
                weights=weights,
            )

            hybrid_weight = self._get_hybrid_score_weight(intent)
            if hybrid_weight > 0:
                for ranked_paper in ranked_papers:
                    hybrid_score = paper_hybrid_scores.get(ranked_paper.paper_id, 0.0)
                    ranked_paper.relevance_score = (
                        ranked_paper.relevance_score * (1 - hybrid_weight)
                    ) + (hybrid_score * hybrid_weight * 100)
                    ranked_paper.ranking_scores["hybrid_score"] = hybrid_score

                ranked_papers.sort(key=lambda r: r.relevance_score, reverse=True)

            return ranked_papers
        except Exception as e:
            logger.error(f"Paper ranking failed: {e}")
            # Fallback: create RankedPaper objects with basic scores
            return [
                RankedPaper(
                    id=p.id,
                    paper_id=p.paper_id,
                    paper=p,
                    relevance_score=paper_hybrid_scores.get(p.paper_id, 0.0),
                    ranking_scores={
                        "hybrid_score": paper_hybrid_scores.get(p.paper_id, 0.0)
                    },
                )
                for p in papers
            ]

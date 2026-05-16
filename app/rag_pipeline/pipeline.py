"""
Unified RAG Pipeline - Database and Scoped Search

Consolidates database-only and scoped paper retrieval with configurable routing.
Routes between SEARCH_DATABASE (all papers) and SEARCH_SCOPED (explicit paper IDs).
"""

from enum import Enum
from typing import AsyncGenerator, List, Optional, Dict, Any, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.container import ServiceContainer
from app.domain.papers.repository import LoadOptions
from app.domain.chunks.types import ChunkRetrieved
from app.llm.schemas.chat import QueryIntent
from app.search.types import RankedPaper
from app.rag_pipeline.schemas import (
    RAGPipelineEvent,
    RAGResult,
    RAGEventType,
    SearchWorkflowConfig,
)
from app.extensions.logger import create_logger
from app.search import (
    append_missing_abstract_chunks,
    parse_search_filter_options,
    reciprocal_rank_fusion,
)

logger = create_logger(__name__)


class PipelineType(str, Enum):
    """Pipeline execution mode."""
    SEARCH_DATABASE = "database"
    SEARCH_SCOPED = "scoped"


class Pipeline:
    """Unified RAG Pipeline: database-only and scoped search modes."""

    def __init__(
        self,
        db_session: AsyncSession,
        container: Optional[ServiceContainer] = None,
    ):
        self.db_session = db_session
        self.container = container or ServiceContainer(db_session)
        self.repository = self.container.paper_repository
        self.search = self.container.local_search_service
        self.ranking_service = self.container.ranking_service
        self.llm = self.container.llm_service

    async def run_search_workflow(
        self,
        pipeline_type: PipelineType,
        config: SearchWorkflowConfig,
        paper_ids: Optional[List[str]] = None,
    ) -> AsyncGenerator[RAGPipelineEvent, None]:
        """Route to appropriate search mode."""
        if pipeline_type == PipelineType.SEARCH_DATABASE:
            async for event in self._run_database_workflow(config):
                yield event
        elif pipeline_type == PipelineType.SEARCH_SCOPED:
            async for event in self._run_scoped_workflow(config, paper_ids or []):
                yield event

    async def run_database_search_workflow(
        self, config: SearchWorkflowConfig
    ) -> AsyncGenerator[RAGPipelineEvent, None]:
        """Backward-compat wrapper: run database-only search."""
        async for event in self.run_search_workflow(PipelineType.SEARCH_DATABASE, config):
            yield event

    async def run_scoped_search_workflow(
        self,
        query: str,
        paper_ids: List[str],
        top_chunks: int = 40,
        top_papers: int = 20,
        enable_reranking: bool = True,
    ) -> AsyncGenerator[RAGPipelineEvent, None]:
        """Backward-compat wrapper: run scoped search with individual parameters."""
        config = SearchWorkflowConfig(
            query=query,
            search_queries=[],
            intent=None,
            filters=None,
            top_papers=top_papers,
            top_chunks=top_chunks,
            enable_reranking=enable_reranking,
            enable_paper_ranking=True,
        )
        async for event in self.run_search_workflow(PipelineType.SEARCH_SCOPED, config, paper_ids):
            yield event


    async def _run_database_workflow(
        self, config: SearchWorkflowConfig
    ) -> AsyncGenerator[RAGPipelineEvent, None]:
        """Full database search: hybrid retrieval → ranking → chunks."""
        intent = config.intent or QueryIntent.COMPREHENSIVE_SEARCH
        search_queries = config.search_queries or [config.query]
        
        yield RAGPipelineEvent(
            type=RAGEventType.SEARCHING,
            data={
                "queries": search_queries,
                "original": config.query,
                "intent": intent.value,
                "filters": config.filters or {},
            },
        )

        paper_rankings: List[List[Tuple[Any, float]]] = []
        for sq in search_queries:
            results = await self._run_hybrid_search(sq, config.filters, intent)
            if results:
                paper_rankings.append(results)

        papers_with_scores = self._fuse_rankings_rrf(paper_rankings, limit=config.top_papers * 2)
        
        if not papers_with_scores:
            yield RAGPipelineEvent(type=RAGEventType.RESULT, data=RAGResult(papers=[], chunks=[]))
            return

        logger.info(f"Database search: {len(papers_with_scores)} unique papers after RRF")

        paper_ids = [str(p.paper_id) for p, _ in papers_with_scores]
        chunks = await self._chunk_search(config.query, paper_ids, config.top_chunks, intent)

        append_missing_abstract_chunks(
            chunks,
            papers_with_scores,
            score_multiplier=0.0,
        )

        if config.enable_reranking and chunks:
            try:
                chunks = self.ranking_service.rerank_chunks(config.query, chunks)
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")

        if config.enable_paper_ranking and papers_with_scores:
            yield RAGPipelineEvent(
                type=RAGEventType.RANKING,
                data={
                    "total_papers": len(papers_with_scores),
                    "total_chunks": len(chunks),
                },
            )

            try:
                enriched = await self.repository.get_papers(
                    paper_ids=paper_ids[:config.top_papers],
                    load_options=LoadOptions(authors=True, journal=True, institutions=True),
                )
                hybrid_scores = {str(p.paper_id): s for p, s in papers_with_scores}
                ranked = await self._rank_papers(
                    config.query, enriched[0], chunks, hybrid_scores, intent
                )
                result_papers = ranked[:config.top_papers]
            except Exception as e:
                logger.error(f"Paper ranking failed: {e}")
                result_papers = [
                    RankedPaper(
                        id=int(p.id),
                        paper_id=str(p.paper_id),
                        paper=p,
                        relevance_score=s,
                        ranking_scores={"hybrid_score": s},
                    )
                    for p, s in papers_with_scores[:config.top_papers]
                ]
        else:
            result_papers = [
                RankedPaper(
                    id=int(p.id),
                    paper_id=str(p.paper_id),
                    paper=p,
                    relevance_score=s,
                    ranking_scores={"hybrid_score": s},
                )
                for p, s in papers_with_scores[:config.top_papers]
            ]

        result_ids = {str(rp.paper_id) for rp in result_papers}
        chunks = [c for c in chunks if str(c.paper_id) in result_ids]

        yield RAGPipelineEvent(
            type=RAGEventType.RESULT,
            data=RAGResult(papers=result_papers, chunks=chunks[:config.top_chunks]),
        )

    async def _run_scoped_workflow(
        self,
        config: SearchWorkflowConfig,
        paper_ids: List[str],
    ) -> AsyncGenerator[RAGPipelineEvent, None]:
        """Lightweight search over explicit paper set."""
        normalized_ids = [str(pid).strip() for pid in paper_ids if str(pid).strip()]
        
        if not normalized_ids:
            yield RAGPipelineEvent(type=RAGEventType.RESULT, data=RAGResult(papers=[], chunks=[]))
            return

        yield RAGPipelineEvent(
            type=RAGEventType.PROCESSING,
            data={"message": "Rewriting query for scoped search"},
        )

        rewritten = await self._rewrite_query_lightweight(config.query)

        yield RAGPipelineEvent(
            type=RAGEventType.SEARCHING,
            data={
                "queries": [rewritten],
                "original": config.query,
                "paper_ids": normalized_ids,
                "scope": "provided",
            },
        )

        yield RAGPipelineEvent(
            type=RAGEventType.PROCESSING,
            data={"message": "Searching chunks in scoped papers"},
        )

        chunks = await self.search.chunks.hybrid_search(
            query=rewritten,
            paper_ids=normalized_ids,
            limit=config.top_chunks,
            bm25_weight=0.4,
            semantic_weight=0.6,
        )

        papers, _ = await self.repository.get_papers(
            skip=0,
            limit=len(normalized_ids),
            paper_ids=normalized_ids,
            load_options=LoadOptions(),
        )

        append_missing_abstract_chunks(
            chunks,
            [(paper, 0.0) for paper in papers],
            score_multiplier=0.0,
        )

        if config.enable_reranking and chunks:
            try:
                chunks = self.ranking_service.rerank_chunks(rewritten, chunks)
            except Exception as e:
                logger.warning(f"Scoped reranking failed: {e}")

        yield RAGPipelineEvent(
            type=RAGEventType.RANKING,
            data={
                "total_papers": len(normalized_ids),
                "total_chunks": len(chunks),
                "scope": "provided",
            },
        )

        ranked = await self._rank_scoped_papers(normalized_ids, chunks, config.top_papers)

        yield RAGPipelineEvent(
            type=RAGEventType.RESULT,
            data=RAGResult(papers=ranked, chunks=chunks[:config.top_chunks]),
        )

    async def _run_hybrid_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        intent: Optional[QueryIntent] = None,
    ) -> List[Tuple[Any, float]]:
        """Hybrid BM25 + semantic search with filters."""
        try:
            results = await self.search.papers.hybrid_search(
                query=query,
                limit=100,
                filter_options=parse_search_filter_options(filters),
            )
            logger.debug(f"Hybrid search '{query[:40]}': {len(results)} papers")
            return results
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            await self.db_session.rollback()
            return []

    async def _chunk_search(
        self,
        query: str,
        paper_ids: List[str],
        top_chunks: int,
        intent: Optional[QueryIntent] = None,
    ) -> List[ChunkRetrieved]:
        """Search chunks in specified papers."""
        try:
            chunks = await self.search.chunks.hybrid_search(
                query=query,
                paper_ids=paper_ids,
                limit=top_chunks,
                bm25_weight=0.4,
                semantic_weight=0.6,
            )
            return chunks
        except Exception as e:
            logger.error(f"Chunk search failed: {e}")
            return []

    async def _rank_papers(
        self,
        query: str,
        papers: List[Any],
        chunks: List[ChunkRetrieved],
        hybrid_scores: Dict[str, float],
        intent: Optional[QueryIntent] = None,
    ) -> List[RankedPaper]:
        """Rank papers using hybrid + authority scoring."""
        try:
            ranked = self.ranking_service.rank_papers(
                query=query,
                papers=papers,
                chunks=chunks,
                weights={"relevance": 0.7, "authority": 0.3},
            )
            return ranked
        except Exception as e:
            logger.error(f"Paper ranking failed: {e}")
            return [
                RankedPaper(
                    id=int(p.id),
                    paper_id=str(p.paper_id),
                    paper=p,
                    relevance_score=hybrid_scores.get(str(p.paper_id), 0.0),
                    ranking_scores={"hybrid_score": hybrid_scores.get(str(p.paper_id), 0.0)},
                )
                for p in papers
            ]

    async def _rank_scoped_papers(
        self,
        paper_ids: List[str],
        chunks: List[ChunkRetrieved],
        top_papers: int,
    ) -> List[RankedPaper]:
        """Rank papers in scoped set by chunk relevance aggregation."""
        papers, _ = await self.repository.get_papers(
            skip=0,
            limit=max(top_papers, 1),
            paper_ids=paper_ids,
            load_options=LoadOptions(authors=True, journal=True),
        )

        if not papers:
            return []

        score_map = {str(p.paper_id): 0.0 for p in papers}
        for chunk in chunks:
            pid = str(getattr(chunk, "paper_id", ""))
            if pid in score_map:
                score_map[pid] += float(getattr(chunk, "relevance_score", 0.0) or 0.0)

        ranked = [
            RankedPaper(
                id=int(p.id),
                paper_id=str(p.paper_id),
                paper=p,
                relevance_score=score_map.get(str(p.paper_id), 0.0),
                ranking_scores={"scoped_chunk_relevance": score_map.get(str(p.paper_id), 0.0)},
            )
            for p in papers
        ]

        ranked.sort(key=lambda x: x.relevance_score, reverse=True)
        return ranked[:top_papers]

    async def _rewrite_query_lightweight(self, query: str) -> str:
        """Lightweight query clarification for scoped search."""
        try:
            breakdown = await self.llm.decompose_user_query_v2(
                user_question=query,
                num_subtopics=1,
                conversation_history=None,
            )
            return breakdown.clarified_question or query
        except Exception as e:
            logger.warning(f"Query rewrite failed, using original: {e}")
            return query

    def _fuse_rankings_rrf(
        self,
        rankings: List[List[Tuple[Any, float]]],
        k: int = 60,
        limit: int = 100,
    ) -> List[Tuple[Any, float]]:
        """Reciprocal Rank Fusion: fuse multiple ranked lists."""
        return reciprocal_rank_fusion(
            rankings,
            key=lambda paper: paper.paper_id,
            k=k,
            limit=limit,
        )

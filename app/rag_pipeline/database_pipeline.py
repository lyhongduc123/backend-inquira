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

import asyncio
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Dict, Any, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm import get_llm_service
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
from app.rag_pipeline.data_collector import get_data_collector

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
        self.chunk_service = self.container.chunk_service
        self.ranking_service = self.container.ranking_service
        
        # Optional overrides
        self.llm = llm_service or get_llm_service()
        self.data_collector = get_data_collector(enabled=enable_data_collection)

    async def run_database_search_workflow(
        self,
        query: str,
        top_papers: int = 50,
        top_chunks: int = 40,
        filters: Optional[Dict[str, Any]] = None,
        enable_reranking: bool = True,
        enable_paper_ranking: bool = True,
        relevance_threshold: float = 0.3,
        conversation_id: Optional[str] = None,
    ) -> AsyncGenerator[RAGPipelineEvent, None]:
        """
        Database-only search workflow.

        Args:
            query: User question
            top_papers: Max papers to return
            top_chunks: Max chunks to return
            filters: Search filters:
                - author: str (author name)
                - year_min: int
                - year_max: int
                - venue: str (venue name)
                - min_citations: int
                - max_citations: int
            enable_reranking: Whether to rerank chunks
            enable_paper_ranking: Whether to apply paper ranking
            relevance_threshold: Minimum relevance score
            conversation_id: Optional conversation ID for context

        Yields:
            RAGPipelineEvent with progress and results
        """
        # Start data collection
        self.data_collector.start_execution(
            query=query,
            pipeline_type="database",
            conversation_id=conversation_id,
            filters=filters,
            config={
                "top_papers": top_papers,
                "top_chunks": top_chunks,
                "enable_reranking": enable_reranking,
                "enable_paper_ranking": enable_paper_ranking,
                "relevance_threshold": relevance_threshold,
            }
        )
        
        ctx = RAGPipelineContext(query)
        
        # Load conversation history if provided
        conversation_history = None
        if conversation_id:
            try:
                from app.domain.conversations.context_manager import ConversationContextManager
                context_mgr = ConversationContextManager()
                conversation_history, _ = await context_mgr.get_conversation_context(
                    conversation_id=conversation_id,
                    db_session=self.db_session,
                    include_current_query=False
                )
                logger.info(f"Loaded {len(conversation_history)} messages for context")
                self.data_collector.record_conversation_context(len(conversation_history))
            except Exception as e:
                logger.warning(f"Failed to load conversation history: {e}")
        
        # Step 1: Query decomposition with intent
        yield RAGPipelineEvent(
            type=RAGEventType.PROCESSING,
            data={"message": "Analyzing query and detecting intent"},
        )
        
        ctx = await self._decompose_query(ctx, conversation_history)
        breakdown_response = ctx.breakdown_response
        
        intent = QueryIntent.COMPREHENSIVE_SEARCH
        if breakdown_response and breakdown_response.intent:
            intent = breakdown_response.intent
            logger.info(f"Query intent: {intent.value}")
        
        # Record decomposition
        self.data_collector.record_decomposition(
            queries=ctx.search_queries or [query],
            intent=intent,
            breakdown_response=breakdown_response
        )
        
        # Merge filters from breakdown with provided filters
        merged_filters = filters or {}
        if breakdown_response and breakdown_response.filters:
            merged_filters.update(breakdown_response.filters)
        
        yield RAGPipelineEvent(
            type=RAGEventType.SEARCHING,
            data={
                "queries": ctx.search_queries or [query],
                "original": query,
                "intent": intent.value,
                "filters": merged_filters,
            },
        )

        # Step 2: Database split retrieval with filters using all search queries
        yield RAGPipelineEvent(
            type=RAGEventType.PROCESSING,
            data={"message": "Searching in database with filters"},
        )
        
        # Collect raw rankings from each query/modality, then do ONE global RRF
        search_queries = ctx.search_queries or [query]
        paper_rankings: List[List[tuple[DBPaper, float]]] = []
        
        for search_query in search_queries:
            bm25_results, semantic_results = await self._database_split_search(
                query=search_query,
                limit=top_papers * 2,  # Get more for filtering
                filters=merged_filters,
                intent=intent,
            )
            if bm25_results:
                paper_rankings.append(bm25_results)
            if semantic_results:
                paper_rankings.append(semantic_results)

            logger.info(
                f"Query '{search_query[:50]}...' retrieved "
                f"bm25={len(bm25_results)}, semantic={len(semantic_results)}"
            )

        db_papers_with_scores = self._fuse_rankings_with_rrf(
            paper_rankings=paper_rankings,
            k=60,
            limit=top_papers * 2,
        )
        
        logger.info(f"After RRF deduplication: {len(db_papers_with_scores)} unique papers")
        
        if not db_papers_with_scores:
            logger.warning("No papers found in database")
            yield RAGPipelineEvent(
                type=RAGEventType.RESULT,
                data=RAGResult(papers=[], chunks=[])
            )
            return
        
        ctx.papers_with_hybrid_scores = db_papers_with_scores
        logger.info(f"Database search returned {len(db_papers_with_scores)} papers")
        
        self.data_collector.record_papers(
            papers=[],
            papers_with_scores=db_papers_with_scores
        )
        
        paper_ids = [p.paper_id for p, _ in db_papers_with_scores[:top_papers]]
        
        if paper_ids:
            yield RAGPipelineEvent(
                type=RAGEventType.PROCESSING,
                data={"message": "Searching chunks in filtered papers"},
            )
            
            chunks = await self._chunk_search(
                query=breakdown_response.clarified_question if breakdown_response else query,
                paper_ids=paper_ids,
                top_chunks=top_chunks,
                intent=intent,
            )
            
            papers_with_chunks = {chunk.paper_id for chunk in chunks}
            papers_without_chunks = [
                (paper, score) for paper, score in db_papers_with_scores[:top_papers]
                if paper.paper_id not in papers_with_chunks and paper.abstract
            ]
            
            if papers_without_chunks:
                logger.info(f"Creating {len(papers_without_chunks)} virtual abstract chunks for papers without chunks")
                for paper, score in papers_without_chunks:
                    virtual_chunk = ChunkRetrieved(
                        chunk_id=f"{paper.paper_id}_abstract",
                        paper_id=paper.paper_id,
                        text=paper.abstract,
                        token_count=len(paper.abstract.split()),
                        chunk_index=0,
                        section_title="Abstract",  # Tag with 'Abstract' for LLM context
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
            logger.info(f"Found {len(chunks)} total chunks ({len(chunks) - len(papers_without_chunks)} real + {len(papers_without_chunks)} abstract)")
            self.data_collector.record_chunks(chunks)
        
        if enable_reranking and ctx.chunks:
            try:
                ctx.chunks = self.ranking_service.rerank_chunks(query, ctx.chunks)
                logger.info(f"Reranked {len(ctx.chunks)} chunks")
            except Exception as e:
                logger.error(f"Error reranking chunks: {e}")
        
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
                    query=query,
                    papers=enriched_papers,
                    chunks=ctx.chunks,
                    paper_hybrid_scores=paper_hybrid_scores,
                    intent=intent,
                )
                
                ctx.result_papers = ranked_papers[:top_papers]
                logger.info(f"Ranked {len(ranked_papers)} papers")
                
                self.data_collector.record_ranking(
                    ranked_papers=ranked_papers,
                    weights={'relevance': 0.7, 'authority': 0.3}
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
                        ranking_scores={"hybrid_score": score}
                    )
                    for p, score in ctx.papers_with_hybrid_scores[:top_papers]
                ]
        
        
        ctx.chunks = [chunk for chunk in ctx.chunks if chunk.paper_id in ctx.result_papers]
        
        saved_path = self.data_collector.end_execution()
        if saved_path:
            logger.info(f"Pipeline execution data saved to: {saved_path}")
        
        yield RAGPipelineEvent(
            type=RAGEventType.RESULT,
            data=RAGResult(
                papers=ctx.result_papers,
                chunks=ctx.chunks[:top_chunks],
            ),
        )

    async def _decompose_query(
        self,
        ctx: RAGPipelineContext,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> RAGPipelineContext:
        """Decompose query and detect intent."""
        try:
            breakdown = await self.llm.decompose_user_query_v2(
                user_question=ctx.query,
                num_subtopics=2,
                conversation_history=conversation_history,
            )
            ctx.breakdown_response = breakdown
            ctx.search_queries = breakdown.search_queries or [ctx.query]
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            ctx.search_queries = [ctx.query]
        
        return ctx

    async def _database_split_search(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
        intent: Optional[QueryIntent] = None,
    ) -> Tuple[List[tuple[DBPaper, float]], List[tuple[DBPaper, float]]]:
        """
        Perform split retrieval (BM25 + semantic) without fusion.

        Fusion is intentionally handled at pipeline level after collecting
        all rankings for all decomposed queries.
        """
        try:
            from app.processor.services.embeddings import get_embedding_service

            # Build filter dict for repository
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
            else:
                logger.warning(
                    "Embedding generation failed in database split search, "
                    "semantic branch skipped"
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
        """Fuse multiple ranked lists with Reciprocal Rank Fusion (RRF)."""
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
                'relevance': 0.7,
                'authority': 0.3,
            }
            ranked_papers = self.ranking_service.rank_papers(
                query=query,
                papers=papers,
                chunks=chunks,
                weights=weights,
            )
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
                    ranking_scores={"hybrid_score": paper_hybrid_scores.get(p.paper_id, 0.0)}
                )
                for p in papers
            ]

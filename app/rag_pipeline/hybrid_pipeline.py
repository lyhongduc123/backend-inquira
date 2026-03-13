# app/rag_pipeline/hybrid_pipeline.py
"""
Hybrid RAG Pipeline with BM25 + Semantic Search at both paper and chunk levels.

Architecture:
1. After retrieval from S2/OA, batch create papers with enrichment
2. Generate title+abstract embeddings and cache in DB
3. Hybrid search in database: BM25 (ts_vector) + semantic (pgvector)
4. Boost papers that match S2 retrieved results
5. Process top papers concurrently (max 5)
6. Hybrid chunk search: BM25 + semantic on full-text
7. Final ranking: 4 relevance scores (paper BM25, paper semantic, chunk BM25, chunk semantic)
8. Intent-based weight adjustment
"""

import asyncio
import gc
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm import get_llm_service
from app.processor.paper_processor import PaperProcessor
from app.retriever.service import RetrievalService, RetrievalServiceType
from app.domain.papers import PaperRepository, LoadOptions, PaperService
from app.domain.chunks import ChunkService, ChunkRepository
from app.domain.chunks.schemas import ChunkRetrieved
from app.core.singletons import get_ranking_service
from app.models.papers import DBPaper
from app.processor.schemas import RankedPaper
from app.llm.schemas import QueryIntent

from app.rag_pipeline.schemas import (
    PipelineResult,
    RAGPipelineContext,
    RAGPipelineEvent,
    RAGResult,
    RAGEventType,
)
from app.extensions.logger import create_logger

from app.rag_pipeline.utils import deduplicate_papers
from app.rag_pipeline.data_collector import get_data_collector

logger = create_logger(__name__)


class HybridPipeline:
    """
    Hybrid BM25 + Semantic RAG Pipeline.
    
    Key differences from standard pipeline:
    - Uses database-first approach (cached papers)
    - Hybrid BM25 + semantic search at paper level
    - Hybrid BM25 + semantic search at chunk level
    - Intent-based weight configuration
    - S2 result boosting
    """

    def __init__(
        self,
        db_session: AsyncSession,
        repository: Optional[PaperRepository] = None,
        retriever: Optional[RetrievalService] = None,
        processor: Optional[PaperProcessor] = None,
        llm_service=None,
        ranking_service=None,
        paper_service=None,
        enable_data_collection: bool = True,
    ):
        """Initialize Hybrid Pipeline with dependency injection."""
        self.db_session = db_session
        self.repository = repository or PaperRepository(db_session)
        self.chunk_repository = ChunkRepository(db_session)
        self.chunk_service = ChunkService(self.chunk_repository)
        self.retriever = retriever or RetrievalService(db=db_session)
        self.processor = processor or PaperProcessor(
            repository=self.repository,
            chunk_repository=self.chunk_repository,
            retrieval_service=self.retriever,
        )
        self.llm = llm_service or get_llm_service()
        self.ranking_service = ranking_service or get_ranking_service()
        self.data_collector = get_data_collector(enabled=enable_data_collection)
        self.paper_service = paper_service or PaperService(self.repository, self.retriever)

    async def run_hybrid_rag_workflow(
        self,
        query: str,
        max_subtopics: int = 3,
        per_subtopic_limit: int = 50,
        top_chunks: int = 40,
        filters: Optional[Dict[str, Any]] = None,
        enable_reranking: bool = True,
        enable_paper_ranking: bool = True,
        relevance_threshold: float = 0.3,
        conversation_id: Optional[str] = None,
    ):
        """
        Hybrid RAG workflow with BM25 + semantic search.

        Args:
            query: User question
            max_subtopics: Max subtopics to generate
            per_subtopic_limit: Max papers per subtopic from S2
            top_chunks: Max chunks to return
            filters: Optional filters
            enable_reranking: Whether to rerank chunks with cross-encoder
            enable_paper_ranking: Whether to apply comprehensive paper ranking
            relevance_threshold: Minimum semantic relevance score for papers
            conversation_id: Optional conversation ID for context
        """
        # Start data collection
        self.data_collector.start_execution(
            query=query,
            pipeline_type="hybrid",
            conversation_id=conversation_id,
            filters=filters,
            config={
                "max_subtopics": max_subtopics,
                "per_subtopic_limit": per_subtopic_limit,
                "top_chunks": top_chunks,
                "enable_reranking": enable_reranking,
                "enable_paper_ranking": enable_paper_ranking,
                "relevance_threshold": relevance_threshold,
            }
        )
        
        ctx = RAGPipelineContext(query)
        
        # Load conversation history if conversation_id provided
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
                logger.info(f"Loaded {len(conversation_history)} messages for query decomposition context")
                self.data_collector.record_conversation_context(len(conversation_history))
            except Exception as e:
                logger.warning(f"Failed to load conversation history for decomposition: {e}")
        
        # Step 1: Break down query and detect intent
        ctx = await self._break_down_query(ctx, max_subtopics, conversation_history)
        breakdown_response = ctx.breakdown_response
        
        intent = QueryIntent.COMPREHENSIVE_SEARCH
        if breakdown_response and breakdown_response.intent:
            intent = breakdown_response.intent
            logger.info(f"Query intent: {intent.value} (confidence: {breakdown_response.intent_confidence or 'N/A'})")
        
        # Record decomposition
        self.data_collector.record_decomposition(
            queries=ctx.search_queries,
            intent=intent,
            breakdown_response=breakdown_response
        )
        
        yield RAGPipelineEvent(
            type=RAGEventType.SEARCHING,
            data={"queries": ctx.search_queries, "original": query, "intent": intent.value},
        )

        # Step 2: Retrieve papers from S2/OA
        ctx = await self._retrieve_papers(ctx, per_subtopic_limit, filters)
        
        if not ctx.papers:
            self.data_collector.record_error("No papers retrieved from S2/OA")
            self.data_collector.end_execution()
            yield RAGPipelineEvent(type=RAGEventType.RESULT, data=RAGResult(papers=[], chunks=[]))
            return

        s2_paper_ids = {str(p.paper_id) for p in ctx.papers}
        logger.info(f"Retrieved {len(s2_paper_ids)} papers from S2/OA")
        
        # Record S2/OA papers (before DB search)
        # Note: These don't have DB scores yet, will record hybrid scores later

        # Step 3: Batch check existing papers and create missing ones
        yield RAGPipelineEvent(
            type=RAGEventType.PROCESSING,
            data={"message": "Batch creating/updating papers with enrichment", "total_papers": len(ctx.papers)},
        )
        
        await self._batch_create_papers(ctx)

        # Step 4: Generate and cache title+abstract embeddings
        yield RAGPipelineEvent(
            type=RAGEventType.PROCESSING,
            data={"message": "Generating embeddings for papers", "total_papers": len(ctx.papers)},
        )
        
        await self._generate_and_cache_embeddings(ctx)
        yield RAGPipelineEvent(
            type=RAGEventType.PROCESSING,
            data={"message": "Hybrid BM25 + semantic search in database"},
        )
        
        db_papers_with_scores = await self._hybrid_paper_search(
            ctx=ctx,
            s2_paper_ids=s2_paper_ids,
            limit=100,
            intent=intent
        )
        
        if not db_papers_with_scores:
            logger.warning("No papers found in hybrid search")
            self.data_collector.record_error("No papers found in hybrid search")
            self.data_collector.end_execution()
            yield RAGPipelineEvent(type=RAGEventType.RESULT, data=RAGResult(papers=[], chunks=[]))
            return
        
        ctx.papers_with_hybrid_scores = db_papers_with_scores
        logger.info(f"Hybrid search returned {len(db_papers_with_scores)} papers")
        
        # Record papers with hybrid search scores
        self.data_collector.record_papers(
            papers=[],
            papers_with_scores=db_papers_with_scores
        )
        
        top_papers_to_process = [p for p, _ in db_papers_with_scores[:10]] 
        
        async for event in self._process_papers_concurrent(ctx, top_papers_to_process):
            if isinstance(event, RAGPipelineEvent):
                yield event
            else:
                ctx = event
        if ctx.processed_paper_ids or ctx.papers_with_hybrid_scores:
            yield RAGPipelineEvent(
                type=RAGEventType.PROCESSING,
                data={"message": "Hybrid chunk search in top papers"},
            )
            
            if ctx.processed_paper_ids:
                chunk_search_paper_ids = ctx.processed_paper_ids
            else:
                chunk_search_paper_ids = [p.paper_id for p, _ in ctx.papers_with_hybrid_scores[:20]]
            
            logger.info(f"Searching chunks in {len(chunk_search_paper_ids)} top papers (intent: {intent.value})")
            
            ctx = await self._hybrid_chunk_search(
                ctx=ctx,
                query=ctx.breakdown_response.clarified_question if ctx.breakdown_response else ctx.query,
                top_chunks=top_chunks,
                paper_ids=chunk_search_paper_ids,
                intent=intent
            )
            
            # Record chunks
            if ctx.chunks:
                self.data_collector.record_chunks(ctx.chunks)

        if enable_reranking and ctx.chunks:
            try:
                ctx.chunks = self.ranking_service.rerank_chunks(query, ctx.chunks)
                logger.info(f"Reranked {len(ctx.chunks)} chunks")
            except Exception as e:
                logger.error(f"Error reranking chunks: {e}", exc_info=True)
                self.data_collector.record_error(f"Reranking error: {str(e)}")

        # Step 9: Final paper ranking with 4 relevance scores
        if enable_paper_ranking and ctx.papers_with_hybrid_scores:
            yield RAGPipelineEvent(
                type=RAGEventType.RANKING,
                data={"total_papers": len(ctx.papers_with_hybrid_scores), "total_chunks": len(ctx.chunks)},
            )

            try:
                # Get enriched papers for ranking
                paper_ids_to_rank = [p.paper_id for p, _ in ctx.papers_with_hybrid_scores]
                enriched_papers, _ = await self.repository.get_papers(
                    paper_ids=paper_ids_to_rank,
                    load_options=LoadOptions(authors=True, journal=True, institutions=True),
                )

                # Inject hybrid scores into ranking
                paper_hybrid_scores = {p.paper_id: score for p, score in ctx.papers_with_hybrid_scores}
                
                ranked_papers = await self._rank_papers_with_hybrid_scores(
                    query=query,
                    papers=enriched_papers,
                    chunks=ctx.chunks,
                    paper_hybrid_scores=paper_hybrid_scores,
                    intent=intent,
                )
                
                ctx.result_papers = ranked_papers
                logger.info(f"Ranked {len(ranked_papers)} papers with hybrid scores")
                
                # Record ranking
                self.data_collector.record_ranking(
                    ranked_papers=ranked_papers,
                    weights=None  # Hybrid scoring uses dynamic weights
                )
            except Exception as e:
                logger.error(f"Error during paper ranking: {e}", exc_info=True)
                self.data_collector.record_error(f"Ranking error: {str(e)}")
                ctx.result_papers = []
        else:
            ctx.result_papers = []

        # End data collection
        self.data_collector.end_execution()
        
        yield RAGPipelineEvent(
            type=RAGEventType.RESULT,
            data=RAGResult(papers=ctx.result_papers, chunks=ctx.chunks)
        )

    async def _break_down_query(self, ctx: RAGPipelineContext, max_subtopics: int, conversation_history: Optional[List[Dict[str, str]]] = None):
        """Break down query into sub-queries."""
        response = await self.llm.decompose_user_query(
            ctx.query, num_subtopics=max_subtopics, conversation_history=conversation_history
        )
        ctx.breakdown_response = response
        ctx.search_queries = response.search_queries
        return ctx

    async def _retrieve_papers(self, ctx: RAGPipelineContext, per_subtopic_limit: int, filters: Optional[Dict[str, Any]]):
        """
        Retrieve papers from S2/OA and add top referenced papers.
        
        Workflow:
        1. Fetch papers from S2/OA for each search query
        2. Deduplicate results
        3. Count references across all papers
        4. Fetch top 5 most referenced papers
        5. Add them to the results
        """
        all_papers = []
        references = []
        for search_query in ctx.search_queries:
            papers, metadata = await self.retriever.hybrid_search(
                query=search_query,
                semantic_limit=per_subtopic_limit,
                filters=filters,
            )
            all_papers.extend(papers)
            await asyncio.sleep(1)
            
        papers = deduplicate_papers(all_papers)
        
        for paper in papers:
            # Safe access for references - handle both dict and object attributes
            paper_refs = getattr(paper, 'references', None) or (paper.get('references') if isinstance(paper, dict) else None)
            if paper_refs:
                references.extend(paper_refs)
 
        from app.rag_pipeline.utils import count_and_rank_references
        
        top_references = count_and_rank_references(references, top_k=5)
        
        if top_references:
            logger.info(
                f"Top referenced papers: {[(ref_id, count) for ref_id, count, _ in top_references]}"
            )
   
            existing_paper_ids = {p.paper_id for p in papers}
            reference_ids_to_fetch = [
                ref_id for ref_id, _, _ in top_references 
                if ref_id not in existing_paper_ids
            ]
            try:
                if reference_ids_to_fetch:
                    fetched_refs = await self.retriever.get_multiple_papers(reference_ids_to_fetch)
                    papers.extend(fetched_refs)
                    logger.info(f"Fetched {len(fetched_refs)} top referenced papers from retriever")
            except Exception as e:
                logger.error(f"Error fetching top referenced papers: {e}", exc_info=True)
                self.data_collector.record_error(f"Error fetching referenced papers: {str(e)}")
           
        ctx.papers = papers  # type: ignore
        return ctx

    async def _batch_create_papers(self, ctx: RAGPipelineContext):
        """Batch create papers with enrichment."""
        from app.domain.papers import PaperService
        
        paper_service = PaperService(self.repository, self.retriever)
        
        try:
            created_papers = await paper_service.batch_create_papers_from_schema(
                papers=ctx.papers,  # type: ignore
                enrich=True
            )
            logger.info(f"Batch created/updated {len(created_papers)} papers with enrichment")
        except Exception as e:
            logger.error(f"Batch paper creation failed: {e}", exc_info=True)
            self.data_collector.record_error(f"Batch paper creation failed: {str(e)}")

    async def _generate_and_cache_embeddings(self, ctx: RAGPipelineContext):
        """Generate title+abstract embeddings and cache in database."""
        # Check which papers already have embeddings in DB
        paper_ids = [str(p.paper_id) for p in ctx.papers]
        existing_embeddings = await self.repository.get_paper_embeddings(paper_ids)
        
        # Filter papers that need embeddings
        papers_needing_embeddings = [
            p for p in ctx.papers 
            if str(p.paper_id) not in existing_embeddings or existing_embeddings[str(p.paper_id)] is None
        ]
        
        if not papers_needing_embeddings:
            logger.info(f"All {len(ctx.papers)} papers already have embeddings cached")
            return
        
        logger.info(f"Generating embeddings for {len(papers_needing_embeddings)}/{len(ctx.papers)} papers")
        
        # Generate embeddings only for papers that need them
        papers_with_new_embeddings = await self.processor.generate_paper_embeddings(papers_needing_embeddings)  # type: ignore
        
        # Cache new embeddings in database
        paper_embeddings = {
            str(p.paper_id): p.embedding
            for p in papers_with_new_embeddings
            if p.embedding is not None
        }
        
        if paper_embeddings:
            await self.repository.bulk_update_paper_embeddings(paper_embeddings)
            logger.info(f"Cached {len(paper_embeddings)} new embeddings in database")

    async def _hybrid_paper_search(
        self,
        ctx: RAGPipelineContext,
        s2_paper_ids: set,
        limit: int,
        intent: QueryIntent
    ) -> List[tuple[DBPaper, float]]:
        """Hybrid BM25 + semantic search with S2 boosting."""
        # Get weights based on intent
        bm25_weight, semantic_weight, s2_boost = self._get_paper_search_weights(intent)
        
        if not ctx.breakdown_response or not ctx.breakdown_response.keyword_queries:
            logger.warning("No keyword queries available for hybrid search, skipping paper search")
            return []
        
        logger.debug(f"Hybrid paper search with query: '{ctx.breakdown_response.clarified_question}'")
        logger.debug(f"Using weights - BM25: {bm25_weight}, Semantic: {semantic_weight}, S2 boost: {s2_boost}")

        papers_with_scores = await self.paper_service.hybrid_search_papers(
            query=ctx.breakdown_response.clarified_question,
            limit=limit,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
        )
        
        logger.debug(f"Hybrid search returned {len(papers_with_scores)} papers before boosting")
    
        boosted_papers = []
        for paper, score in papers_with_scores:
            final_score = score
            if paper.paper_id in s2_paper_ids:
                final_score += s2_boost
                logger.debug(f"Paper {paper.paper_id} boosted (S2 match): {score:.3f} -> {final_score:.3f}")
            boosted_papers.append((paper, final_score))
        
        # Re-sort by boosted scores
        boosted_papers.sort(key=lambda x: x[1], reverse=True)
        
        return boosted_papers

    async def _process_papers_concurrent(
        self,
        ctx: RAGPipelineContext,
        papers_to_process: List[DBPaper]
    ) -> AsyncGenerator:
        """Process top papers concurrently (max 5)."""
        MAX_SUCCESSFUL = 5
        MAX_WORKERS = 2
        
        from app.core.dtos.paper import PaperEnrichedDTO
        paper_dtos = [PaperEnrichedDTO.model_validate(p) for p in papers_to_process]
        
        paper_ids = [str(p.paper_id) for p in paper_dtos]
        
        # Early exit if no papers to process
        if not paper_ids:
            ctx.processed_paper_ids = []
            yield ctx
            return
        
        processed_status = await self.processor.paper_service.batch_check_processed_papers(paper_ids)
        
        papers_needing_processing = [
            p for p in paper_dtos
            if not processed_status.get(str(p.paper_id), False)
        ]
        
        logger.info(f"Processing {len(papers_needing_processing)} papers ({len(paper_ids) - len(papers_needing_processing)} already processed)")
        
        if not papers_needing_processing:
            ctx.processed_paper_ids = [pid for pid, is_proc in processed_status.items() if is_proc]
            yield ctx
            return
        
        # Process papers concurrently (limit to MAX_SUCCESSFUL)
        results = await self.processor.process_papers_v2(
            papers=paper_dtos,
            filtered_papers=papers_needing_processing[:MAX_SUCCESSFUL],
            max_workers=MAX_WORKERS
        )
        
        ctx.processed_paper_ids = [pid for pid, success in results.items() if success][:MAX_SUCCESSFUL]
        
        yield ctx

    async def _hybrid_chunk_search(
        self,
        ctx: RAGPipelineContext,
        query: str,
        top_chunks: int,
        paper_ids: Optional[List[str]],
        intent: QueryIntent
    ) -> RAGPipelineContext:
        """Hybrid BM25 + semantic chunk search."""
        # Get weights based on intent
        bm25_weight, semantic_weight = self._get_chunk_search_weights(intent)
        
        # Use service layer for business logic
        chunks = await self.chunk_service.hybrid_search_chunks(
            query=query,
            limit=top_chunks,
            paper_ids=paper_ids,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
        )
        
        ctx.chunks = chunks
        logger.info(f"Hybrid chunk search returned {len(chunks)} chunks (scope: {'filtered' if paper_ids else 'full database'})")
        
        return ctx

    async def _rank_papers_with_hybrid_scores(
        self,
        query: str,
        papers: List[DBPaper],
        chunks: List[ChunkRetrieved],
        paper_hybrid_scores: Dict[str, float],
        intent: QueryIntent,
    ) -> List[RankedPaper]:
        """Rank papers incorporating hybrid search scores."""
        weights = self._get_ranking_weights(intent)
        ranked_papers = self.ranking_service.rank_papers(
            query=query,
            papers=papers,
            chunks=chunks,
            weights=weights,
        )
        
        hybrid_weight = self._get_hybrid_score_weight(intent)
        
        for ranked_paper in ranked_papers:
            paper_id = ranked_paper.paper_id
            hybrid_score = paper_hybrid_scores.get(paper_id, 0)
            original_score = ranked_paper.relevance_score
            blended_score = (original_score * (1 - hybrid_weight)) + (hybrid_score * hybrid_weight * 100)
            
            ranked_paper.relevance_score = blended_score
            ranked_paper.ranking_scores["hybrid_paper_score"] = hybrid_score
        
        ranked_papers.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return ranked_papers

    def _get_paper_search_weights(self, intent: QueryIntent) -> tuple[float, float, float]:
        """Get BM25/semantic weights for paper search based on intent."""
        if intent == QueryIntent.FOUNDATIONAL:
            return 0.6, 0.4, 10.0  # bm25_weight, semantic_weight, s2_boost
        elif intent == QueryIntent.COMPARISON:
            return 0.4, 0.6, 8.0
        elif intent == QueryIntent.AUTHOR_PAPERS:
            return 0.4, 0.6, 5.0
        else:  
            return 0.3, 0.7, 5.0

    def _get_chunk_search_weights(self, intent: QueryIntent) -> tuple[float, float]:
        """Get BM25/semantic weights for chunk search based on intent."""
        if intent == QueryIntent.FOUNDATIONAL:
            return 0.6, 0.4
        else:
            return 0.4, 0.6

    def _get_hybrid_score_weight(self, intent: QueryIntent) -> float:
        """Get weight for hybrid score in final ranking."""
        if intent == QueryIntent.FOUNDATIONAL:
            return 0.5
        elif intent == QueryIntent.COMPARISON:
            return 0.4
        else:
            return 0.3
        
    def _get_ranking_weights(self, intent: QueryIntent) -> Dict[str, float]:
        """Get weights for different relevance factors in final ranking."""
        if intent == QueryIntent.FOUNDATIONAL:
            return {"relevance": 0.4, "authority": 0.6}
        elif intent == QueryIntent.COMPARISON:
            return {"relevance": 0.6, "authority": 0.4}
        else:
            return {"relevance": 0.7, "authority": 0.3}

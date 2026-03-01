# app/rag/pipeline.py

import asyncio
import gc
from datetime import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Dict, Any, TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm import get_llm_service
from app.processor.paper_processor import PaperProcessor
from app.retriever.service import RetrievalService, RetrievalServiceType
from app.papers.repository import PaperRepository, LoadOptions
from app.chunks.repository import ChunkRepository
from app.chunks.schemas import ChunkRetrieved
from app.core.singletons import get_ranking_service
from app.models.papers import DBPaper
from app.processor.schemas import RankedPaper

from app.rag_pipeline.schemas import (
    PipelineResult,
    RAGPipelineContext,
    RAGPipelineEvent,
    RAGResult,
    RAGEventType,
)
from app.extensions.logger import create_logger

from app.rag_pipeline.utils import deduplicate_papers
from app.processor.services import transformer
from app.trust import compute_scores

if TYPE_CHECKING:
    from app.processor.paper_processor import PaperProcessor
    from app.processor.services.ranking import RankingService


logger = create_logger(__name__)


class Pipeline:
    """
    General multi-step orchestration service for various workflows.

    Workflows:
    - Paper RAG: Retrieve papers → process → extract chunks → answer questions
    - Author Enrichment: Fetch author papers → process → compute metrics
    - Future: Institution enrichment, citation graph building, bulk ingestion

    Responsibilities:
    - Orchestrate multi-step processes
    - Track progress with event streaming
    - Reuse processors (PaperProcessor, services)
    - Handle cross-cutting concerns (logging, caching, error handling)

    Uses dependency injection for better testability and shared instances.

    Note: Previously named RAGPipeline. Renamed to Pipeline to reflect general orchestration role.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        repository: Optional[PaperRepository] = None,
        retriever: Optional[RetrievalService] = None,
        processor: Optional[PaperProcessor] = None,
        llm_service=None,
        ranking_service: Optional["RankingService"] = None,
    ):
        """Initialize Pipeline with dependency injection.

        Args:
            db_session: Required database session
            repository: Optional shared repository (created if not provided)
            retriever: Optional retrieval service (created if not provided)
            processor: Optional paper processor (created if not provided)
            llm_service: Optional LLM service (singleton if not provided)
            ranking_service: Optional ranking service (singleton if not provided)
        """
        self.db_session = db_session
        self.repository = repository or PaperRepository(db_session)
        self.chunk_repository = ChunkRepository(db_session)
        self.retriever = retriever or RetrievalService(db=db_session)
        self.processor = processor or PaperProcessor(
            repository=self.repository,
            chunk_repository=self.chunk_repository,
            retrieval_service=self.retriever,
        )
        self.llm = llm_service or get_llm_service()
        self.ranking_service = ranking_service or get_ranking_service()  # Use singleton

    async def run_paper_rag_workflow(
        self,
        query: str,
        max_subtopics: int = 3,
        per_subtopic_limit: int = 50,
        top_chunks: int = 40,
        filters: Optional[Dict[str, Any]] = None,
        enable_reranking: bool = True,
        enable_paper_ranking: bool = True,
        relevance_threshold: float = 0.3,
        similarity_top_k: Optional[int] = 20,
        auto_optimize: bool = True,
    ):
        """
        Paper RAG workflow: Retrieve papers, filter by embedding similarity, process content.

        Args:
            query: User question
            max_subtopics: Max subtopics to generate
            per_subtopic_limit: Max papers per subtopic (default: 50)
            top_chunks: Max chunks to return
            filters: Optional filters (yearRange, category, openAccessOnly, excludePreprints, topJournalsOnly)
            enable_reranking: Whether to rerank chunks with cross-encoder
            enable_paper_ranking: Whether to apply comprehensive paper ranking
            relevance_threshold: Minimum semantic relevance score for papers (0-1)
            similarity_top_k: Number of top papers to keep after similarity filtering (None = use threshold)
            auto_optimize: Enable automatic pipeline optimization based on query intent (default: True)
        """
        ctx = RAGPipelineContext(query)
        
        ctx = await self._break_down_query(ctx, max_subtopics)
        breakdown_response = ctx.breakdown_response
        if auto_optimize and breakdown_response and breakdown_response.intent:
            logger.info(f"Query intent: {breakdown_response.intent.value} (confidence: {breakdown_response.intent_confidence or 'N/A'})")
            
            if breakdown_response.filters:
                filters = {**(filters or {}), **breakdown_response.filters}
            
            if breakdown_response.skip_ranking:
                enable_paper_ranking = False
        
        yield RAGPipelineEvent(
            type=RAGEventType.SEARCHING,
            data={"queries": ctx.search_queries, "original": query},
        )

        ctx = await self._retrieve_papers(
            ctx,
            per_subtopic_limit,
            filters,
        )

        should_filter = True
        if auto_optimize and breakdown_response and breakdown_response.skip_title_abstract_filter:
            logger.info("Skipping title/abstract filter based on query intent")
            should_filter = False
        
        if should_filter:
            yield RAGPipelineEvent(
                type=RAGEventType.PROCESSING,
                data={"message": "Generating embeddings and filtering papers by relevance", "total_papers": len(ctx.papers)},
            )
            
            ctx = await self._embed_and_filter_papers(ctx, query, similarity_top_k, relevance_threshold)
            
            if not ctx.papers:
                logger.warning("No papers passed similarity filtering")
                yield RAGPipelineEvent(type=RAGEventType.RESULT, data=RAGResult(papers=[], chunks=[]))
                return

        
        async for event in self._process_papers(ctx):
            if isinstance(event, RAGPipelineEvent):
                yield event
            else:
                ctx = event  # Update context

        # Check if any papers were successfully processed
        if not ctx.processed_paper_ids:
            logger.warning("No papers were successfully processed (all failed PDF/content retrieval)")
            if not ctx.papers:
                yield RAGPipelineEvent(
                    type=RAGEventType.RESULT,
                    data=RAGResult(papers=[], chunks=[])
                )
                return
            else:
                logger.info(f"Continuing with {len(ctx.papers)} papers using abstracts only (no chunks available)")

        if ctx.processed_paper_ids:
            logger.info(f"Successfully processed {len(ctx.processed_paper_ids)} papers with full content, {len(ctx.papers) - len(ctx.processed_paper_ids)} papers with abstracts only")
            self._write_log(ctx)
        
        # Only retrieve chunks if we have processed papers
        if ctx.processed_paper_ids:
            try:
                ctx = await self._retrieve_chunks(ctx, top_chunks)
                logger.info(f"Retrieved {len(ctx.chunks)} chunks from {len(ctx.processed_paper_ids)} processed papers")
            except Exception as e:
                logger.error(f"Error retrieving chunks: {e}", exc_info=True)
                ctx.chunks = []
        else:
            logger.info("No chunks available, using paper abstracts only for context")
            ctx.chunks = []

        if enable_reranking and ctx.chunks:
            try:
                ctx.chunks = self.ranking_service.rerank_chunks(query, ctx.chunks)
                logger.info(f"Reranked {len(ctx.chunks)} chunks")
            except Exception as e:
                logger.error(f"Error reranking chunks: {e}", exc_info=True)

        if enable_paper_ranking and ctx.papers:
            yield RAGPipelineEvent(
                type=RAGEventType.RANKING,
                data={"total_papers": len(ctx.papers), "total_chunks": len(ctx.chunks)},
            )

            try:
                enriched_papers, total = await self.repository.get_papers(
                    paper_ids=[p.paper_id for p in ctx.papers],
                    load_options=LoadOptions(authors=True, journal=True, institutions=True),
                )

                logger.info(
                    f"Loaded {len(enriched_papers)} enriched papers from DB for ranking"
                )
                await self._compute_trust_scores_lazy(enriched_papers)
                ranked_papers = self.ranking_service.rank_papers(
                    query=query,
                    papers=enriched_papers,
                    chunks=ctx.chunks,
                    enable_diversity=True,
                    relevance_threshold=relevance_threshold,
                )
                logger.info(f"Ranked to {len(ranked_papers)} papers")
                ctx.result_papers = ranked_papers
            except Exception as e:
                logger.error(f"Error during paper ranking: {e}", exc_info=True)
                # Fallback to unranked papers
                ctx.result_papers = []
        else:
            try:
                if should_filter:
                    logger.info(f"Skipping paper ranking, loading {len(ctx.filtered_papers)} filtered papers with default scores")
                    ctx.papers = ctx.filtered_papers
                enriched_papers, total = await self.repository.get_papers(
                    paper_ids=[p.paper_id for p in ctx.papers],
                    load_options=LoadOptions(authors=True, journal=True, institutions=True),
                )
                
                # Create RankedPaper objects with default/neutral scores
                ctx.result_papers = [
                    RankedPaper(
                        id=paper.id,
                        paper_id=paper.paper_id,
                        paper=paper,
                        relevance_score=1.0,  # Default neutral score
                        ranking_scores={
                            "citation_quality": 0.0,
                            "venue_prestige": 0.0,
                            "author_reputation": 0.0,
                            "recency": 0.0,
                            "institution_trust": 0.0,
                        }
                    )
                    for paper in enriched_papers
                ]
                logger.info(f"Created {len(ctx.result_papers)} papers with default scores")
            except Exception as e:
                logger.error(f"Error loading papers for result: {e}", exc_info=True)
                ctx.result_papers = []

        # Final validation before returning results
        if not ctx.result_papers:
            logger.warning("No papers available for final result (all processing/ranking failed)")
        
        if not ctx.chunks:
            logger.warning(f"No chunks available (processed {len(ctx.processed_paper_ids)} papers but chunks may be empty)")

        yield RAGPipelineEvent(
            type=RAGEventType.RESULT, data=RAGResult(papers=ctx.result_papers, chunks=ctx.chunks)
        )

    async def run(self, *args, **kwargs):
        """Backwards compatibility: redirects to run_paper_rag_workflow()"""
        async for event in self.run_paper_rag_workflow(*args, **kwargs):
            yield event

    async def run_author_enrichment_workflow(
        self,
        author_id: str,
        oa_author_id: Optional[str] = None,
        limit: int = 500,
    ) -> PipelineResult:
        """
        Author enrichment workflow: Fetch author papers, process them, compute career metrics.

        This workflow orchestrates the complete author enrichment process:
        1. Fetch all papers for author from APIs (S2/OpenAlex)
        2. Convert to DTOs and ensure paper records in database
        3. Compute career metrics (first pub year, FWCI, positions, etc.) via AuthorService
        4. Update last_paper_indexed_at timestamp
        5. Return enrichment result

        Args:
            author_id: Unique author identifier (S2 ID or OpenAlex ID)
            limit: Maximum papers to fetch (default: 500)

        Returns:
            PipelineResult with papers and metadata
        """
        logger.info(f"Starting author enrichment workflow for {author_id}")
        papers = await self.retriever.get_author_papers(author_id=author_id)
        author = await self.retriever.get_author(oa_author_id) if oa_author_id else None

        if limit and len(papers) > limit:
            papers = papers[:limit]
        logger.info(f"Retrieved {len(papers)} papers from API")

        if not papers:
            logger.warning(f"No papers found for author {author_id}")
            return PipelineResult(papers=[], author=author)

        processed_count = 0
        for paper in papers:
            try:
                await self.processor.ensure_paper_record(paper)
                processed_count += 1
            except Exception as e:
                logger.error(
                    f"Error ensuring paper {paper.paper_id}: {e}", exc_info=True
                )

        logger.info(f"Processed {processed_count}/{len(papers)} papers successfully")

        from app.authors.service import AuthorService

        author_service = AuthorService(self.db_session)

        logger.info(f"Computing career metrics for author {author_id}")
        await author_service.compute_career_metrics(author_id)

        from app.authors.repository import AuthorRepository

        author_repository = AuthorRepository(self.db_session)
        i10_index = (
            author.summary_stats.get("i10_index")
            if author and author.summary_stats
            else None
        )
        
        await author_repository.update_author(
            author_id, {"i10_index": i10_index, "last_paper_indexed_at": datetime.now()}
        )

        try:
            import asyncio
            asyncio.create_task(author_repository.compute_author_relationships(author_id))
            logger.info(f"Queued author relationships computation for {author_id}")
        except Exception as e:
            logger.warning(f"Failed to queue author relationships computation: {e}")

        logger.info(f"Author enrichment workflow completed for {author_id}")

        return PipelineResult(papers=papers, author=author)

    async def _break_down_query(
        self, ctx: RAGPipelineContext, max_subtopics: int = 3
    ) -> RAGPipelineContext:
        """
        Use LLM to break down user query into subtopics/search queries.
        Now also includes intent classification for pipeline optimization.

        Args:
            ctx: Pipeline context
            max_subtopics: Max subtopics to generate

        Returns:
            Updated context with search queries and breakdown response
        """
        breakdown = await self.llm.decompose_user_query(
            user_question=ctx.query, max_subtopics=max_subtopics
        )
        ctx.search_queries = breakdown.search_queries[:max_subtopics]
        ctx.breakdown_response = breakdown  # Store full response for intent access
        logger.info(f"Generated search queries: {ctx.search_queries}")
        if breakdown.intent:
            logger.info(f"Detected intent: {breakdown.intent.value}")
        return ctx

    async def _retrieve_papers(
        self,
        ctx: RAGPipelineContext,
        per_subtopic_limit,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RAGPipelineContext:
        """
        Retrieve papers for given search queries.

        Args:
            ctx: RAGPipelineContext containing search queries
            per_subtopic_limit: Max papers per subtopic
            filters: Optional filters
        Returns:
            Updated RAGPipelineContext with retrieved papers
        """
        all_papers = []

        for idx, search_query in enumerate(ctx.search_queries, 1):  
            papers, metadata = await self.retriever.hybrid_search(
                query=search_query,
                semantic_limit=per_subtopic_limit,
                filters=filters,
            )
            all_papers.extend(papers)
            logger.info(metadata)
            await asyncio.sleep(1)

        papers = deduplicate_papers(all_papers)
        logger.info(f"Total unique papers: {len(papers)}")

        ctx.papers = papers  # type: ignore
        return ctx

    async def _process_papers(
        self,
        ctx: RAGPipelineContext,
    ) -> AsyncGenerator[RAGPipelineEvent | RAGPipelineContext, None]:
        """
        Process papers sequentially until we get 5 successful papers with content.
        Processes in small batches, stops when target is reached or papers exhausted.

        Args:
            ctx: RAGPipelineContext containing papers
        Returns:
            Updated RAGPipelineContext with processed paper IDs
        """
        MAX_SUCCESSFUL_PAPERS = 5
        BATCH_SIZE = 3  # Process 3 papers at a time for efficiency
        
        successful_paper_ids = []
        attempted_count = 0
        batch_start_idx = 0
        
        logger.info(f"Starting sequential processing - target: {MAX_SUCCESSFUL_PAPERS} successful papers")
        
        # Process papers in batches until we reach target or run out of papers
        while len(successful_paper_ids) < MAX_SUCCESSFUL_PAPERS and batch_start_idx < len(ctx.papers):
            batch_end_idx = min(batch_start_idx + BATCH_SIZE, len(ctx.papers))
            batch_papers = ctx.papers[batch_start_idx:batch_end_idx]
            
            logger.info(f"Processing batch {batch_start_idx//BATCH_SIZE + 1}: papers {batch_start_idx+1}-{batch_end_idx} (need {MAX_SUCCESSFUL_PAPERS - len(successful_paper_ids)} more successes)")
            
            # Process this batch concurrently
            processed_results = await self.processor.process_papers_concurrent(
                batch_papers, ctx.filtered_papers, max_workers=2
            )
            
            # Check results in order and collect successful papers
            for paper in batch_papers:
                paper_id_str = str(paper.paper_id)
                success = processed_results.get(paper_id_str, False)
                attempted_count += 1
                
                yield RAGPipelineEvent(
                    type="processing",
                    data={
                        "paper_id": paper_id_str,
                        "success": success,
                        "progress": len(successful_paper_ids),
                        "total": MAX_SUCCESSFUL_PAPERS,
                        "attempted": attempted_count,
                        "message": f"Processed paper {attempted_count}/{len(ctx.papers)} - {len(successful_paper_ids)}/{MAX_SUCCESSFUL_PAPERS} successful",
                    },
                )
                
                # Add to successful list if processed and we haven't reached limit
                if success and len(successful_paper_ids) < MAX_SUCCESSFUL_PAPERS:
                    successful_paper_ids.append(paper_id_str)
                    logger.info(f"Paper {paper_id_str} successfully processed ({len(successful_paper_ids)}/{MAX_SUCCESSFUL_PAPERS})")
                
                # Stop if we've reached our target
                if len(successful_paper_ids) >= MAX_SUCCESSFUL_PAPERS:
                    logger.info(f"Reached target of {MAX_SUCCESSFUL_PAPERS} successful papers, stopping processing")
                    break
            
            batch_start_idx = batch_end_idx
        
        logger.info(
            f"Processing complete: {len(successful_paper_ids)} papers with full content, "
            f"{len(ctx.papers) - len(successful_paper_ids)} papers with abstracts only "
            f"(attempted {attempted_count}/{len(ctx.papers)} papers)"
        )
        
        ctx.processed_paper_ids = successful_paper_ids
        yield ctx

    async def _retrieve_chunks(
        self,
        ctx: RAGPipelineContext,
        top_chunks: int = 40,
    ) -> RAGPipelineContext:
        """
        Retrieve relevant chunks for given paper IDs.

        Args:
            ctx: RAGPipelineContext containing paper IDs
            top_chunks: Max chunks to return
        Returns:
            Updated RAGPipelineContext with retrieved chunks
        """
        if not ctx.processed_paper_ids:
            logger.warning("No processed paper IDs available for chunk retrieval")
            ctx.chunks = []
            return ctx

        all_chunks = []
        queries_for_chunks = [ctx.query] + ctx.search_queries
        
        for chunk_query in queries_for_chunks:
            try:
                query_chunks = await self.retriever.get_relevant_chunks(
                    query=chunk_query,
                    paper_ids=ctx.processed_paper_ids,
                    limit=top_chunks
                    // len(queries_for_chunks),  # Distribute limit across queries
                )
                all_chunks.extend(query_chunks)
            except Exception as e:
                logger.error(f"Error retrieving chunks for query '{chunk_query[:50]}...': {e}")
                continue

        ctx.chunks = self._dedup_chunks(all_chunks)
        
        if not ctx.chunks:
            logger.warning(f"No chunks retrieved from {len(ctx.processed_paper_ids)} processed papers")
        
        return ctx

    async def _rank_papers(self, ctx: RAGPipelineContext) -> RAGPipelineContext:
        """Placeholder for paper ranking logic (currently integrated in main workflow)"""
        return ctx

    async def _embed_and_filter_papers(
        self,
        ctx: RAGPipelineContext,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> RAGPipelineContext:
        """
        Generate embeddings for papers and filter by similarity to query.
        Papers with existing embeddings (from DB cache) are reused.

        Args:
            ctx: RAGPipelineContext containing papers
            query: User's clarified question
            top_k: Number of top papers to keep (default: top papers with PDFs)
            min_score: Minimum similarity score threshold

        Returns:
            Updated context with filtered papers and embeddings cached to DB
        """
        if not ctx.papers:
            return ctx
        
        logger.info(f"Generating embeddings for {len(ctx.papers)} papers")
        ctx.papers = await self.processor.generate_paper_embeddings(ctx.papers)
        
        paper_embeddings = {
            str(p.paper_id): p.embedding
            for p in ctx.papers
            if p.embedding is not None
        }
        
        if paper_embeddings:
            await self.repository.bulk_update_paper_embeddings(paper_embeddings)
            logger.info(f"Cached {len(paper_embeddings)} embeddings to database")
        
        logger.info(f"Filtering papers by similarity to: {query}")
        filtered_papers = await self.processor.filter_papers_by_similarity(
            papers=ctx.papers,
            query=query,
            top_k=top_k,
            min_score=min_score,
            prefer_open_access=True,
        )
        
        logger.info(
            f"Filtered from {len(ctx.papers)} to {len(filtered_papers)} papers "
            f"based on semantic similarity"
        )
        
        ctx.filtered_papers = filtered_papers
        return ctx

    async def _compute_trust_scores_lazy(self, papers: List[DBPaper]) -> None:
        """
        Compute trust scores for papers missing them using lazy evaluation.
        Computes from already-loaded relationships and persists to database.

        Args:
            papers: List of DBPaper objects with author/institution relationships loaded
        """
        papers_without_trust = [
            p
            for p in papers
            if p.author_trust_score is None or p.institutional_trust_score is None
        ]

        if not papers_without_trust:
            logger.debug("All papers have trust scores, skipping lazy computation")
            return

        logger.info(
            f"{len(papers_without_trust)} papers missing trust scores - computing with lazy scoring"
        )

        for paper in papers_without_trust:
            author_papers = getattr(paper, "paper_authors", [])

            if author_papers:
                author_papers_data = [
                    {
                        "author_reputation_score": (
                            ap.author.reputation_score if ap.author else None
                        ),
                        "institution_reputation_score": (
                            ap.institution.reputation_score if ap.institution else None
                        ),
                        "institution_id": ap.institution_id,
                        "country_code": (
                            ap.institution.country_code if ap.institution else None
                        ),
                    }
                    for ap in author_papers
                ]

                scores = compute_scores.compute_paper_trust_scores(author_papers_data)

                # Update paper fields in-memory
                paper.author_trust_score = scores["author_trust_score"]
                paper.institutional_trust_score = scores["institutional_trust_score"]
                paper.network_diversity_score = scores["network_diversity_score"]
                paper.institutions_distinct_count = scores[
                    "institutions_distinct_count"
                ]
                paper.countries_distinct_count = scores["countries_distinct_count"]

                await self.repository.update_paper(
                    paper_id=str(paper.paper_id),
                    update_data={
                        "author_trust_score": scores["author_trust_score"],
                        "institutional_trust_score": scores[
                            "institutional_trust_score"
                        ],
                        "network_diversity_score": scores["network_diversity_score"],
                        "institutions_distinct_count": scores[
                            "institutions_distinct_count"
                        ],
                        "countries_distinct_count": scores["countries_distinct_count"],
                    },
                )
            else:
                logger.warning(
                    f"Paper {paper.paper_id} has no author relationships loaded - skipping lazy scoring"
                )

        logger.info(
            f"Computed and persisted trust scores for {len(papers_without_trust)} papers"
        )

    def _write_log(self, ctx: RAGPipelineContext):
        """Writing processed datas for debug

        Args:
            ctx (RAGPipelineContext): Context containing the data to log
        """
        base_dir = Path(__file__).parent
        logs_dir = base_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().isoformat().replace(":", "-")
        filename = logs_dir / f"processed_papers-{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(ctx.__dict__, f, ensure_ascii=False, indent=4, default=str)

    def _dedup_chunks(self, chunks: List["ChunkRetrieved"]) -> List["ChunkRetrieved"]:
        """Deduplicate chunks by chunk_id

        Args:
            chunks: List of ChunkRetrieved objects to deduplicate

        Returns:
            Deduplicated list of ChunkRetrieved objects
        """
        seen_chunk_ids = set()
        deduped_chunks = []
        for chunk in chunks:
            if chunk.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk.chunk_id)
                deduped_chunks.append(chunk)
        return deduped_chunks

"""
Bulk preprocessing service using Semantic Scholar bulk search API.

Clean architecture with retriever service integration for:
- Bulk search API calls
- Batch paper details fetching
- OpenAlex enrichment
- Database caching
- Progress tracking
"""

from app.retriever.schemas.openalex import OAAuthorResponse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import traceback
from sqlalchemy.ext.asyncio import AsyncSession

from app.extensions.logger import create_logger
from app.domain.papers import PaperRepository, PaperService
from app.domain.authors import AuthorService
from app.domain.papers.journal_service import JournalService
from app.domain.papers.conference_service import ConferenceService
from app.domain.papers.linking_service import PaperLinkingService
from app.retriever.service import RetrievalService
from app.processor.paper_processor import PaperProcessor
from app.processor.preprocessing_repository import PreprocessingRepository
from app.models.preprocessing_state import DBPreprocessingState
from app.domain.chunks import ChunkRepository

logger = create_logger(__name__)


class PreprocessingService:
    """
    Service for bulk preprocessing using Semantic Scholar bulk search.

    Architecture:
    - Uses RetrievalService for all API interactions (search, fetch, enrich)
    - PaperProcessor handles RAG pipeline (extract, chunk, embed)
    - Journal/Conference linking after paper creation
    - Citation/Reference graph building from paper metadata
    - Database caching to avoid duplicate processing
    - State management for resume/pause functionality
    """

    def __init__(
        self,
        db_session: AsyncSession,
        paper_repository: Optional[PaperRepository] = None,
        preprocessing_repo: Optional[PreprocessingRepository] = None,
        retriever: Optional[RetrievalService] = None,
        paper_service: Optional[PaperService] = None,
        processor: Optional[PaperProcessor] = None,
        journal_service: Optional[JournalService] = None,
        conference_service: Optional[ConferenceService] = None,
        linking_service: Optional[PaperLinkingService] = None,
    ):
        """
        Initialize preprocessing service with dependency injection.
        
        Args:
            db_session: Database session
            paper_repository: Repository for paper database operations (optional)
            preprocessing_repo: Repository for preprocessing state (optional)
            retriever: Service for API interactions (optional)
            paper_service: Service for paper business logic (optional)
            processor: Service for RAG pipeline (optional)
            journal_service: Service for journal linking (optional)
            conference_service: Service for conference linking (optional)
            linking_service: Service for citation/reference linking (optional)
        """
        self.db_session = db_session
        
        # Accept via DI or create as fallback
        self.repository = paper_repository or PaperRepository(db_session)
        self.preprocessing_repo = preprocessing_repo or PreprocessingRepository(db_session)
        self.retriever = retriever or RetrievalService(db=db_session)
        
        # PaperService with fallback
        if paper_service:
            self.paper_service = paper_service
        else:
            self.paper_service = PaperService(self.repository, self.retriever)
        
        # PaperProcessor with fallback
        if processor:
            self.processor = processor
        else:
            chunk_repository = ChunkRepository(db_session)
            self.processor = PaperProcessor(
                repository=self.repository,
                chunk_repository=chunk_repository,
                retrieval_service=self.retriever,
            )
        
        # Enrichment services with fallbacks
        self.journal_service = journal_service or JournalService(db_session)
        self.conference_service = conference_service or ConferenceService(db_session)
        self.linking_service = linking_service or PaperLinkingService(
            db=db_session, paper_repository=self.repository
        )
        self.author_service = AuthorService(db_session)

    # ==================== Main Entry Point ====================

    async def process_bulk_search(
        self,
        job_id: str,
        search_query: str,
        target_count: int,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        fields_of_study: Optional[List[str]] = None,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Process papers from Semantic Scholar bulk search API.

        Sequential Workflow:
        1. Fetch and index all papers (creates DB schemas, links journals/conferences, extracts refs)
        2. Link citations and references across the indexed batch
        3. Generate title/abstract embeddings for newly indexed metadata
        4. Resolve content (PDF -> Chunk -> Embed) for pending open access papers

        Args:
            job_id: Unique job identifier for tracking
            search_query: Search query string
            target_count: Number of papers to process
            year_min: Minimum publication year
            year_max: Maximum publication year
            fields_of_study: List of fields to filter by
            resume: Whether to resume from previous state

        Returns:
            Statistics about the preprocessing job
        """
        logger.info(f"[Preprocessing] Starting job {job_id}")
        logger.info(f"[Preprocessing] Query: '{search_query}', Target: {target_count}")

        # Initialize state
        state = await self._initialize_job_state(job_id, target_count, resume)

        if state.is_completed:
            logger.info(f"[Preprocessing] Job {job_id} already completed")
            return self._state_to_stats(state)

        if state.is_running:
            logger.warning(f"[Preprocessing] Job {job_id} is already running")
            return self._state_to_stats(state)

        # Mark as running
        await self._update_state(
            state, is_running=True, message="Phase 1: Indexing metadata..."
        )

        try:
            continuation_token = state.continuation_token if resume else None

            # Phase 1: Indexing Loop
            while state.processed_count < target_count:
                # Check for pause
                state = await self._refresh_state(job_id)
                if state.is_paused:
                    logger.info(f"[Preprocessing] Job {job_id} paused by user")
                    await self._update_state(
                        state, is_running=False, message="Paused by user"
                    )
                    return self._state_to_stats(state)

                # Fetch batch from bulk search
                batch_size = min(100, target_count - state.processed_count)
                bulk_result = await self._fetch_bulk_search_batch(
                    query=search_query,
                    limit=batch_size,
                    token=continuation_token,
                    year_min=year_min,
                    year_max=year_max,
                    fields_of_study=fields_of_study,
                )

                if not bulk_result or not bulk_result.get("data"):
                    logger.info("[Preprocessing] No more results from bulk search")
                    break

                papers_data = bulk_result["data"]
                continuation_token = bulk_result.get("token")

                # Update continuation token
                await self.preprocessing_repo.set_state_continuation_token(
                    state=state,
                    continuation_token=continuation_token,
                )

                # Process batch: Index metadata and link journals/conferences
                await self._index_metadata_batch(papers_data, state, target_count)

                # Check if we should stop
                state = await self._refresh_state(job_id)
                if not continuation_token or state.processed_count >= target_count:
                    break

                # Update progress
                progress_msg = (
                    f"Indexed: {state.processed_count}/{target_count} | "
                    f"Skipped: {state.skipped_count} | "
                    f"Errors: {state.error_count}"
                )
                await self._update_state(state, message=progress_msg)
                
            state = await self._refresh_state(job_id)
            
            # Phase 2: Link citations/references from the extracted batch
            await self._update_state(state, message="Phase 2: Linking citations...")
            await self._link_citations_from_batch(state)

            # Phase 3: Embed title and abstract for DB papers missing them
            await self._update_state(state, message="Phase 3: Generating metadata embeddings...")
            await self._generate_missing_embeddings(state)

            # Phase 4: Resolve PDF content and embed chunks sequentially
            await self._update_state(state, message="Phase 4: Resolving content & chunking...")
            await self._process_pending_content(state)
            
            # Process any other unprocessed papers if necessary
            await self._process_unprocessed_papers(state)

            await self._complete_job(state)

            logger.info(
                f"[Preprocessing] Job {job_id} completed: {self._state_to_stats(state)}"
            )
            return self._state_to_stats(state)

        except Exception as e:
            logger.error(
                f"[Preprocessing] Fatal error in job {job_id}: {e}", exc_info=True
            )
            await self._log_error_to_file(
                job_id=job_id,
                stage="process_bulk_search",
                message=f"Fatal error in preprocessing job {job_id}",
                error=e,
                context={
                    "search_query": search_query,
                    "target_count": target_count,
                },
            )
            try:
                await self.db_session.rollback()
            except Exception as rollback_error:
                logger.error(
                    f"[Preprocessing] Rollback failed for job {job_id}: {rollback_error}",
                    exc_info=True,
                )
            state = await self._refresh_state(job_id)
            if state:
                await self._update_state(
                    state, is_running=False, message=f"Error: {str(e)}"
                )
            raise

    # ==================== Batch Processing ====================

    async def _index_metadata_batch(
        self,
        papers_data: List[Dict[str, Any]],
        state: DBPreprocessingState,
        target_count: int,
    ) -> None:
        """
        Index a batch of papers into the database without processing PDF content.

        Steps:
        1. Filter for open access papers only
        2. Extract paper IDs
        3. Batch fetch full details via RetrievalService
        4. Enrich with OpenAlex metadata
        5. Create database schemas, link authors/journals/conferences.
        """
        # Filter for open access papers
        oa_papers = self._filter_open_access_papers(papers_data)
        if not oa_papers:
            logger.info("[Preprocessing] No open access papers in batch")
            return

        # Extract paper IDs
        paper_ids = self._extract_paper_ids(oa_papers)
        if not paper_ids:
            logger.warning("[Preprocessing] No valid paper IDs in batch")
            return

        logger.info(f"[Preprocessing] Fetching details for {len(paper_ids)} papers...")

        idx = 0
        enriched_papers = []
        batch_size = 100
        if len(paper_ids) >= 100:
            logger.info(
                f"[Preprocessing] Large batch detected ({len(paper_ids)} papers), processing in sub-batches..."
            )
            for i in range(0, len(paper_ids), batch_size):
                sub_batch_ids = paper_ids[i : i + batch_size]
                logger.info(
                    f"[Preprocessing] Processing sub-batch {idx+1}: {len(sub_batch_ids)} papers"
                )
                sub_enriched = await self._fetch_and_enrich_papers(
                    sub_batch_ids,
                    job_id=state.job_id,
                )
                enriched_papers.extend(sub_enriched)
                idx += 1
        else:
            enriched_papers = await self._fetch_and_enrich_papers(
                paper_ids,
                job_id=state.job_id,
            )

        if not enriched_papers:
            logger.warning("[Preprocessing] No enriched papers returned")
            return

        logger.info(
            f"[Preprocessing] Indexing {len(enriched_papers)} enriched papers"
        )

        # Process each paper
        job_id = state.job_id
        for paper in enriched_papers:
            # Always refresh state from DB to avoid detached/expired object access
            state = await self._refresh_state(job_id)
            cached_processed_count = state.processed_count
            cached_job_id = state.job_id
            
            if cached_processed_count >= target_count:
                break

            # Check for pause
            state = await self._refresh_state(cached_job_id)
            if state.is_paused:
                logger.info(f"[Preprocessing] Job {cached_job_id} paused during batch")
                return

            await self._index_single_paper(paper, state, target_count)

    async def _fetch_and_enrich_papers(
        self,
        paper_ids: List[str],
        job_id: Optional[str] = None,
    ) -> List[Any]:
        """
        Fetch full paper details and enrich with OpenAlex metadata.

        Uses RetrievalService.get_multiple_papers which:
        1. Batch fetches from Semantic Scholar
        2. Enriches with OpenAlex metadata
        3. Returns normalized PaperEnrichedDTO objects
        """
        try:
            enriched_papers = await self.retriever.get_multiple_papers(paper_ids)
            logger.info(
                f"[Preprocessing] Enriched {len(enriched_papers)} papers with OpenAlex"
            )
            return enriched_papers
        except Exception as e:
            logger.error(f"[Preprocessing] Error fetching/enriching papers: {e}")
            await self._log_error_to_file(
                job_id=job_id,
                stage="fetch_and_enrich",
                message="Failed to fetch and enrich papers",
                error=e,
                context={"paper_ids": paper_ids},
            )
            return []

    async def _index_single_paper(
        self, paper: Any, state: DBPreprocessingState, target_count: int
    ) -> None:
        """
        Index a single schema paper: check cache, create DB paper, link journal/conference, extract refs.

        Handles:
        - Database cache lookup (skip if exists)
        - Paper creation via PaperService (creates authors and institutions naturally)
        - Journal/Conference linking
        - Emitting references data to batch batch
        - Note: Content RAG pipeline is NOT run here; only metadata is persisted.

        Returns citation data for later batch linking via state modification.
        """
        try:
            paper_id = str(paper.paper_id)

            # Check if already exists (skip if exists)
            if await self.preprocessing_repo.paper_exists(paper_id):
                state.skipped_count += 1
                state.current_index += 1
                logger.info(f"[Preprocessing] Paper {paper_id} already exists, skipping")
                return

            # Create paper in database. This creates authors and institutions right away.
            db_paper = await self.paper_service.ingest_paper_metadata(paper)
            if not db_paper:
                logger.warning(f"Failed to create paper {paper_id}")
                state.error_count += 1
                state.current_index += 1
                return

            logger.info(f"Created paper {paper_id}")

            # Link to journal if ISSN/Venue available
            if db_paper.issn or db_paper.issn_l or db_paper.venue:
                try:
                    journal = await self.journal_service.link_journal_to_paper(
                        paper=db_paper,
                        venue=db_paper.venue,
                        issn=(
                            db_paper.issn[0]
                            if db_paper.issn and len(db_paper.issn) > 0
                            else None
                        ),
                        issn_l=db_paper.issn_l,
                    )
                    if journal:
                        logger.info(
                            f"[Preprocessing] Linked paper {paper_id} to journal: "
                            f"{journal.title} (Q{journal.sjr_best_quartile})"
                        )
                except Exception as e:
                    logger.warning(
                        f"[Preprocessing] Failed to link journal for {paper_id}: {e}"
                    )
                    await self._log_error_to_file(
                        job_id=state.job_id,
                        stage="journal_linking",
                        message=f"Failed to link journal for paper {paper_id}",
                        error=e,
                        context={"paper_id": paper_id},
                    )

            # Link to conference if venue available
            if db_paper.venue:
                try:
                    conference = (
                        await self.conference_service.link_conference_to_paper(
                            paper=db_paper, venue=db_paper.venue
                        )
                    )
                    if conference:
                        logger.info(
                            f"[Preprocessing] Linked paper {paper_id} to conference: "
                            f"{conference.title} ({conference.acronym}, rank: {conference.rank})"
                        )
                except Exception as e:
                    logger.warning(
                        f"[Preprocessing] Failed to link conference for {paper_id}: {e}"
                    )
                    await self._log_error_to_file(
                        job_id=state.job_id,
                        stage="conference_linking",
                        message=f"Failed to link conference for paper {paper_id}",
                        error=e,
                        context={"paper_id": paper_id},
                    )

            # Extract references for later citation linking
            references_data = self._extract_references_from_paper(paper)
            if references_data:
                if not hasattr(state, "_citation_batch"):
                    state._citation_batch = []  # type: ignore 
                state._citation_batch.append((paper_id, references_data))  # type: ignore

            # We consider the paper successfully indexed since schema is created
            state.processed_count += 1
            logger.info(f"[Preprocessing] Successfully indexed schema for {paper_id}")
            state.current_index += 1

            # Cache values before session operations (state will be detached)
            processed_count = state.processed_count
            job_id = state.job_id

            # Commit and free memory
            await self._commit_and_clear_session()
            state = await self._refresh_state(job_id)

            # Log progress
            if processed_count % 5 == 0:
                logger.info(
                    f"[Preprocessing] Indexing Progress: {processed_count}/{target_count}"
                )

        except Exception as e:
            logger.error(f"[Preprocessing] Error indexing paper template: {e}")
            await self._log_error_to_file(
                job_id=getattr(state, "job_id", None),
                stage="index_single_paper",
                message="Error indexing paper",
                error=e,
                context={"paper_id": getattr(paper, "paper_id", None)},
            )
            try:
                await self.db_session.rollback()
            except Exception as rollback_error:
                logger.error(
                    f"[Preprocessing] Rollback failed while handling indexing error: {rollback_error}",
                    exc_info=True,
                )
            # Cache values before operations
            try:
                error_count = state.error_count + 1
                current_index = state.current_index + 1
                job_id = state.job_id
            except Exception:
                # If state is already detached, refresh it first
                logger.warning("State detached, refreshing before update")
                return
            
            # Update through fresh state
            await self.db_session.commit()
            self.db_session.expunge_all()
            state = await self._refresh_state(job_id)
            state.error_count = error_count
            state.current_index = current_index
            await self.db_session.commit()

    async def _process_pending_content(self, state: DBPreprocessingState) -> None:
        """
        Phase 4: Resolves PDF content and embeds chunks sequentially for pending papers.
        Retrieves all open access papers from DB with status pending/False.
        """
        try:
            logger.info("[Preprocessing] Phase 4: Checking for pending open-access papers...")
            from app.core.dtos.paper import PaperEnrichedDTO
            
            limit = 50
            total_resolved = 0
            
            while True:
                pending_papers = await self.preprocessing_repo.get_unprocessed_papers(limit=limit)
                if not pending_papers:
                    logger.info("[Preprocessing] No more pending open-access papers to process")
                    break
                    
                logger.info(f"[Preprocessing] Batch: processing content for {len(pending_papers)} pending papers...")
                
                paper_dtos = []
                for db_paper in pending_papers:
                    try:
                        paper_dtos.append((str(db_paper.paper_id), PaperEnrichedDTO.from_db_model(db_paper)))
                    except Exception as dto_err:
                        logger.error(f"[Preprocessing] Failed to convert DB paper to DTO, skipping: {dto_err}")
                
                for paper_id, paper_dto in paper_dtos:
                    logger.info(f"[Preprocessing] Extracting structure and chunking: {paper_id}")
                    
                    try:
                        # paper_dto is already converted above
                        
                        # Process individual paper sequentially: resolves PDF, chunk, embed
                        success = await self._run_rag_pipeline(paper_dto, paper_id)
                        
                        if success:
                            # DB status is updated to "completed" implicitly within the pipeline upon success
                            total_resolved += 1
                            logger.info(f"[Preprocessing] Processed content successfully: {paper_id}")
                        else:
                            # The RAG pipeline sets status to "failed" inside
                            logger.warning(f"[Preprocessing] Failed chunk extraction for {paper_id}, marked as failed.")
                            
                    except Exception as inner_e:
                        logger.error(f"[Preprocessing] Context failure on PDF extraction: {inner_e}")
                        await self.repository.update_paper_processing_status(paper_id, "failed")
                
                # Prevent runaway loops in case records fail to update processing flag
                await self.db_session.commit()
                # Clear session to prevent detached errors on next loop
                self.db_session.expunge_all()
                
            logger.info(f"[Preprocessing] Completed Phase 4. Processed content for {total_resolved} total pending papers")
        
        except Exception as e:
            logger.error(f"[Preprocessing] Content processing phase crashed: {e}", exc_info=True)
            await self._log_error_to_file(
                job_id=state.job_id if state else "manual_trigger",
                stage="process_pending_content",
                message="Error processing full text content phase",
                error=e
            )

    async def run_content_processing(self, limit: int = 50) -> Dict[str, int]:
        """
        Manually trigger Phase 4: resolve PDF content, chunk, and embed for all pending papers.

        This is the standalone, router-triggerable version of the Phase 4 step that runs
        automatically inside `process_bulk_search`. It does not require an active job.

        Returns:
            Stats dict with `processed`, `failed`, `total` counts.
        """
        from app.core.dtos.paper import PaperEnrichedDTO

        stats = {"total": 0, "processed": 0, "failed": 0}

        while True:
            pending_papers = await self.preprocessing_repo.get_unprocessed_papers(limit=limit)
            if not pending_papers:
                break

            stats["total"] += len(pending_papers)

            # Build minimal DTOs from scalar columns only — no relationship access.
            # PaperEnrichedDTO.from_db_model reads `.authors` which triggers a lazy
            # load outside the greenlet. We only need identity + content-resolution
            # fields for the RAG pipeline: paper_id, external_ids, pdf urls.
            paper_dtos: List[tuple[str, PaperEnrichedDTO]] = []
            for p in pending_papers:
                try:
                    dto = PaperEnrichedDTO(
                        paper_id=str(p.paper_id),
                        title=p.title or "",
                        abstract=p.abstract,
                        is_open_access=bool(p.is_open_access),
                        open_access_pdf=p.open_access_pdf,
                        pdf_url=p.pdf_url,
                        external_ids=p.external_ids,
                        source=p.source or "SemanticScholar",
                        is_processed=bool(p.is_processed),
                        processing_status=p.processing_status or "pending",
                        authors=[],  # not needed for content processing
                    )
                    paper_dtos.append((str(p.paper_id), dto))
                except Exception as dto_err:
                    logger.error(f"[Preprocessing] DTO build failed for {p.paper_id}, skipping: {dto_err}")
                    stats["failed"] += 1

            # Session is no longer needed after this point — expunge before async work
            self.db_session.expunge_all()

            for paper_id, paper_dto in paper_dtos:
                try:
                    success = await self._run_rag_pipeline(paper_dto, paper_id)
                    if success:
                        stats["processed"] += 1
                    else:
                        stats["failed"] += 1
                except Exception as e:
                    logger.error(f"[Preprocessing] Content processing failed for {paper_id}: {e}")
                    stats["failed"] += 1

        logger.info(
            "[Preprocessing] Manual content processing complete: "
            f"total={stats['total']}, processed={stats['processed']}, failed={stats['failed']}"
        )
        return stats



    async def _run_rag_pipeline(self, paper: Any, paper_id: str) -> bool:
        """
        Run RAG pipeline for paper (extract, chunk, embed).

        Returns:
            True if successful, False otherwise
        """
        try:
            success = await self.processor.process_single_paper(paper)
            return success
        except Exception as e:
            logger.error(
                f"[Preprocessing] RAG pipeline error for {paper_id}: {e}", exc_info=True
            )
            return False

    async def _log_error_to_file(
        self,
        job_id: Optional[str],
        stage: str,
        message: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist structured preprocessing errors to a per-job JSONL file."""
        try:
            output_dir = Path("preprocessing_logs") / "errors"
            output_dir.mkdir(parents=True, exist_ok=True)

            safe_job_id = (job_id or "unknown_job").replace("/", "_").replace("\\", "_")
            output_file = output_dir / f"preprocessing_errors_{safe_job_id}.jsonl"

            payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "job_id": job_id,
                "stage": stage,
                "message": message,
                "error_type": type(error).__name__,
                "error": str(error),
                "context": context or {},
                "traceback": traceback.format_exc(),
            }

            with output_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as file_error:
            logger.error(
                f"[Preprocessing] Failed to persist error log to file: {file_error}"
            )

    # ==================== API Calls via RetrievalService ====================

    async def _fetch_bulk_search_batch(
        self,
        query: str,
        limit: int = 100,
        token: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        fields_of_study: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch papers using Semantic Scholar bulk search API.

        Note: Currently uses direct API call. Could be moved to RetrievalService
        for better abstraction.

        Returns:
            {
                "total": int,
                "token": str (optional),
                "data": List[Dict]
            }
        """
        from app.retriever.provider import SemanticScholarProvider
        from app.retriever.service import RetrievalServiceType

        semantic_provider = self.retriever.get_provider_as(
            RetrievalServiceType.SEMANTIC, SemanticScholarProvider
        )

        try:
            # Use get_bulk_paper method from semantic provider
            results = await semantic_provider.get_bulk_paper(
                query, token=token, fields_of_study=fields_of_study
            )

            # Package in expected format
            return {
                "total": len(results),
                "token": token,  # Note: Need to extract continuation token properly
                "data": results,
            }
        except Exception as e:
            logger.error(f"[Preprocessing] Error fetching bulk search: {e}")
            await self._log_error_to_file(
                job_id=None,
                stage="fetch_bulk_search_batch",
                message="Failed to fetch bulk search batch",
                error=e,
                context={
                    "query": query,
                    "limit": limit,
                    "year_min": year_min,
                    "year_max": year_max,
                    "fields_of_study": fields_of_study,
                },
            )
            return None

    # ==================== Helper Functions ====================

    async def _generate_missing_embeddings(self, state: Optional[DBPreprocessingState] = None) -> None:
        """
        Generate title + abstract embeddings for papers that don't have them.

        This ensures all papers in the database have embeddings for semantic search.
        Particularly useful for papers that failed during initial processing.

        Args:
            state: Optional preprocessing state for tracking
        """
        try:
            logger.info("[Preprocessing] Checking for papers missing embeddings...")

            # Get papers without embeddings via repository
            papers = await self.preprocessing_repo.get_papers_missing_embeddings(limit=1000)

            if not papers:
                logger.info("[Preprocessing] No papers missing embeddings")
                return

            logger.info(
                f"[Preprocessing] Generating embeddings for {len(papers)} papers"
            )

            # Generate embeddings directly without DTO conversion to avoid lazy-loading 'authors'
            texts = [
                f"{p.title}\n\n{p.abstract or ''}"
                for p in papers
            ]
            
            embeddings = await self.processor.embedding_service.create_embeddings_batch(
                texts, batch_size=20, task="search_document"
            )

            # Update database with new embeddings
            paper_embeddings = {
                str(p.paper_id): emb
                for p, emb in zip(papers, embeddings)
                if emb is not None
            }

            if paper_embeddings:
                await self.repository.bulk_update_paper_embeddings(paper_embeddings)
                logger.info(
                    f"[Preprocessing] Successfully generated embeddings for "
                    f"{len(paper_embeddings)} papers"
                )

        except Exception as e:
            logger.error(
                f"[Preprocessing] Error generating embeddings: {e}", exc_info=True
            )
            await self._log_error_to_file(
                job_id=getattr(state, "job_id", None) if state else None,
                stage="generate_missing_embeddings",
                message="Failed generating missing embeddings",
                error=e,
            )

    async def compute_all_author_metrics(
        self,
        only_unprocessed: bool = False,
        conflict_threshold_percent: float = 50.0,
        batch_size: int = 100,
    ) -> Dict[str, int]:
        """
        Compute trust metrics for all authors and process conflict flags.

        Metrics computed for each author:
        - `reputation_score`
        - `retracted_papers_count`
        - `g_index`
        - `has_retracted_papers`
        - `is_conflict`

        Processing logic:
        - If author `is_processed` is False, fetch Semantic Scholar author details.
        - If `openalex_id` exists, fetch OpenAlex author details.
        - Compare citation counts and flag `is_conflict` when diff >= 50%.
        """
        stats = {
            "total_authors": 0,
            "processed_authors": 0,
            "conflicts": 0,
            "errors": 0,
        }

        offset = 0
        while True:
            authors = await self.preprocessing_repo.list_authors_for_metrics(
                limit=batch_size,
                offset=offset,
                only_unprocessed=only_unprocessed,
            )
            if not authors:
                break

            stats["total_authors"] += len(authors)

            author_oa_ids: list[str] = []
            for author in authors:
                raw = author.external_ids or {}
                oa_raw = raw.get("OpenAlex") or author.openalex_id
                if oa_raw:
                    author_oa_ids.append(self._normalize_openalex_id(str(oa_raw)))
                    
            oa_batch: List[OAAuthorResponse] = []
            if author_oa_ids:
                oa_batch = await self._fetch_multiple_authors(author_oa_ids)

            logger.debug(f"[Preprocessing] Fetched {len(oa_batch)} OA authors")
            logger.debug(oa_batch)
            oa_map: Dict[str, OAAuthorResponse] = {
                self._normalize_openalex_id(oa.id): oa for oa in oa_batch
            }

            # Per-author processing
            for author in authors:
                try:
                    # Resolve this author's OA entry from the pre-fetched map
                    raw = author.external_ids or {}
                    oa_raw = raw.get("OpenAlex") or author.openalex_id
                    oa_data = oa_map.get(self._normalize_openalex_id(str(oa_raw))) if oa_raw else None

                    # Use DB-cached S2 fields — no external API call needed here
                    has_conflict = await self._compute_single_author_metrics(
                        author=author,
                        oa_data=oa_data,
                        conflict_threshold_percent=conflict_threshold_percent,
                    )
                    stats["processed_authors"] += 1
                    if has_conflict:
                        stats["conflicts"] += 1

                except Exception as e:
                    stats["errors"] += 1
                    logger.error(
                        "[Preprocessing] Failed author metrics for %s: %s",
                        author.author_id, e, exc_info=True,
                    )

            offset += len(authors)


        logger.info(
            "[Preprocessing] Author trust metrics completed: "
            f"total={stats['total_authors']}, processed={stats['processed_authors']}, "
            f"conflicts={stats['conflicts']}, errors={stats['errors']}"
        )
        return stats

    async def _compute_single_author_metrics(
        self,
        author,
        oa_data: Optional["OAAuthorResponse"],
        conflict_threshold_percent: float,
    ) -> bool:
        """
        Compute and persist author trust metrics.

        S2-side data comes from cached DB fields (h_index, total_citations, total_papers)
        set during ingestion — no external API call needed.

        Conflict detection (any flag → is_conflict = True):
          - citation count delta >= threshold  (DB cached S2 vs OA live)
          - paper count delta >= threshold
          - h-index delta >= threshold

        Reputation score (0–100) derives solely from local data:
          base   = min(h_index, 50) * 2  →  0–100 scale
          deduct = retracted_papers_count * 10
        """
        update_data: Dict[str, Any] = {}
        has_conflict = bool(author.is_conflict)  # preserve existing flag by default

        # ── Conflict detection: cached S2 fields vs live OA data ─────────────
        if oa_data:
            # Read S2-side from DB cache — these were set during ingestion
            db_citations: Optional[int] = author.total_citations
            db_paper_count: Optional[int] = author.total_papers
            db_h_index: Optional[int] = author.h_index

            oa_citations = int(oa_data.cited_by_count)
            oa_paper_count = int(oa_data.works_count)
            oa_stats = oa_data.summary_stats or {}
            oa_h_index: Optional[int] = oa_stats.get("h_index")

            citation_conflict = self._has_citation_conflict(
                db_citations, oa_citations, conflict_threshold_percent
            )
            paper_conflict = self._has_paper_count_conflict(
                db_paper_count, oa_paper_count, conflict_threshold_percent
            )
            h_index_conflict = self._has_h_index_conflict(
                db_h_index, oa_h_index, conflict_threshold_percent
            )

            has_conflict = (citation_conflict and paper_conflict) or (citation_conflict and h_index_conflict) or (paper_conflict and h_index_conflict)
            update_data["is_conflict"] = has_conflict
            update_data["is_processed"] = True

            if has_conflict:
                logger.info(
                    "[Preprocessing] Author %s flagged as conflict "
                    "(citations=%s, papers=%s, h_index=%s)",
                    author.author_id, citation_conflict, paper_conflict, h_index_conflict,
                )

        # ── Metrics from locally-indexed papers ───────────────────────────────
        papers = await self.preprocessing_repo.get_author_papers_for_metrics(author.id)
        retracted_papers_count = sum(1 for p in papers if bool(p.is_retracted))
        has_retracted_papers = retracted_papers_count > 0
        retraction_rate = retracted_papers_count / len(papers) if len(papers) > 0 else None
        i10_index = sum(1 for p in papers if p.citation_count >= 10)

        update_data.update(
            {
                "retracted_papers_count": retracted_papers_count,
                "has_retracted_papers": has_retracted_papers,
                "retraction_rate": retraction_rate,
                "i10_index": i10_index,
                "openalex_counts_by_year": oa_data.counts_by_year if oa_data else None,
            }
        )

        await self.preprocessing_repo.update_author_metrics(author.id, update_data)
        return has_conflict

    async def _fetch_author_source_data(self, author) -> Tuple[Optional[Dict[str, Any]], Any]:
        """Fetch Semantic Scholar and optional OpenAlex author details."""
        from app.retriever.provider import SemanticScholarProvider
        from app.retriever.service import RetrievalServiceType

        semantic_data: Optional[Dict[str, Any]] = None
        openalex_data = None

        if author.author_id:
            try:
                semantic_provider = self.retriever.get_provider_as(
                    RetrievalServiceType.SEMANTIC,
                    SemanticScholarProvider,
                )
                semantic_map = await semantic_provider.get_multiple_authors(
                    [str(author.author_id)]
                )
                if semantic_map:
                    semantic_data = semantic_map.get(str(author.author_id))
            except Exception as e:
                logger.warning(
                    f"[Preprocessing] Failed to fetch S2 author detail for {author.author_id}: {e}"
                )

        if author.openalex_id:
            try:
                openalex_data = await self.retriever.get_author(
                    self._normalize_openalex_id(str(author.openalex_id))
                )
            except Exception as e:
                logger.warning(
                    f"[Preprocessing] Failed to fetch OpenAlex author detail for {author.author_id}: {e}"
                )

        return semantic_data, openalex_data

    async def _fetch_multiple_authors(self, author_oa_ids: List[str]) -> List[OAAuthorResponse]:
        """Fetch OpenAlex author details for multiple authors."""
        from app.retriever.provider import OpenAlexProvider
        from app.retriever.service import RetrievalServiceType
        
        try:
            openalex_data = await self.retriever.get_provider_as(
                RetrievalServiceType.OPENALEX,
                OpenAlexProvider,
            ).get_multiple_authors(author_oa_ids, limit=len(author_oa_ids))
            return openalex_data
        except Exception as e:
            logger.warning(
                f"[Preprocessing] Failed to fetch OpenAlex author details: {e}"
            )

        return []

    @staticmethod
    def _normalize_openalex_id(openalex_id: str) -> str:
        """Normalize OpenAlex author id (full URL -> short ID)."""
        return openalex_id.removeprefix("https://openalex.org/")

    @staticmethod
    def _has_citation_conflict(
        semantic_citations: Optional[int],
        openalex_citations: Optional[int],
        threshold_percent: float = 50.0,
    ) -> bool:
        """Detect conflict when citation delta ratio between S2 and OA >= threshold."""
        if semantic_citations is None or openalex_citations is None:
            return False

        baseline = max(int(semantic_citations), int(openalex_citations), 1)
        diff_ratio = abs(int(semantic_citations) - int(openalex_citations)) / baseline
        citation_count_conflict = diff_ratio >= (threshold_percent / 100.0)

        return citation_count_conflict

    @staticmethod
    def _has_paper_count_conflict(
        semantic_paper_count: Optional[int],
        openalex_paper_count: Optional[int],
        threshold_percent: float = 50.0,
    ) -> bool:
        """Detect conflict when paper count delta ratio between S2 and OA >= threshold."""
        if semantic_paper_count is None or openalex_paper_count is None:
            return False

        baseline = max(int(semantic_paper_count), int(openalex_paper_count), 1)
        diff_ratio = abs(int(semantic_paper_count) - int(openalex_paper_count)) / baseline
        paper_count_conflict = diff_ratio >= (threshold_percent / 100.0)

        return paper_count_conflict

    @staticmethod
    def _has_h_index_conflict(
        semantic_h_index: Optional[int],
        openalex_h_index: Optional[int],
        threshold_percent: float = 50.0,
    ) -> bool:
        """Detect conflict when h-index delta ratio between S2 and OA >= threshold."""
        if semantic_h_index is None or openalex_h_index is None:
            return False

        baseline = max(int(semantic_h_index), int(openalex_h_index), 1)
        diff_ratio = abs(int(semantic_h_index) - int(openalex_h_index)) / baseline
        h_index_conflict = diff_ratio >= (threshold_percent / 100.0)

        return h_index_conflict

    async def _process_unprocessed_papers(self, state: DBPreprocessingState) -> None:
        """
        Process papers that have is_processed = False.

        Runs the RAG pipeline (extract, chunk, embed) for papers that were created
        but not fully processed. This can happen if RAG pipeline fails during initial
        processing.

        Args:
            state: Preprocessing state for tracking
        """
        try:
            logger.info("[Preprocessing] Checking for unprocessed papers...")

            # Get unprocessed papers via repository
            papers = await self.preprocessing_repo.get_unprocessed_papers(limit=100)

            if not papers:
                logger.info("[Preprocessing] No unprocessed papers found")
                return

            logger.info(f"[Preprocessing] Processing {len(papers)} unprocessed papers")

            processed_count = 0
            error_count = 0

            for db_paper in papers:
                try:
                    # Convert to enriched DTO for processing
                    from app.core.dtos.paper import PaperEnrichedDTO

                    paper_dto = PaperEnrichedDTO.from_db_model(db_paper)

                    # Run RAG pipeline
                    success = await self._run_rag_pipeline(
                        paper_dto, str(db_paper.paper_id)
                    )

                    if success:
                        processed_count += 1
                        logger.info(
                            f"[Preprocessing] Successfully processed paper "
                            f"{db_paper.paper_id}"
                        )
                    else:
                        error_count += 1
                        logger.warning(
                            f"[Preprocessing] Failed to process paper "
                            f"{db_paper.paper_id}"
                        )

                    # Commit after each paper to avoid losing progress
                    await self._commit_and_clear_session()

                except Exception as e:
                    error_count += 1
                    logger.error(
                        f"[Preprocessing] Error processing paper "
                        f"{db_paper.paper_id}: {e}"
                    )
                    await self._log_error_to_file(
                        job_id=state.job_id,
                        stage="process_unprocessed_papers",
                        message=f"Error processing unprocessed paper {db_paper.paper_id}",
                        error=e,
                        context={"paper_id": str(db_paper.paper_id)},
                    )
                    await self._commit_and_clear_session()

            logger.info(
                f"[Preprocessing] Finished processing unprocessed papers: "
                f"{processed_count} successful, {error_count} errors"
            )

        except Exception as e:
            logger.error(
                f"[Preprocessing] Error processing unprocessed papers: {e}",
                exc_info=True,
            )

    async def _link_citations_from_batch(self, state: DBPreprocessingState) -> None:
        """
        Link citations/references between papers using collected data.

        Accepts that most cited papers won't exist in the database yet.
        The batch_link_citations_references method only creates citations
        for papers that exist, which is the expected behavior.

        Args:
            state: Preprocessing state with accumulated citation data
        """
        # Check if we have collected citation data
        if not hasattr(state, "_citation_batch") or not state._citation_batch:  # type: ignore
            logger.info("[Preprocessing] No citations to link")
            return

        citation_data = state._citation_batch  # type: ignore

        if not citation_data:
            logger.info("[Preprocessing] Citation batch is empty")
            return

        logger.info(
            f"[Preprocessing] Linking citations for {len(citation_data)} papers"
        )

        try:
            # Call batch citation linking service
            linked_count = (
                await self.linking_service.batch_link_citations_references(
                    citation_data=citation_data
                )
            )

            logger.info(
                f"[Preprocessing] Successfully linked {linked_count} citations "
                f"(most cited papers not yet indexed, which is expected)"
            )
        except Exception as e:
            logger.error(f"[Preprocessing] Error linking citations: {e}", exc_info=True)
            await self._log_error_to_file(
                job_id=state.job_id,
                stage="citation_linking",
                message="Failed linking citations from batch",
                error=e,
                context={"citation_batch_size": len(citation_data)},
            )

    def _extract_references_from_paper(self, paper: Any) -> List[str]:
        """
        Extract reference paper IDs from enriched paper data.

        Accepts that most references won't be indexed yet - this is fine.
        The batch_link_citations_references method handles missing papers gracefully.

        Args:
            paper: PaperEnrichedDTO with references field

        Returns:
            List of referenced paper IDs (can be empty)
        """
        references = []

        # Check if paper has references field
        if not hasattr(paper, "references") or not paper.references:
            return references

        # Extract paper IDs from references
        # References format: List[Dict[str, Any]] with 'paperId' field
        for ref in paper.references:
            if isinstance(ref, dict):
                ref_id = ref.get("paperId")
                if ref_id:
                    references.append(str(ref_id))
            elif hasattr(ref, "paperId"):
                if ref.paperId:
                    references.append(str(ref.paperId))

        return references

    def _filter_open_access_papers(
        self, papers_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter for papers with open access PDFs only."""
        return [
            p for p in papers_data if p.get("isOpenAccess") and p.get("openAccessPdf")
        ]

    def _extract_paper_ids(self, papers: List[Dict[str, Any]]) -> List[str]:
        """Extract valid paper IDs from paper data."""
        paper_ids = []
        for p in papers:
            paper_id = p.get("paperId")
            if paper_id:
                paper_ids.append(str(paper_id))
        return paper_ids

    # ==================== State Management ====================

    async def _initialize_job_state(
        self, job_id: str, target_count: int, resume: bool
    ) -> DBPreprocessingState:
        """Get existing state or create new one."""
        state = await self.preprocessing_repo.get_state_by_job_id(job_id)

        if state and not resume:
            # Reset for fresh start
            state.current_index = 0
            state.processed_count = 0
            state.skipped_count = 0
            state.error_count = 0
            state.target_count = target_count
            state.is_completed = False
            state.is_running = False
            state.is_paused = False
            state.completed_at = None  # type: ignore
            state.continuation_token = None  # type: ignore
        elif not state:
            # Create new state
            state = await self.preprocessing_repo.create_state(
                job_id=job_id,
                target_count=target_count,
            )
            return state
        else:
            # Resume - clear pause flag
            if state.is_paused:
                state.is_paused = False
                state.status_message = "Resuming from pause..."  # type: ignore

        await self.preprocessing_repo.save_state(state, refresh=True)
        return state

    async def _refresh_state(self, job_id: str) -> DBPreprocessingState:
        """Re-fetch state from database."""
        state = await self.preprocessing_repo.get_state_by_job_id(job_id)
        if not state:
            raise ValueError(f"Preprocessing state not found for job_id={job_id}")
        return state

    async def _update_state(
        self,
        state: DBPreprocessingState,
        is_running: Optional[bool] = None,
        message: Optional[str] = None,
    ) -> None:
        """Update state fields and commit."""
        if is_running is not None:
            state.is_running = is_running
        if message is not None:
            state.status_message = message  # type: ignore
        await self.preprocessing_repo.save_state(state)

    async def _complete_job(self, state: DBPreprocessingState) -> None:
        """Mark job as completed."""
        state.is_completed = True
        state.is_running = False
        state.completed_at = datetime.now()  # type: ignore
        state.status_message = (
            f"Completed: {state.processed_count} papers processed"
        )  # type: ignore
        await self.preprocessing_repo.save_state(state)

    async def _commit_and_clear_session(self) -> None:
        """Commit changes and clear session to free memory."""
        await self.db_session.commit()
        self.db_session.expunge_all()

    def _state_to_stats(self, state: DBPreprocessingState) -> Dict[str, Any]:
        """Convert state to statistics dict with computed metrics."""
        papers_per_second = 0.0
        eta_seconds = None

        if state.created_at and state.updated_at and state.is_running:
            try:
                # Calculate elapsed time
                elapsed = (state.updated_at - state.created_at).total_seconds()  # type: ignore
                if elapsed > 0 and state.processed_count > 0:
                    papers_per_second = state.processed_count / elapsed
                    remaining = state.target_count - state.processed_count
                    if papers_per_second > 0:
                        eta_seconds = int(remaining / papers_per_second)
            except (TypeError, AttributeError):
                # Handle datetime conversion issues gracefully
                pass

        progress_percent = 0.0
        if state.target_count > 0:
            progress_percent = round(
                (state.processed_count / state.target_count * 100), 1
            )

        return {
            "job_id": state.job_id,
            "current_index": state.current_index,
            "processed_count": state.processed_count,
            "skipped_count": state.skipped_count,
            "error_count": state.error_count,
            "target_count": state.target_count,
            "is_completed": state.is_completed,
            "is_running": state.is_running,
            "is_paused": getattr(state, "is_paused", False),
            "status_message": getattr(state, "status_message", None),
            "current_file": getattr(state, "current_file", None),
            "continuation_token": getattr(state, "continuation_token", None),
            "papers_per_second": round(papers_per_second, 2),
            "eta_seconds": eta_seconds,
            "progress_percent": progress_percent,
            "created_at": str(state.created_at) if state.created_at else None,
            "updated_at": str(state.updated_at) if state.updated_at else None,
            "completed_at": str(state.completed_at) if state.completed_at else None,
        }

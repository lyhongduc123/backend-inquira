"""
Bulk preprocessing service using Semantic Scholar bulk search API.

Clean architecture with retriever service integration for:
- Bulk search API calls
- Batch paper details fetching
- OpenAlex enrichment
- Database caching
- Progress tracking
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.extensions.logger import create_logger
from app.domain.papers import PaperRepository, PaperService
from app.domain.papers.journal_service import JournalService
from app.domain.papers.conference_service import ConferenceService
from app.domain.papers.enrichment_service import PaperEnrichmentService
from app.retriever.service import RetrievalService
from app.utils.transformers import batch_normalized_to_papers
from app.processor.paper_processor import PaperProcessor
from app.processor.preprocessing_repository import PreprocessingRepository
from app.models.preprocessing_state import DBPreprocessingState
from app.models.papers import DBPaper
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
        enrichment_service: Optional[PaperEnrichmentService] = None,
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
            enrichment_service: Service for citation/reference linking (optional)
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
        self.enrichment_service = enrichment_service or PaperEnrichmentService(
            db=db_session, paper_repository=self.repository
        )

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

        Workflow:
        1. Initialize/resume job state
        2. Fetch papers via bulk search (paginated)
        3. Batch fetch full details + enrich with OpenAlex
        4. Check database cache (skip existing)
        5. Create papers and process through RAG pipeline

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
            state, is_running=True, message="Starting bulk search..."
        )

        try:
            continuation_token = state.continuation_token if resume else None

            # Main processing loop
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
                state.continuation_token = continuation_token  # type: ignore
                await self.db_session.commit()

                # Process batch
                await self._process_batch(papers_data, state, target_count)

                # Check if we should stop
                state = await self._refresh_state(job_id)
                if not continuation_token or state.processed_count >= target_count:
                    break

                # Update progress
                progress_msg = (
                    f"Processed: {state.processed_count}/{target_count} | "
                    f"Skipped: {state.skipped_count} | "
                    f"Errors: {state.error_count}"
                )
                await self._update_state(state, message=progress_msg)

            state = await self._refresh_state(job_id)
            await self._link_citations_from_batch(state)

            # Generate embeddings for papers missing them
            await self._generate_missing_embeddings(state)

            # Process any unprocessed papers
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
            state = await self._refresh_state(job_id)
            if state:
                await self._update_state(
                    state, is_running=False, message=f"Error: {str(e)}"
                )
            raise

    # ==================== Batch Processing ====================

    async def _process_batch(
        self,
        papers_data: List[Dict[str, Any]],
        state: DBPreprocessingState,
        target_count: int,
    ) -> None:
        """
        Process a batch of papers from bulk search results.

        Steps:
        1. Filter for open access papers only
        2. Extract paper IDs
        3. Batch fetch full details via RetrievalService
        4. Enrich with OpenAlex metadata
        5. Transform to Paper DTOs
        6. Check database cache
        7. Create + process through RAG pipeline
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
                sub_enriched = await self._fetch_and_enrich_papers(sub_batch_ids)
                enriched_papers.extend(sub_enriched)
                idx += 1

                from app.retriever.result_logger import save_results_to_json

                save_results_to_json(
                    [p.model_dump(mode='json') for p in sub_enriched], output_dir="preprocessing_logs"
                )
        else:
            enriched_papers = await self._fetch_and_enrich_papers(paper_ids)

        if not enriched_papers:
            logger.warning("[Preprocessing] No enriched papers returned")
            return

        logger.info(
            f"[Preprocessing] Processing {len(enriched_papers)} enriched papers"
        )

        # Process each paper
        for paper in enriched_papers:
            # Cache state values before any session operations to avoid greenlet errors
            cached_processed_count = state.processed_count
            cached_job_id = state.job_id
            
            if cached_processed_count >= target_count:
                break

            # Check for pause
            state = await self._refresh_state(cached_job_id)
            if state.is_paused:
                logger.info(f"[Preprocessing] Job {cached_job_id} paused during batch")
                return

            await self._process_single_paper(paper, state, target_count)

    async def _fetch_and_enrich_papers(self, paper_ids: List[str]) -> List[Any]:
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
            return []

    async def _process_single_paper(
        self, paper: Any, state: DBPreprocessingState, target_count: int
    ) -> None:
        """
        Process a single paper: check cache, create, link journal/conference, and run RAG pipeline.

        Handles:
        - Database cache lookup (skip if exists)
        - Paper creation via PaperService
        - Journal/Conference linking (if venue/ISSN available)
        - RAG pipeline processing (extract, chunk, embed)
        - State updates and error handling
        - Memory management (commit + expunge)

        Returns citation data for later batch linking.
        """
        try:
            paper_id = str(paper.paper_id)

            # Check if already exists (skip if exists)
            if await self.preprocessing_repo.paper_exists(paper_id):
                state.skipped_count += 1
                state.current_index += 1
                logger.debug(f"Paper {paper_id} already exists, skipping")
                return

            # Create paper in database
            db_paper = await self.paper_service.create_paper_from_schema(paper)
            if not db_paper:
                logger.warning(f"Failed to create paper {paper_id}")
                state.error_count += 1
                state.current_index += 1
                return

            logger.info(f"Created paper {paper_id}")

            # Link to journal if ISSN available
            if db_paper.issn or db_paper.issn_l or db_paper.venue:
                try:
                    journal = await self.journal_service.enrich_paper_with_journal(
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

            # Link to conference if venue available
            if db_paper.venue:
                try:
                    conference = (
                        await self.conference_service.enrich_paper_with_conference(
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

            # Extract references for later citation linking
            references_data = self._extract_references_from_paper(paper)
            if references_data:
                # Store for batch processing (add to state or batch accumulator)
                # For now, we'll track this in memory and process at job completion
                if not hasattr(state, "_citation_batch"):
                    state._citation_batch = []  # type: ignore 
                state._citation_batch.append((paper_id, references_data))  # type: ignore

            # Process through RAG pipeline
            success = await self._run_rag_pipeline(paper, paper_id)

            # Update state
            if success:
                state.processed_count += 1
                logger.info(f"[Preprocessing] Successfully processed {paper_id}")
            else:
                state.error_count += 1
                logger.warning(f"[Preprocessing] Failed to process {paper_id}")

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
                    f"[Preprocessing] Progress: {processed_count}/{target_count}"
                )

        except Exception as e:
            logger.error(f"[Preprocessing] Error processing paper: {e}")
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
            await self._commit_and_clear_session()
            state = await self._refresh_state(job_id)
            state.error_count = error_count
            state.current_index = current_index
            await self.db_session.commit()

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
            return None

    # ==================== Helper Functions ====================

    async def _generate_missing_embeddings(self, state: DBPreprocessingState) -> None:
        """
        Generate title + abstract embeddings for papers that don't have them.

        This ensures all papers in the database have embeddings for semantic search.
        Particularly useful for papers that failed during initial processing.

        Args:
            state: Preprocessing state for tracking
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

            # Convert to DTO format for processor
            from app.core.dtos.paper import PaperDTO

            paper_dtos = [PaperDTO.from_db_model(p) for p in papers]

            # Generate embeddings
            papers_with_embeddings = await self.processor.generate_paper_embeddings(paper_dtos)  # type: ignore

            # Update database with new embeddings
            paper_embeddings = {
                str(p.paper_id): p.embedding
                for p in papers_with_embeddings
                if p.embedding is not None
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
                await self.enrichment_service.batch_link_citations_references(
                    citation_data=citation_data
                )
            )

            logger.info(
                f"[Preprocessing] Successfully linked {linked_count} citations "
                f"(most cited papers not yet indexed, which is expected)"
            )
        except Exception as e:
            logger.error(f"[Preprocessing] Error linking citations: {e}", exc_info=True)

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
        stmt = select(DBPreprocessingState).where(DBPreprocessingState.job_id == job_id)
        result = await self.db_session.execute(stmt)
        state = result.scalar_one_or_none()

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
            state = DBPreprocessingState(
                job_id=job_id,
                target_count=target_count,
                current_index=0,
                processed_count=0,
                skipped_count=0,
                error_count=0,
                is_completed=False,
                is_running=False,
                is_paused=False,
            )
            self.db_session.add(state)
        else:
            # Resume - clear pause flag
            if state.is_paused:
                state.is_paused = False
                state.status_message = "Resuming from pause..."  # type: ignore

        await self.db_session.commit()
        await self.db_session.refresh(state)
        return state

    async def _refresh_state(self, job_id: str) -> DBPreprocessingState:
        """Re-fetch state from database."""
        stmt = select(DBPreprocessingState).where(DBPreprocessingState.job_id == job_id)
        result = await self.db_session.execute(stmt)
        return result.scalar_one()

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
        await self.db_session.commit()

    async def _complete_job(self, state: DBPreprocessingState) -> None:
        """Mark job as completed."""
        state.is_completed = True
        state.is_running = False
        state.completed_at = datetime.now()  # type: ignore
        state.status_message = (
            f"Completed: {state.processed_count} papers processed"
        )  # type: ignore
        await self.db_session.commit()

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

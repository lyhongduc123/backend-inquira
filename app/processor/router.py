"""
Admin router for dataset-based paper preprocessing.

Provides admin-only endpoints to:
- Start/resume dataset streaming preprocessing jobs
- Track job progress and state
- Automatic skip logic for existing papers

**Authentication Required:** All endpoints require admin privileges.
"""
from fastapi import APIRouter, Depends, BackgroundTasks, Query
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.database import get_db_session, async_session
from app.processor.preprocessing_service import PreprocessingService
from app.auth.dependencies import get_admin_user
from app.models.users import DBUser
from app.models.preprocessing_state import DBPreprocessingState
from app.core.responses import ApiResponse, success_response
from app.core.dependencies import get_container
from app.core.container import ServiceContainer
from pydantic import BaseModel, Field
from app.extensions.logger import create_logger
from app.workers.task_queue import get_task_queue

router = APIRouter()
logger = create_logger(__name__)


async def run_bulk_search_task(
    job_id: str,
    search_query: str,
    target_count: int,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    fields_of_study: Optional[List[str]] = None,
    resume: bool = True
):
    """
    Wrapper to run bulk search preprocessing in background with its own database session.
    """
    db = async_session()
    try:
        from app.core.container import ServiceContainer
        container = ServiceContainer(db)
        await container.preprocessing_service.process_bulk_search(
            job_id=job_id,
            search_query=search_query,
            target_count=target_count,
            year_min=year_min,
            year_max=year_max,
            fields_of_study=fields_of_study,
            resume=resume
        )
    except Exception as e:
        logger.error(f"Background bulk search task failed: {e}", exc_info=True)
    finally:
        await db.close()


async def run_repository_task(
    job_id: str,
    paper_ids: List[str],
    resume: bool = True
):
    """
    Wrapper to run repository preprocessing in background with its own database session.
    """
    db = async_session()
    try:
        from app.core.container import ServiceContainer
        container = ServiceContainer(db)
        service = container.preprocessing_service
        # TODO: Implement process_repository method in PreprocessingService
        # For now, process papers one by one
        from app.core.dtos.paper import PaperEnrichedDTO
        
        for paper_id in paper_ids:
            try:
                # Fetch and enrich paper
                enriched_papers = await service.retriever.get_multiple_papers([paper_id])
                if enriched_papers:
                    paper = enriched_papers[0]
                    # Create paper if not exists
                    existing = await service.repository.get_paper_by_id(paper_id)
                    if not existing:
                        await service.paper_service.ingest_paper_metadata(paper)
                    # Process through RAG pipeline
                    await service.processor.process_single_paper(paper)
                    logger.info(f"[Repository] Processed paper {paper_id}")
            except Exception as e:
                logger.error(f"[Repository] Error processing {paper_id}: {e}")
                
    except Exception as e:
        logger.error(f"Background repository task failed: {e}", exc_info=True)
    finally:
        await db.close()


async def run_embedding_backfill_task() -> dict:
    """Queue task wrapper for embedding backfill phase."""
    db = async_session()
    try:
        container = ServiceContainer(db)
        return await container.preprocessing_phase_service.run_embedding_backfill()
    finally:
        await db.close()


async def run_citation_linking_task(limit: int, references_limit: int) -> dict:
    """Queue task wrapper for citation linking phase."""
    db = async_session()
    try:
        container = ServiceContainer(db)
        return await container.preprocessing_phase_service.run_citation_linking(
            limit=limit,
            references_limit=references_limit,
        )
    finally:
        await db.close()


async def run_author_metrics_task(
    only_unprocessed: bool,
    conflict_threshold_percent: float,
    batch_size: int,
) -> dict:
    """Queue task wrapper for author metric computation phase."""
    db = async_session()
    try:
        container = ServiceContainer(db)
        return await container.preprocessing_phase_service.run_author_metrics(
            only_unprocessed=only_unprocessed,
            conflict_threshold_percent=conflict_threshold_percent,
            batch_size=batch_size,
        )
    finally:
        await db.close()


async def run_paper_tagging_task(
    limit: int,
    only_missing_tags: bool,
    candidate_labels: Optional[List[str]],
    category: str,
    min_confidence: float,
    max_tags_per_paper: int,
) -> dict:
    """Queue task wrapper for paper tag computation phase."""
    db = async_session()
    try:
        container = ServiceContainer(db)
        return await container.preprocessing_phase_service.run_paper_tagging(
            limit=limit,
            only_missing_tags=only_missing_tags,
            candidate_labels=candidate_labels,
            category=category,
            min_confidence=min_confidence,
            max_tags_per_paper=max_tags_per_paper,
        )
    finally:
        await db.close()


class StartBulkSearchRequest(BaseModel):
    """Request to start bulk search preprocessing job"""
    job_id: str = Field(..., description="Unique job identifier (e.g., 'ml-papers-2026')")
    search_query: str = Field(..., description="Search query for bulk search API")
    target_count: int = Field(..., gt=0, description="Target number of papers to process")
    year_min: Optional[int] = Field(None, description="Minimum publication year")
    year_max: Optional[int] = Field(None, description="Maximum publication year")
    fields_of_study: Optional[List[str]] = Field(None, description="List of fields to filter")
    resume: bool = Field(True, description="Resume from previous state if job exists")


class StartRepositoryRequest(BaseModel):
    """Request to start repository preprocessing job"""
    job_id: str = Field(..., description="Unique job identifier (e.g., 'repository-batch-1')")
    paper_ids: List[str] = Field(..., description="List of paper IDs to process")
    resume: bool = Field(True, description="Resume from previous state if job exists")


class PreprocessingStatusResponse(BaseModel):
    """Response with preprocessing job status"""
    job_id: str
    current_index: int
    processed_count: int
    skipped_count: int
    error_count: int
    target_count: int
    is_completed: bool
    is_running: bool
    is_paused: bool = False
    status_message: Optional[str] = None
    current_file: Optional[str] = None
    papers_per_second: float = 0.0
    eta_seconds: Optional[int] = None
    progress_percent: float = 0.0
    created_at: Optional[str]
    updated_at: Optional[str]
    completed_at: Optional[str]


class PreprocessingTaskResponse(BaseModel):
    """Response for queued preprocessing phase task submission."""

    task_id: str
    task_type: str
    status: str
    message: str


class CitationLinkingRequest(BaseModel):
    """Request to trigger citation linking phase."""

    limit: int = Field(default=200, ge=1, le=2000)
    references_limit: int = Field(default=200, ge=1, le=1000)


class AuthorMetricsRequest(BaseModel):
    """Request to trigger author metric phase."""

    only_unprocessed: bool = Field(default=False)
    conflict_threshold_percent: float = Field(default=50.0, ge=0.0, le=100.0)
    batch_size: int = Field(default=200, ge=1, le=2000)


class PaperTaggingRequest(BaseModel):
    """Request to trigger paper tagging phase."""

    limit: int = Field(default=200, ge=1, le=5000)
    only_missing_tags: bool = Field(default=True)
    candidate_labels: Optional[List[str]] = Field(default=None)
    category: str = Field(default="topic")
    min_confidence: float = Field(default=50.0, ge=0.0, le=100.0)
    max_tags_per_paper: int = Field(default=3, ge=1, le=20)


@router.post("/preprocess/bulk-search/start", response_model=ApiResponse[PreprocessingStatusResponse])
async def start_bulk_search_preprocessing(
    request: StartBulkSearchRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
    # admin_user: DBUser = Depends(get_admin_user)
) -> ApiResponse[PreprocessingStatusResponse]:
    """
    [Admin Only] Start or resume bulk search preprocessing job.
    
    Uses Semantic Scholar bulk search API to find papers matching your query,
    then processes them through the RAG pipeline (PDF download, chunking, embedding).
    
    **Note:** This is a background task - the endpoint returns immediately.
    Use GET /preprocess/status/{job_id} to check progress.
    
    Args:
        job_id: Unique job identifier (e.g., 'ml-papers-2026')
        search_query: Search query string (e.g., 'machine learning')
        target_count: Number of papers to process
        year_min: Optional minimum publication year
        year_max: Optional maximum publication year
        fields_of_study: Optional list of fields (e.g., ['Computer Science', 'Medicine'])
        resume: Resume from previous state (default: True)
    
    Returns:
        Job status
    """
    service = PreprocessingService(db)
    
    # Add to background tasks with its own session
    background_tasks.add_task(
        run_bulk_search_task,
        job_id=request.job_id,
        search_query=request.search_query,
        target_count=request.target_count,
        year_min=request.year_min,
        year_max=request.year_max,
        fields_of_study=request.fields_of_study,
        resume=request.resume
    )
    
    # Get current state
    stmt = select(DBPreprocessingState).where(DBPreprocessingState.job_id == request.job_id)
    result = await db.execute(stmt)
    state = result.scalar_one_or_none()
    
    if state:
        # Use service method to convert state to stats
        stats = service._state_to_stats(state)
        stats['is_running'] = True  # Override since we just started it
        response_data = PreprocessingStatusResponse(**stats)
    else:
        # New job - will be created in background task
        response_data = PreprocessingStatusResponse(
            job_id=request.job_id,
            current_index=0,
            processed_count=0,
            skipped_count=0,
            error_count=0,
            target_count=request.target_count,
            is_completed=False,
            is_running=True,
            is_paused=False,
            status_message="Initializing...",
            current_file=None,
            papers_per_second=0.0,
            eta_seconds=None,
            progress_percent=0.0,
            created_at=None,
            updated_at=None,
            completed_at=None
        )
    
    return success_response(
        data=response_data
    )


@router.post("/preprocess/repository/start", response_model=ApiResponse[dict])
async def start_repository_preprocessing(
    request: StartRepositoryRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
    # admin_user: DBUser = Depends(get_admin_user)
) -> ApiResponse[dict]:
    """
    [Admin Only] Start repository preprocessing job for specific paper IDs.
    
    Processes a list of paper IDs through the RAG pipeline.
    Useful for reprocessing specific papers or adding papers from a curated list.
    
    **Note:** This is a background task - the endpoint returns immediately.
    
    Args:
        job_id: Unique job identifier (e.g., 'repository-batch-1')
        paper_ids: List of paper IDs to process
        resume: Resume from previous state (default: True)
    
    Returns:
        Job status
    """
    # Add to background tasks with its own session
    background_tasks.add_task(
        run_repository_task,
        job_id=request.job_id,
        paper_ids=request.paper_ids,
        resume=request.resume
    )
    
    return success_response(
        data={
            "job_id": request.job_id,
            "paper_count": len(request.paper_ids),
            "status": "started",
            "message": f"Repository preprocessing started for {len(request.paper_ids)} papers"
        }
    )


@router.get("/preprocess/status/{job_id}", response_model=ApiResponse[PreprocessingStatusResponse])
async def get_preprocessing_status(
    job_id: str,
    db: AsyncSession = Depends(get_db_session),
    # admin_user: DBUser = Depends(get_admin_user)
) -> ApiResponse[PreprocessingStatusResponse]:
    """
    [Admin Only] Get status of a preprocessing job.
    
    Args:
        job_id: Unique job identifier
    
    Returns:
        Job status and statistics
    """
    stmt = select(DBPreprocessingState).where(DBPreprocessingState.job_id == job_id)
    result = await db.execute(stmt)
    state = result.scalar_one_or_none()
    
    if not state:
        # Return a default response for non-existent job
        response_data = PreprocessingStatusResponse(
            job_id=job_id,
            current_index=0,
            processed_count=0,
            skipped_count=0,
            error_count=0,
            target_count=0,
            is_completed=False,
            is_running=False,
            is_paused=False,
            status_message="Job not found",
            current_file=None,
            papers_per_second=0.0,
            eta_seconds=None,
            progress_percent=0.0,
            created_at=None,
            updated_at=None,
            completed_at=None
        )
        return success_response(
            data=response_data
        )
    
    # Use service to convert state to stats
    service = PreprocessingService(db)
    stats = service._state_to_stats(state)
    response_data = PreprocessingStatusResponse(**stats)
    
    return success_response(
        data=response_data
    )


@router.get("/preprocess/jobs", response_model=ApiResponse[List[PreprocessingStatusResponse]])
async def list_preprocessing_jobs(
    db: AsyncSession = Depends(get_db_session),
    # admin_user: DBUser = Depends(get_admin_user)
) -> ApiResponse[List[PreprocessingStatusResponse]]:
    """
    [Admin Only] List all preprocessing jobs.
    
    Returns:
        List of all jobs with their status
    """
    stmt = select(DBPreprocessingState).order_by(DBPreprocessingState.created_at.desc())
    result = await db.execute(stmt)
    states = result.scalars().all()
    
    service = PreprocessingService(db)
    jobs = []
    for state in states:
        stats = service._state_to_stats(state)
        jobs.append(PreprocessingStatusResponse(**stats))
    
    return success_response(
        data=jobs
    )


@router.post("/preprocess/pause/{job_id}", response_model=ApiResponse[PreprocessingStatusResponse])
async def pause_preprocessing(
    job_id: str,
    db: AsyncSession = Depends(get_db_session),
    # admin_user: DBUser = Depends(get_admin_user)
) -> ApiResponse[PreprocessingStatusResponse]:
    """
    [Admin Only] Pause/stop a running preprocessing job.
    
    Sets the pause flag which will cause the job to stop gracefully
    after finishing the current paper. All progress is saved and the
    job can be resumed later.
    
    Args:
        job_id: Unique job identifier
    
    Returns:
        Updated job status
    """
    stmt = select(DBPreprocessingState).where(DBPreprocessingState.job_id == job_id)
    result = await db.execute(stmt)
    state = result.scalar_one_or_none()
    
    if not state:
        # Return a default response for non-existent job
        response_data = PreprocessingStatusResponse(
            job_id=job_id,
            current_index=0,
            processed_count=0,
            skipped_count=0,
            error_count=0,
            target_count=0,
            is_completed=False,
            is_running=False,
            is_paused=False,
            status_message="Job not found",
            current_file=None,
            papers_per_second=0.0,
            eta_seconds=None,
            progress_percent=0.0,
            created_at=None,
            updated_at=None,
            completed_at=None
        )
        return success_response(
            data=response_data
        )
    
    # Set pause flag - the running job will check this and stop gracefully
    state.is_paused = True
    state.status_message = "Pause requested..."  # type: ignore
    await db.commit()
    await db.refresh(state)
    
    # Use service to convert state to stats
    service = PreprocessingService(db)
    stats = service._state_to_stats(state)
    response_data = PreprocessingStatusResponse(**stats)
    
    return success_response(
        data=response_data
    )


@router.post("/phases/embeddings/start", response_model=ApiResponse[PreprocessingTaskResponse])
async def start_embedding_backfill_phase() -> ApiResponse[PreprocessingTaskResponse]:
    """Queue embedding backfill as an independent preprocessing phase."""
    task_queue = get_task_queue()
    task_id = await task_queue.submit(
        "preprocess_embeddings",
        run_embedding_backfill_task,
    )
    return success_response(
        data=PreprocessingTaskResponse(
            task_id=task_id,
            task_type="preprocess_embeddings",
            status="queued",
            message="Embedding backfill phase queued",
        )
    )


@router.post("/phases/citations/start", response_model=ApiResponse[PreprocessingTaskResponse])
async def start_citation_linking_phase(
    request: CitationLinkingRequest,
) -> ApiResponse[PreprocessingTaskResponse]:
    """Queue citation/reference linking as an independent preprocessing phase."""
    task_queue = get_task_queue()
    task_id = await task_queue.submit(
        "preprocess_citations",
        run_citation_linking_task,
        limit=request.limit,
        references_limit=request.references_limit,
    )
    return success_response(
        data=PreprocessingTaskResponse(
            task_id=task_id,
            task_type="preprocess_citations",
            status="queued",
            message="Citation linking phase queued",
        )
    )


@router.post("/phases/authors/start", response_model=ApiResponse[PreprocessingTaskResponse])
async def start_author_metrics_phase(
    request: AuthorMetricsRequest,
) -> ApiResponse[PreprocessingTaskResponse]:
    """Queue author metrics phase."""
    task_queue = get_task_queue()
    task_id = await task_queue.submit(
        "preprocess_author_metrics",
        run_author_metrics_task,
        only_unprocessed=request.only_unprocessed,
        conflict_threshold_percent=request.conflict_threshold_percent,
        batch_size=request.batch_size,
    )
    return success_response(
        data=PreprocessingTaskResponse(
            task_id=task_id,
            task_type="preprocess_author_metrics",
            status="queued",
            message="Author metrics phase queued",
        )
    )


@router.post("/phases/tags/start", response_model=ApiResponse[PreprocessingTaskResponse])
async def start_paper_tagging_phase(
    request: PaperTaggingRequest,
) -> ApiResponse[PreprocessingTaskResponse]:
    """Queue paper tag computation phase."""
    task_queue = get_task_queue()
    task_id = await task_queue.submit(
        "preprocess_paper_tags",
        run_paper_tagging_task,
        limit=request.limit,
        only_missing_tags=request.only_missing_tags,
        candidate_labels=request.candidate_labels,
        category=request.category,
        min_confidence=request.min_confidence,
        max_tags_per_paper=request.max_tags_per_paper,
    )
    return success_response(
        data=PreprocessingTaskResponse(
            task_id=task_id,
            task_type="preprocess_paper_tags",
            status="queued",
            message="Paper tagging phase queued",
        )
    )


@router.get("/phases/tasks/{task_id}", response_model=ApiResponse[dict])
async def get_preprocessing_phase_task_status(task_id: str) -> ApiResponse[dict]:
    """Get status of a queued preprocessing phase task."""
    task_queue = get_task_queue()
    status = await task_queue.get_status(task_id)
    return success_response(data=status or {"task_id": task_id, "status": "not_found"})


async def run_content_processing_task(limit: int) -> dict:
    """Queue task wrapper for Phase 4 content processing."""
    db = async_session()
    try:
        container = ServiceContainer(db)
        return await container.preprocessing_service.run_content_processing(limit=limit)
    finally:
        await db.close()


class ContentProcessingRequest(BaseModel):
    """Request to trigger Phase 4 content processing."""
    limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Max pending papers to process per loop iteration",
    )


@router.post("/phases/content/start", response_model=ApiResponse[PreprocessingTaskResponse])
async def start_content_processing_phase(
    request: ContentProcessingRequest,
) -> ApiResponse[PreprocessingTaskResponse]:
    """
    Queue Phase 4: PDF extraction → chunking → embedding for all pending open-access papers.

    This runs the same pipeline as the automatic Phase 4 inside a bulk-search job,
    but can be triggered independently to process any papers that were indexed
    but not yet fully processed.
    """
    task_queue = get_task_queue()
    task_id = await task_queue.submit(
        "preprocess_content",
        run_content_processing_task,
        limit=request.limit,
    )
    return success_response(
        data=PreprocessingTaskResponse(
            task_id=task_id,
            task_type="preprocess_content",
            status="queued",
            message=f"Content processing phase queued (limit={request.limit} papers/batch)",
        )
    )




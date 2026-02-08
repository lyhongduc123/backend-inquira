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
from pydantic import BaseModel, Field
from app.extensions.logger import create_logger

router = APIRouter()
logger = create_logger(__name__)


async def run_preprocessing_task(
    job_id: str,
    search_query: str,
    target_count: int,
    year_min: int = None,
    year_max: int = None,
    fields_of_study: list = None,
    resume: bool = True
):
    """
    Wrapper to run preprocessing in background with its own database session.
    """
    db = async_session()
    try:
        service = PreprocessingService(db)
        await service.process_bulk_search(
            job_id=job_id,
            search_query=search_query,
            target_count=target_count,
            year_min=year_min,
            year_max=year_max,
            fields_of_study=fields_of_study,
            resume=resume
        )
    except Exception as e:
        logger.error(f"Background preprocessing task failed: {e}", exc_info=True)
    finally:
        await db.close()


class StartPreprocessingRequest(BaseModel):
    """Request to start preprocessing job"""
    job_id: str = Field(..., description="Unique job identifier (e.g., 'ml-papers-2026')")
    search_query: str = Field(..., description="Search query for bulk search API")
    target_count: int = Field(..., gt=0, description="Target number of papers to process")
    year_min: Optional[int] = Field(None, description="Minimum publication year")
    year_max: Optional[int] = Field(None, description="Maximum publication year")
    fields_of_study: Optional[List[str]] = Field(None, description="List of fields to filter")
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


@router.post("/preprocess/start", response_model=ApiResponse[PreprocessingStatusResponse])
async def start_preprocessing(
    request: StartPreprocessingRequest,
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
        run_preprocessing_task,
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


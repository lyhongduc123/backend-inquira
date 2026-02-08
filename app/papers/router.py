"""
Paper router for CRUD operations
"""
from fastapi import APIRouter, Query, Depends, Request
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db_session
from app.papers.service import PaperService
from app.papers.repository import PaperRepository
from app.chunks.service import ChunkService
from app.chunks.repository import ChunkRepository
from app.chunks.schemas import ChunkResponse
from app.papers.schemas import (
    PaperDetail,
    PaperSummary,
    PaperUpdate,
)
from app.auth.dependencies import get_current_user
from app.models.users import DBUser
from app.core.responses import ApiResponse, PaginatedData, success_response, paginated_response
from app.core.exceptions import NotFoundException

router = APIRouter()


@router.get("", response_model=ApiResponse[PaginatedData[PaperSummary]])
async def list_papers(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    processed_only: bool = Query(False, description="Show only processed papers"),
    source: Optional[str] = Query(None, description="Filter by source (openalex, semantic, arxiv)"),
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> ApiResponse[PaginatedData[PaperSummary]]:
    """
    List all papers with pagination
    
    - **page**: Page number for pagination
    - **page_size**: Number of items per page
    - **processed_only**: Filter to show only processed papers
    - **source**: Filter by paper source
    """
    repository = PaperRepository(db)
    service = PaperService(repository)
    
    papers, total = await service.list_papers(
        page=page,
        page_size=page_size,
        processed_only=processed_only,
        source=source
    )
    
    request_id = getattr(request.state, 'request_id', None)
    return paginated_response(
        data=papers,
        total=total,
        page=page,
        page_size=page_size,
        request_id=request_id
    )


@router.get("/{paper_id}", response_model=ApiResponse[PaperDetail])
async def get_paper(
    request: Request,
    paper_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> ApiResponse[PaperDetail]:
    """
    Get a single paper by paper_id
    
    - **paper_id**: The paper's unique identifier (e.g., W1234567890, arxiv:1234.5678)
    """
    repository = PaperRepository(db)
    service = PaperService(repository)
    
    paper = await service.get_paper(paper_id)
    if not paper:
        raise NotFoundException(f"Paper {paper_id} not found")
    
    # Update last accessed timestamp
    await repository.update_last_accessed(paper_id)
    
    request_id = getattr(request.state, 'request_id', None)
    return success_response(data=paper, request_id=request_id)


@router.patch("/{paper_id}", response_model=ApiResponse[PaperDetail])
async def update_paper(
    request: Request,
    paper_id: str,
    update_data: PaperUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> ApiResponse[PaperDetail]:
    """
    Update a paper's metadata
    
    - **paper_id**: The paper's unique identifier
    - **update_data**: Fields to update (title, abstract, venue, etc.)
    """
    repository = PaperRepository(db)
    service = PaperService(repository)
    
    paper = await service.update_paper(paper_id, update_data)
    if not paper:
        raise NotFoundException(f"Paper {paper_id} not found")
    
    request_id = getattr(request.state, 'request_id', None)
    return success_response(data=paper, request_id=request_id)


@router.delete("/{paper_id}", response_model=ApiResponse[dict])
async def delete_paper(
    request: Request,
    paper_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> ApiResponse[dict]:
    """
    Delete a paper and all its chunks
    
    - **paper_id**: The paper's unique identifier
    """
    repository = PaperRepository(db)
    service = PaperService(repository)
    
    deleted = await service.delete_paper(paper_id)
    if not deleted:
        raise NotFoundException(f"Paper {paper_id} not found")
    
    request_id = getattr(request.state, 'request_id', None)
    return success_response(
        data={"message": f"Paper {paper_id} deleted successfully"},
        request_id=request_id
    )


@router.get("/{paper_id}/chunks", response_model=ApiResponse[list[ChunkResponse]])
async def get_paper_chunks(
    request: Request,
    paper_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> ApiResponse[list[ChunkResponse]]:
    """
    Get all chunks for a paper
    
    - **paper_id**: The paper's unique identifier
    """
    paper_repository = PaperRepository(db)
    paper_service = PaperService(paper_repository)
    
    # Verify paper exists
    paper = await paper_service.get_paper(paper_id)
    if not paper:
        raise NotFoundException(f"Paper {paper_id} not found")
    
    # Get chunks using ChunkService
    chunk_repository = ChunkRepository(db)
    chunk_service = ChunkService(chunk_repository)
    chunks = await chunk_service.get_paper_chunks(paper_id)
    
    request_id = getattr(request.state, 'request_id', None)
    return success_response(data=chunks, request_id=request_id)

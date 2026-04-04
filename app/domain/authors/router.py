"""
Router for Author management API
"""
from typing import Optional, Dict
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.database import get_db_session
from app.core.dependencies import get_container
from app.core.container import ServiceContainer
from .schemas import (
    AuthorResponse,
    AuthorListResponse,
    AuthorDetailResponse,
    AuthorDetailWithPapersResponse,
    QuartileBreakdown,
    CoAuthor,
    AuthorCollaborationListResponse,
    CitingAuthorsListResponse,
    ReferencedAuthorsListResponse,
    AuthorPublicationsListResponse,
    AuthorPaperSummary,
    CitingAuthor,
    ReferencedAuthor
)
from app.models.authors import DBAuthor
from app.extensions.logger import create_logger
logger = create_logger(__name__)

router = APIRouter()

# In-memory author enrichment task tracking (same lifecycle as task queue)
AUTHOR_ENRICHMENT_TASKS: Dict[str, str] = {}

@router.get("", response_model=AuthorListResponse)
async def list_authors(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    search: Optional[str] = None,
    verified_only: bool = False,
    db: AsyncSession = Depends(get_db_session)
):
    """List all authors with pagination and filters"""
    query = select(DBAuthor)
    
    if search:
        query = query.where(
            DBAuthor.name.ilike(f"%{search}%") | 
            DBAuthor.display_name.ilike(f"%{search}%")
        )
    
    if verified_only:
        query = query.where(DBAuthor.verified == True)
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query) or 0
    
    # Apply pagination
    query = query.offset((page - 1) * page_size).limit(page_size)
    query = query.order_by(DBAuthor.total_citations.desc().nulls_last())
    
    result = await db.execute(query)
    authors = result.scalars().all()
    
    return AuthorListResponse(
        total=total,
        page=page,
        page_size=page_size,
        authors=[AuthorResponse.model_validate(author) for author in authors]
    )


@router.get("/{author_id}", response_model=AuthorResponse)
async def get_author(
    author_id: str,
    container: ServiceContainer = Depends(get_container)
):
    """Get author by ID"""
    author = await container.author_repository.get_author(author_id)
    
    if not author:
        raise HTTPException(status_code=404, detail="Author not found")
    
    return AuthorResponse.model_validate(author)


@router.get("/{author_id}/details", response_model=AuthorDetailWithPapersResponse)
async def get_author_details(
    author_id: str,
    auto_enrich: bool = Query(
        default=True,
        description="Automatically enrich if not enriched"
    ),
    refresh_live_metrics: bool = Query(
        default=False,
        description="Refresh author metrics from Semantic Scholar before responding (slower)."
    ),
    container: ServiceContainer = Depends(get_container)
):
    """
    Get comprehensive author profile with papers, quartile breakdown, and co-authors.
    
    - Auto-enriches on first visit (triggers background job)
    - Cached for 30 days
    - Returns full publication history
    """
    from app.workers.task_queue import get_task_queue
    from app.workers.enrichment_worker import EnrichmentWorker
    
    # Get author details from service
    result = await container.author_service.get_author_profile(
        author_id=author_id,
        auto_enrich=auto_enrich,
        refresh_live_metrics=refresh_live_metrics,
    )
    
    author = result["author"]
    enrichment_status = result["enrichment_status"]

    # If there is an active tracked task for this author, surface live status.
    existing_task_id = AUTHOR_ENRICHMENT_TASKS.get(author_id)
    if existing_task_id:
        task_status = await get_task_queue().get_status(existing_task_id)
        if not task_status:
            AUTHOR_ENRICHMENT_TASKS.pop(author_id, None)
        else:
            status_value = task_status.get("status")
            if status_value in ("pending", "running"):
                enrichment_status = {
                    "status": "enriching",
                    "task_id": existing_task_id,
                    "message": "Author data is being updated in background."
                }
            elif status_value == "completed":
                AUTHOR_ENRICHMENT_TASKS.pop(author_id, None)
                enrichment_status = {
                    "status": "completed",
                    "task_id": existing_task_id,
                    "message": "Author enrichment completed."
                }
            elif status_value == "failed":
                AUTHOR_ENRICHMENT_TASKS.pop(author_id, None)
                enrichment_status = {
                    "status": "failed",
                    "task_id": existing_task_id,
                    "message": task_status.get("error") or "Author enrichment failed."
                }
    
    if enrichment_status and enrichment_status.get("status") == "needs_enrichment" and auto_enrich:
        task_queue = get_task_queue()
        task_id = await task_queue.submit(
            "author_enrichment",
            EnrichmentWorker.enrich_author_background,
            author_id=author_id,
            limit=100
        )
        AUTHOR_ENRICHMENT_TASKS[author_id] = task_id
        
        enrichment_status = {
            "status": "enriching",
            "task_id": task_id,
            "message": "Author data is being updated in background. Refresh in 30-60 seconds for updated data."
        }
        
        logger.info(f"Submitted background enrichment for author {author_id}, task {task_id}")
    
    # Build response
    quartile_breakdown = QuartileBreakdown(**result["quartile_breakdown"])
    co_authors = [CoAuthor(**ca) for ca in result["co_authors"]]
    
    author_dict = {
        **AuthorDetailResponse.model_validate(author).model_dump(),
        "papers": result["papers"],
        "quartile_breakdown": quartile_breakdown,
        "co_authors": co_authors,
        "counts_by_year": result.get("counts_by_year", {}),
        "papers_by_year": result["papers_by_year"],
        "is_enriched": result["is_enriched"],
        "enrichment_status": enrichment_status
    }
    
    return AuthorDetailWithPapersResponse(**author_dict)


@router.get("/{author_id}/collaborations", response_model=AuthorCollaborationListResponse)
async def get_author_collaborations(
    author_id: str,
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(10, ge=1, le=100, description="Items to return"),
    refresh_live_metrics: bool = Query(
        default=False,
        description="Refresh collaboration metrics from Semantic Scholar before responding (slower)."
    ),
    container: ServiceContainer = Depends(get_container)
):
    """
    Get authors who have collaborated with this author (co-authored papers).
    Ordered by number of collaborations (descending).
    """
    author = await container.author_repository.get_author(author_id)
    if not author:
        raise HTTPException(status_code=404, detail="Author not found")
    
    co_author_data, total = await container.author_service.get_co_authors_paginated(
        author_id=author_id,
        offset=offset,
        limit=limit,
        refresh_live_metrics=refresh_live_metrics,
    )
    co_authors = [CoAuthor(**ca) for ca in co_author_data]
    
    return AuthorCollaborationListResponse(
        total=total,
        offset=offset,
        limit=limit,
        co_authors=co_authors
    )


@router.get("/{author_id}/citing", response_model=CitingAuthorsListResponse)
async def get_authors_citing_author(
    author_id: str,
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(10, ge=1, le=100, description="Items to return"),
    refresh_live_metrics: bool = Query(
        default=False,
        description="Refresh citing author metrics from Semantic Scholar before responding (slower)."
    ),
    container: ServiceContainer = Depends(get_container)
):
    """
    Get authors who have cited this author's papers.
    Ordered by number of citations (descending).
    
    Note: Data is computed asynchronously after author enrichment.
    If not yet available, returns empty list.
    """
    author = await container.author_repository.get_author(author_id)
    if not author:
        raise HTTPException(status_code=404, detail="Author not found")
    
    citing_author_data, total = await container.author_service.get_citing_authors(
        author_id=author_id,
        offset=offset,
        limit=limit,
        refresh_live_metrics=refresh_live_metrics,
    )
    
    # If no data and author was recently enriched, trigger computation
    if total == 0 and author.last_paper_indexed_at:
        import asyncio
        asyncio.create_task(
            container.author_service.compute_author_relationships(author_id)
        )
    
    citing_authors = [CitingAuthor(**ca) for ca in citing_author_data]

    return CitingAuthorsListResponse(
        total=total,
        offset=offset,
        limit=limit,
        citing_authors=citing_authors
    )


@router.get("/{author_id}/referenced", response_model=ReferencedAuthorsListResponse)
async def get_authors_referenced_by_author(
    author_id: str,
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(10, ge=1, le=100, description="Items to return"),
    refresh_live_metrics: bool = Query(
        default=False,
        description="Refresh referenced author metrics from Semantic Scholar before responding (slower)."
    ),
    container: ServiceContainer = Depends(get_container)
):
    """
    Get authors that this author has referenced/cited in their papers.
    Ordered by number of references (descending).
    
    Note: Data is computed asynchronously after author enrichment.
    If not yet available, returns empty list.
    """
    author = await container.author_repository.get_author(author_id)
    if not author:
        raise HTTPException(status_code=404, detail="Author not found")
    
    referenced_author_data, total = await container.author_service.get_referenced_authors(
        author_id=author_id,
        offset=offset,
        limit=limit,
        refresh_live_metrics=refresh_live_metrics,
    )
    
    # If no data and author was recently enriched, trigger computation
    if total == 0 and author.last_paper_indexed_at:
        import asyncio
        asyncio.create_task(
            container.author_service.compute_author_relationships(author_id)
        )
    
    referenced_authors = [ReferencedAuthor(**ra) for ra in referenced_author_data]

    return ReferencedAuthorsListResponse(
        total=total,
        offset=offset,
        limit=limit,
        referenced_authors=referenced_authors
    )


@router.get("/{author_id}/publications", response_model=AuthorPublicationsListResponse)
async def get_author_publications(
    author_id: str,
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(10, ge=1, le=100, description="Items to return"),
    sort_by: str = Query("year", pattern="^(year|citation)$", description="Sort by: year or citation"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order: asc or desc"),
    refresh_live_metrics: bool = Query(
        default=False,
        description="Refresh publication author metrics from Semantic Scholar before responding (slower)."
    ),
    container: ServiceContainer = Depends(get_container),
):
    """Get paginated publications for an author with sorting."""
    author = await container.author_repository.get_author(author_id)
    if not author:
        raise HTTPException(status_code=404, detail="Author not found")

    papers, total = await container.author_service.get_author_publications_paginated(
        author_id=author_id,
        offset=offset,
        limit=limit,
        sort_by=sort_by,
        sort_order=sort_order,
        refresh_live_metrics=refresh_live_metrics,
    )

    items = [AuthorPaperSummary.model_validate(p) for p in papers]
    return AuthorPublicationsListResponse(
        total=total,
        offset=offset,
        limit=limit,
        items=items,
    )
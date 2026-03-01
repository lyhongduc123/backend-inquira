"""
Router for Author management API
"""
from typing import Optional
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.database import get_db_session
from app.core.dependencies import get_container
from app.core.container import ServiceContainer
from app.authors.schemas import (
    AuthorCreate,
    AuthorUpdate,
    AuthorResponse,
    AuthorListResponse,
    AuthorStatsResponse,
    AuthorDetailResponse,
    AuthorDetailWithPapersResponse,
    AuthorEnrichmentRequest,
    AuthorEnrichmentResponse,
    QuartileBreakdown,
    CoAuthor,
    AuthorCollaborationListResponse,
    CitingAuthorsListResponse,
    ReferencedAuthorsListResponse,
    CitingAuthor,
    ReferencedAuthor
)
from app.papers.schemas import PaperMetadata
from app.models.authors import DBAuthor
from app.extensions.logger import create_logger
logger = create_logger(__name__)

router = APIRouter()


@router.get("/stats", response_model=AuthorStatsResponse)
async def get_author_stats(db: AsyncSession = Depends(get_db_session)):
    """Get author statistics"""
    total = await db.scalar(select(func.count()).select_from(DBAuthor))
    verified = await db.scalar(
        select(func.count()).select_from(DBAuthor).where(DBAuthor.verified == True)
    )
    with_orcid = await db.scalar(
        select(func.count()).select_from(DBAuthor).where(DBAuthor.orcid.isnot(None))
    )
    with_retracted = await db.scalar(
        select(func.count()).select_from(DBAuthor).where(DBAuthor.has_retracted_papers == True)
    )
    avg_h_index = await db.scalar(
        select(func.avg(DBAuthor.h_index)).select_from(DBAuthor).where(DBAuthor.h_index.isnot(None))
    )
    avg_citations = await db.scalar(
        select(func.avg(DBAuthor.total_citations)).select_from(DBAuthor).where(DBAuthor.total_citations.isnot(None))
    )
    
    return AuthorStatsResponse(
        total_authors=total or 0,
        verified_authors=verified or 0,
        with_orcid=with_orcid or 0,
        with_retracted_papers=with_retracted or 0,
        average_h_index=float(avg_h_index) if avg_h_index else None,
        average_citations=float(avg_citations) if avg_citations else None
    )


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
    author = await container.author_repository.get_author_by_id(author_id)
    
    if not author:
        raise HTTPException(status_code=404, detail="Author not found")
    
    return AuthorResponse.model_validate(author)


@router.post("", response_model=AuthorResponse, status_code=201)
async def create_author(
    author_data: AuthorCreate,
    container: ServiceContainer = Depends(get_container)
):
    """Create a new author"""
    
    # Check if author already exists
    existing = await container.author_repository.get_author_by_id(author_data.author_id)
    if existing:
        raise HTTPException(status_code=409, detail="Author already exists")
    
    author = await container.author_repository.create_author(author_data.model_dump())
    return AuthorResponse.model_validate(author)


@router.patch("/{author_id}", response_model=AuthorResponse)
async def update_author(
    author_id: str,
    author_data: AuthorUpdate,
    container: ServiceContainer = Depends(get_container)
):
    """Update an existing author"""
    
    # Only update non-None fields
    update_data = {k: v for k, v in author_data.model_dump().items() if v is not None}
    
    author = await container.author_repository.update_author(author_id, update_data)
    if not author:
        raise HTTPException(status_code=404, detail="Author not found")
    
    return AuthorResponse.model_validate(author)


@router.delete("/{author_id}", status_code=204)
async def delete_author(
    author_id: str,
    container: ServiceContainer = Depends(get_container)
):
    """Delete an author"""
    author = await container.author_repository.get_author_by_id(author_id)
    
    if not author:
        raise HTTPException(status_code=404, detail="Author not found")
    
    await container.db_session.delete(author)
    await container.db_session.commit()


@router.get("/{author_id}/details", response_model=AuthorDetailWithPapersResponse)
async def get_author_details(
    author_id: str,
    auto_enrich: bool = Query(
        default=True,
        description="Automatically enrich if not enriched"
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
    
    author = await container.author_repository.get_author_by_id(author_id)
    
    if not author:
        raise HTTPException(status_code=404, detail="Author not found")
    
    # Check if enrichment is needed
    now = datetime.now(timezone.utc)
    last_indexed = author.last_paper_indexed_at
    needs_enrichment = (
        last_indexed is None or
        (now - last_indexed).total_seconds() > 30 * 24 * 3600  # type: ignore # 30 days in seconds
    )
    
    enrichment_status = None
    if needs_enrichment and auto_enrich:
        # Submit background task instead of blocking
        task_queue = get_task_queue()
        task_id = await task_queue.submit(
            "author_enrichment",
            EnrichmentWorker.enrich_author_background,
            author_id=author_id,
            limit=500
        )
        
        enrichment_status = {
            "status": "enriching",
            "task_id": task_id,
            "message": "Author data is being updated in background. Refresh in 30-60 seconds for updated data."
        }
        
        logger.info(f"Submitted background enrichment for author {author_id}, task {task_id}")
    
    # Get papers and related data (return what we have now)
    papers = await container.author_repository.get_author_papers_with_metadata(author_id)
    paper_metadata_list = [
        container.transformer_service.dbpaper_to_metadata(paper)
        for paper in papers
    ]

    quartile_dict = await container.author_repository.get_quartile_breakdown(author_id)
    quartile_breakdown = QuartileBreakdown(**quartile_dict)

    co_author_data = await container.author_repository.get_co_authors(author_id, limit=10)
    co_authors = [CoAuthor(**ca) for ca in co_author_data]
    
    papers_by_year = {}
    for paper in papers:
        if paper.publication_date:
            year = paper.publication_date.year
            papers_by_year[year] = papers_by_year.get(year, 0) + 1
    
    author_dict = {
        **AuthorDetailResponse.model_validate(author).model_dump(),
        "papers": paper_metadata_list,
        "quartile_breakdown": quartile_breakdown,
        "co_authors": co_authors,
        "papers_by_year": papers_by_year,
        "is_enriched": author.last_paper_indexed_at is not None,
        "enrichment_status": enrichment_status  # Add status to response
    }
    
    return AuthorDetailWithPapersResponse(**author_dict)


@router.get("/{author_id}/collaborations", response_model=AuthorCollaborationListResponse)
async def get_author_collaborations(
    author_id: str,
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of collaborators to return"),
    container: ServiceContainer = Depends(get_container)
):
    """
    Get authors who have collaborated with this author (co-authored papers).
    Ordered by number of collaborations (descending).
    """
    author = await container.author_repository.get_author_by_id(author_id)
    if not author:
        raise HTTPException(status_code=404, detail="Author not found")
    
    # Get total count first
    all_co_authors = await container.author_repository.get_co_authors(
        author_id, limit=10000, offset=0
    )
    total = len(all_co_authors)
    
    # Get paginated results
    co_author_data = await container.author_repository.get_co_authors(
        author_id, limit=limit, offset=offset
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
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of citing authors to return"),
    container: ServiceContainer = Depends(get_container)
):
    """
    Get authors who have cited this author's papers.
    Ordered by number of citations (descending).
    
    Note: Data is computed asynchronously after author enrichment.
    If not yet available, returns empty list.
    """
    author = await container.author_repository.get_author_by_id(author_id)
    if not author:
        raise HTTPException(status_code=404, detail="Author not found")
    
    citing_author_data, total = await container.author_repository.get_citing_authors(
        author_id, limit=limit, offset=offset
    )
    
    # If no data and author was recently enriched, trigger computation
    if total == 0 and author.last_paper_indexed_at:
        import asyncio
        asyncio.create_task(
            container.author_repository.compute_author_relationships(author_id)
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
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of referenced authors to return"),
    container: ServiceContainer = Depends(get_container)
):
    """
    Get authors that this author has referenced/cited in their papers.
    Ordered by number of references (descending).
    
    Note: Data is computed asynchronously after author enrichment.
    If not yet available, returns empty list.
    """
    author = await container.author_repository.get_author_by_id(author_id)
    if not author:
        raise HTTPException(status_code=404, detail="Author not found")
    
    referenced_author_data, total = await container.author_repository.get_referenced_authors(
        author_id, limit=limit, offset=offset
    )
    
    # If no data and author was recently enriched, trigger computation
    if total == 0 and author.last_paper_indexed_at:
        import asyncio
        asyncio.create_task(
            container.author_repository.compute_author_relationships(author_id)
        )
    
    referenced_authors = [ReferencedAuthor(**ra) for ra in referenced_author_data]
    
    return ReferencedAuthorsListResponse(
        total=total,
        offset=offset,
        limit=limit,
        referenced_authors=referenced_authors
    )
"""
Router for Author management API
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.database import get_db_session
from app.authors.repository import AuthorRepository
from app.authors.schemas import (
    AuthorCreate,
    AuthorUpdate,
    AuthorResponse,
    AuthorListResponse,
    AuthorStatsResponse
)
from app.models.authors import DBAuthor

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
    db: AsyncSession = Depends(get_db_session)
):
    """Get author by ID"""
    repository = AuthorRepository(db)
    author = await repository.get_author_by_id(author_id)
    
    if not author:
        raise HTTPException(status_code=404, detail="Author not found")
    
    return AuthorResponse.model_validate(author)


@router.post("", response_model=AuthorResponse, status_code=201)
async def create_author(
    author_data: AuthorCreate,
    db: AsyncSession = Depends(get_db_session)
):
    """Create a new author"""
    repository = AuthorRepository(db)
    
    # Check if author already exists
    existing = await repository.get_author_by_id(author_data.author_id)
    if existing:
        raise HTTPException(status_code=409, detail="Author already exists")
    
    author = await repository.create_author(author_data.model_dump())
    return AuthorResponse.model_validate(author)


@router.patch("/{author_id}", response_model=AuthorResponse)
async def update_author(
    author_id: str,
    author_data: AuthorUpdate,
    db: AsyncSession = Depends(get_db_session)
):
    """Update an existing author"""
    repository = AuthorRepository(db)
    
    # Only update non-None fields
    update_data = {k: v for k, v in author_data.model_dump().items() if v is not None}
    
    author = await repository.update_author(author_id, update_data)
    if not author:
        raise HTTPException(status_code=404, detail="Author not found")
    
    return AuthorResponse.model_validate(author)


@router.delete("/{author_id}", status_code=204)
async def delete_author(
    author_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Delete an author"""
    repository = AuthorRepository(db)
    author = await repository.get_author_by_id(author_id)
    
    if not author:
        raise HTTPException(status_code=404, detail="Author not found")
    
    await db.delete(author)
    await db.commit()
    
    return None

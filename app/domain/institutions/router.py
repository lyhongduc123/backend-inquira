"""
Router for Institution management API
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.db.database import get_db_session
from .repository import InstitutionRepository
from .schemas import (
    InstitutionResponse,
    InstitutionListResponse,
    InstitutionStatsResponse
)
from app.models.institutions import DBInstitution

router = APIRouter()


@router.get("/stats", response_model=InstitutionStatsResponse)
async def get_institution_stats(db: AsyncSession = Depends(get_db_session)):
    """Get institution statistics"""
    total = await db.scalar(select(func.count()).select_from(DBInstitution))
    
    # Get stats
    avg_works = await db.scalar(
        select(func.avg(DBInstitution.total_papers))
        .select_from(DBInstitution)
        .where(DBInstitution.total_papers.isnot(None))
    )
    avg_citations = await db.scalar(
        select(func.avg(DBInstitution.total_citations))
        .select_from(DBInstitution)
        .where(DBInstitution.total_citations.isnot(None))
    )
    
    # Group by country
    country_result = await db.execute(
        select(DBInstitution.country_code, func.count())
        .select_from(DBInstitution)
        .where(DBInstitution.country_code.isnot(None))
        .group_by(DBInstitution.country_code)
    )
    by_country = {row[0]: row[1] for row in country_result.all() if row[0]}
    
    # Group by type
    type_result = await db.execute(
        select(DBInstitution.type, func.count())
        .select_from(DBInstitution)
        .where(DBInstitution.type.isnot(None))
        .group_by(DBInstitution.type)
    )
    by_type = {row[0]: row[1] for row in type_result.all() if row[0]}
    
    return InstitutionStatsResponse(
        total_institutions=total or 0,
        by_country=by_country,
        by_type=by_type,
        average_works_count=float(avg_works) if avg_works else None,
        average_citations=float(avg_citations) if avg_citations else None
    )


@router.get("", response_model=InstitutionListResponse)
async def list_institutions(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    search: Optional[str] = None,
    country_code: Optional[str] = None,
    institution_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db_session)
):
    """List all institutions with pagination and filters"""
    query = select(DBInstitution)
    
    if search:
        query = query.where(
            DBInstitution.name.ilike(f"%{search}%") | 
            DBInstitution.display_name.ilike(f"%{search}%")
        )
    
    if country_code:
        query = query.where(DBInstitution.country_code == country_code)
    
    if institution_type:
        query = query.where(DBInstitution.type == institution_type)
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query) or 0
    
    # Apply pagination
    query = query.offset((page - 1) * page_size).limit(page_size)
    query = query.order_by(DBInstitution.total_citations.desc().nulls_last())
    
    result = await db.execute(query)
    institutions = result.scalars().all()
    
    return InstitutionListResponse(
        total=total,
        page=page,
        page_size=page_size,
        institutions=[InstitutionResponse.model_validate(inst) for inst in institutions]
    )


@router.get("/{institution_id}", response_model=InstitutionResponse)
async def get_institution(
    institution_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Get institution by ID"""
    repository = InstitutionRepository(db)
    institution = await repository.get_institution_by_id(institution_id)
    
    if not institution:
        raise HTTPException(status_code=404, detail="Institution not found")
    
    return InstitutionResponse.model_validate(institution)

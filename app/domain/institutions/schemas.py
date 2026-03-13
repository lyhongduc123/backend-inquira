"""
Pydantic schemas for Institution API
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from app.core.model import CamelModel


class InstitutionBase(CamelModel):
    """Base institution schema"""
    name: str
    display_name: Optional[str] = None
    ror_id: Optional[str] = None
    country_code: Optional[str] = None
    institution_type: Optional[str] = None


class InstitutionCreate(InstitutionBase):
    """Schema for creating a new institution"""
    institution_id: str
    external_ids: Optional[Dict[str, Any]] = None
    homepage_url: Optional[str] = None
    wikipedia_url: Optional[str] = None


class InstitutionUpdate(CamelModel):
    """Schema for updating an institution"""
    name: Optional[str] = None
    display_name: Optional[str] = None
    ror_id: Optional[str] = None
    country_code: Optional[str] = None
    institution_type: Optional[str] = None
    homepage_url: Optional[str] = None
    wikipedia_url: Optional[str] = None


class InstitutionResponse(CamelModel):
    """Detailed institution response"""
    id: int
    institution_id: str
    name: str
    display_name: Optional[str] = None
    ror_id: Optional[str] = None
    external_ids: Optional[Dict[str, Any]] = None
    country_code: Optional[str] = None
    institution_type: Optional[str] = None
    homepage_url: Optional[str] = None
    wikipedia_url: Optional[str] = None
    works_count: Optional[int] = None
    cited_by_count: Optional[int] = None
    summary_stats: Optional[Dict[str, Any]] = None
    geo: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class InstitutionListResponse(CamelModel):
    """List response for institutions"""
    total: int
    page: int
    page_size: int
    institutions: List[InstitutionResponse]


class InstitutionStatsResponse(CamelModel):
    """Institution statistics"""
    total_institutions: int
    by_country: Dict[str, int]
    by_type: Dict[str, int]
    average_works_count: Optional[float] = None
    average_citations: Optional[float] = None

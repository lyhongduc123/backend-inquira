"""
Pydantic schemas for Author API
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class AuthorBase(BaseModel):
    """Base author schema"""
    name: str
    display_name: Optional[str] = None
    orcid: Optional[str] = None


class AuthorCreate(AuthorBase):
    """Schema for creating a new author"""
    author_id: str
    openalex_id: Optional[str] = None
    h_index: Optional[int] = None
    i10_index: Optional[int] = None
    total_citations: Optional[int] = None
    total_papers: Optional[int] = None
    external_ids: Optional[Dict[str, Any]] = None
    verified: bool = False

class Author(AuthorCreate):
    """Author mirroring database model"""
    id: int
    retracted_papers_count: Optional[int] = None
    first_publication_year: Optional[int] = None
    last_known_institution_id: Optional[str] = None
    reputation_score: Optional[float] = None
    field_weighted_citation_impact: Optional[float] = None
    collaboration_diversity_score: Optional[float] = None
    is_corresponding_author_frequently: Optional[bool] = None
    average_author_position: Optional[float] = None
    has_retracted_papers: bool = False
    self_citation_rate: Optional[float] = None
    homepage_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_paper_indexed_at: Optional[datetime] = None

    class Config:
        from_attributes = True
    

class AuthorUpdate(BaseModel):
    """Schema for updating an author"""
    name: Optional[str] = None
    display_name: Optional[str] = None
    orcid: Optional[str] = None
    h_index: Optional[int] = None
    i10_index: Optional[int] = None
    total_citations: Optional[int] = None
    total_papers: Optional[int] = None
    reputation_score: Optional[float] = None


class AuthorResponse(BaseModel):
    """Detailed author response"""
    id: int
    author_id: str
    openalex_id: Optional[str] = None
    name: str
    display_name: Optional[str] = None
    orcid: Optional[str] = None
    external_ids: Optional[Dict[str, Any]] = None
    h_index: Optional[int] = None
    i10_index: Optional[int] = None
    total_citations: Optional[int] = None
    total_papers: Optional[int] = None
    verified: bool = False
    retracted_papers_count: Optional[int] = None
    first_publication_year: Optional[int] = None
    last_known_institution_id: Optional[str] = None
    reputation_score: Optional[float] = None
    field_weighted_citation_impact: Optional[float] = None
    collaboration_diversity_score: Optional[float] = None
    is_corresponding_author_frequently: Optional[bool] = None
    average_author_position: Optional[float] = None
    has_retracted_papers: bool = False
    self_citation_rate: Optional[float] = None
    homepage_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_paper_indexed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class AuthorListResponse(BaseModel):
    """List response for authors"""
    total: int
    page: int
    page_size: int
    authors: List[AuthorResponse]


class AuthorStatsResponse(BaseModel):
    """Author statistics"""
    total_authors: int
    verified_authors: int
    with_orcid: int
    with_retracted_papers: int
    average_h_index: Optional[float] = None
    average_citations: Optional[float] = None

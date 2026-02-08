"""
Paper schemas for API requests/responses
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pgvector import Vector
from pydantic import BaseModel, Field
from app.authors.schemas import Author



class PaperBase(BaseModel):
    """Base paper schema"""

    title: str
    abstract: Optional[str] = None
    authors: List[Author] = []
    publication_date: Optional[datetime] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    is_open_access: bool = False
    source: str = "manual"


class PaperCreate(PaperBase):
    """Schema for creating a new paper"""

    paper_id: str
    external_ids: Optional[Dict[str, Any]] = None


class PaperUpdate(BaseModel):
    """Schema for updating a paper"""

    title: Optional[str] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    is_open_access: Optional[bool] = None


class PaperDetail(BaseModel):
    """Detailed paper response"""

    id: int
    paper_id: str
    title: str
    authors: List[Dict[str, Any]]
    abstract: str
    publication_date: Optional[datetime] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    is_open_access: bool
    source: str
    external_ids: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    relevance_score: Optional[float] = None
    citation_count: int
    reference_count: int
    is_processed: bool
    processing_status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PaperSummary(BaseModel):
    """Summary paper response for lists"""

    id: int
    paper_id: str
    title: str
    authors: List[Dict[str, Any]]
    publication_date: Optional[datetime] = None
    venue: Optional[str] = None
    citation_count: int
    is_processed: bool
    created_at: datetime

    class Config:
        from_attributes = True

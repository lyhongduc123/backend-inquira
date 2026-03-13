"""
Bookmark schemas for API
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from app.core.model import CamelModel


class BookmarkCreate(CamelModel):
    """Request to create a bookmark"""
    paper_id: str = Field(..., description="Paper ID to bookmark")
    notes: Optional[str] = Field(None, max_length=5000, description="Optional notes about the paper")


class BookmarkUpdate(CamelModel):
    """Request to update a bookmark"""
    notes: Optional[str] = Field(None, max_length=5000, description="Optional notes about the paper")


class BookmarkResponse(CamelModel):
    """Bookmark response without user_id (user is already authenticated)"""
    id: int
    paper_id: str
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime


class BookmarkWithPaperResponse(CamelModel):
    """Bookmark with paper details"""
    id: int
    paper_id: str
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    paper: Optional['PaperMetadata'] = Field(None, description="Paper metadata")


class BookmarkListResponse(CamelModel):
    """Paginated list of bookmarks"""
    items: List[BookmarkWithPaperResponse]
    total: int
    skip: int
    limit: int


# Import for type checking
from app.domain.papers.schemas import PaperMetadata
BookmarkWithPaperResponse.model_rebuild()

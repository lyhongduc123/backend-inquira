"""
Bookmark schemas for API
"""
from pydantic import BaseModel, Field
from typing import Optional
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
    """Bookmark response"""
    id: int
    user_id: int
    paper_id: str
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime


class BookmarkWithPaperResponse(CamelModel):
    """Bookmark with paper details"""
    id: int
    user_id: int
    paper_id: str
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    paper: Optional[dict] = Field(None, description="Paper metadata")

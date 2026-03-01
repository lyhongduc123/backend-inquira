"""
Chunk schemas for API requests/responses
"""

from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from app.core.model import CamelModel


class ChunkBase(CamelModel):
    """Base chunk schema"""
    chunk_id: str
    paper_id: str
    text: str
    token_count: int
    chunk_index: int
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    label: Optional[str] = None
    level: Optional[int] = None

    model_config = {
        "from_attributes": True
    }


class Chunk(ChunkBase):
    """Full chunk schema with database fields"""
    id: int
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    docling_metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[list[float]] = None
    created_at: datetime


class ChunkRetrieved(Chunk):
    """Chunk with relevance score for retrieval results"""
    relevance_score: float


class ChunkResponse(ChunkBase):
    """Paper chunk response for API"""
    pass


class ChunkCreate(BaseModel):
    """Schema for creating a new chunk"""
    chunk_id: str
    paper_id: str
    text: str
    token_count: int
    chunk_index: int
    embedding: list[float]
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    label: Optional[str] = None
    level: Optional[int] = None

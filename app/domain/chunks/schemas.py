"""
Chunk schemas for API requests/responses
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel
from app.domain.chunks.types import Chunk, ChunkBase, ChunkRetrieved


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
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    docling_metadata: Optional[Dict[str, Any]] = None
__all__ = [
    "Chunk",
    "ChunkBase",
    "ChunkCreate",
    "ChunkResponse",
    "ChunkRetrieved",
]

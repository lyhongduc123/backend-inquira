"""Internal chunk data contracts used by search and RAG workflows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from app.core.model import CamelModel


class ChunkBase(CamelModel):
    """Base chunk data shared by API and internal contracts."""

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
    """Full chunk data with database fields."""

    id: int
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    docling_metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[list[float]] = None
    created_at: datetime


class ChunkRetrieved(Chunk):
    """Retrieved chunk with a relevance score."""

    relevance_score: float

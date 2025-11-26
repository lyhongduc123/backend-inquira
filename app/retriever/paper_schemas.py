from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class Author(BaseModel):
    """Author information"""
    name: str
    author_id: Optional[str] = None


class Paper(BaseModel):
    """Paper metadata from Semantic Scholar"""
    paper_id: str = Field(..., description="Internal paper ID (e.g., P12345)")
    title: str
    authors: List[Author] = []
    abstract: Optional[str] = None
    publication_date: Optional[datetime] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    is_open_access: bool = False
    open_access_pdf: Optional[Dict[str, Any]] = None  # {"url": str, "status": str, "license": str}
    
    source: str = "semantic_scholar"
    external_id: Optional[str] = None
    
    relevance_score: Optional[float] = None
    citation_count: int = 0
    influential_citation_count: Optional[int] = 0
    reference_count: int = 0
    
    is_processed: bool = False
    processing_status: str = "pending"


class PaperChunk(BaseModel):
    """Chunked paper text with embedding"""
    chunk_id: str = Field(..., description="Chunk ID (e.g., P12345::C7)")
    paper_id: str
    text: str
    token_count: int
    chunk_index: int
    section_title: Optional[str] = None
    page_number: Optional[int] = None


class PaperSearchRequest(BaseModel):
    """Request for paper search"""
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    fields: Optional[List[str]] = None


class PaperSearchResponse(BaseModel):
    """Response from paper search"""
    papers: List[Paper]
    total: int
    offset: int = 0


class Citation(BaseModel):
    """Citation reference"""
    chunk_id: str
    paper_id: str
    text: str
    confidence: Optional[float] = None

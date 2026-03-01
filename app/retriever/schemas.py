"""
Retriever schemas - Using core DTOs for consistency
Re-exports for backward compatibility during transition
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

# Import core DTOs as source of truth
from app.core.dtos.author import AuthorDTO as Author
from app.core.dtos.paper import PaperDTO as Paper


# API-specific response schema (stays here as it's retriever-specific)
class PaperResponse(BaseModel):
    """
    External DTO for API responses to frontend.
    Can include computed fields, enriched data, and formatted information.
    """
    paper_id: str
    title: str
    authors: List[Author] = []
    abstract: Optional[str] = None
    publication_date: Optional[datetime] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    is_open_access: bool = False
    source: str
    openalex_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    doi: Optional[str] = None
    citation_count: Optional[int] = 0
    influential_citation_count: Optional[int] = 0
    fwci: Optional[float] = None
    topics: Optional[List[Dict[str, Any]]] = None
    keywords: Optional[List[Dict[str, Any]]] = None
    concepts: Optional[List[Dict[str, Any]]] = None
    
    # Trust scores
    author_trust_score: Optional[float] = None
    institutional_trust_score: Optional[float] = None
    is_retracted: bool = False
    language: Optional[str] = None
    is_processed: bool = False
    biblio: Optional[Dict[str, Any]] = None
    primary_location: Optional[Dict[str, Any]] = None
    relevance_score: Optional[float] = None
    ranking_score: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True

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

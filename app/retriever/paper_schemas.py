from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class Author(BaseModel):
    """Author information"""
    name: str
    author_id: Optional[str] = None
    citation_count: Optional[int] = None
    h_index: Optional[int] = None
    # OpenAlex-specific fields
    orcid: Optional[str] = None
    institutions: Optional[List[Dict[str, Any]]] = None
    affiliations: Optional[List[Dict[str, Any]]] = None


class Paper(BaseModel):
    """Paper metadata compatible with normalized provider results and database model"""
    # Core identifiers and metadata
    paper_id: str = Field(..., description="Unique paper ID (internal or from provider)")
    title: str
    abstract: Optional[str] = None
    authors: List[Author] = []
    publication_date: Optional[datetime] = None
    venue: Optional[str] = None
    
    # URLs
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    is_open_access: bool = False
    open_access_pdf: Optional[Dict[str, Any]] = None  # {"url": str, "status": str, "license": str}

    # Source and identifiers
    source: str
    external_ids: Optional[Dict[str, Any]] = None

    # Summary and embeddings (populated during processing)
    summary: Optional[str] = None
    summary_embedding: Optional[List[float]] = None

    # Relevance and citation metrics
    relevance_score: Optional[float] = None
    citation_count: Optional[int] = 0
    influential_citation_count: Optional[int] = 0
    reference_count: Optional[int] = 0

    # Rich metadata (OpenAlex specific)
    topics: Optional[List[Dict[str, Any]]] = None  # Research topics with scores
    keywords: Optional[List[Dict[str, Any]]] = None  # Keywords with scores
    concepts: Optional[List[Dict[str, Any]]] = None  # Concepts with scores and hierarchy levels
    mesh_terms: Optional[List[Dict[str, Any]]] = None  # MeSH terms for biomedical papers

    # Citation quality metrics (OpenAlex)
    citation_percentile: Optional[Dict[str, Any]] = None  # Percentile rankings
    fwci: Optional[float] = None  # Field-weighted citation impact

    # Paper quality indicators
    is_retracted: bool = False
    language: Optional[str] = None  # ISO language code

    # Bibliographic information
    biblio: Optional[Dict[str, Any]] = None  # Volume, issue, pages
    primary_location: Optional[Dict[str, Any]] = None  # Primary publication venue details
    locations: Optional[List[Dict[str, Any]]] = None  # All publication locations

    # Author collaboration metadata
    corresponding_author_ids: Optional[List[str]] = None  # IDs of corresponding authors
    institutions_distinct_count: Optional[int] = None  # Number of unique institutions
    countries_distinct_count: Optional[int] = None  # Number of unique countries

    # Processing metadata
    is_processed: bool = False
    processing_status: str = "pending"
    processing_error: Optional[str] = None

    # External IDs for API lookups
    openalex_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None
    publication_year: Optional[int] = None

    # Additional OpenAlex fields for ranking
    authorships: Optional[List[Dict[str, Any]]] = None
    
    # Flexible data storage for full API responses
    openalex_data: Optional[Dict[str, Any]] = None
    semantic_data: Optional[Dict[str, Any]] = None
    
    # Ranking score (attached during ranking)
    ranking_score: Optional[Dict[str, Any]] = None
    
    # Database ID (from DBPaper.id)
    id: Optional[int] = None

    class Config:
        from_attributes = True
    


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

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class Author(BaseModel):
    """Author information"""
    name: str
    author_id: Optional[str] = None
    citation_count: Optional[int] = None
    h_index: Optional[int] = None
    paper_count: Optional[int] = None
    
    # OpenAlex-specific fields
    orcid: Optional[str] = None
    institutions: Optional[List[Dict[str, Any]]] = None
    affiliations: Optional[List[Dict[str, Any]]] = None


class Paper(BaseModel):
    """
    Internal DTO aligned with DBPaper model.
    Used for processing and database operations.
    
    NOTE: This schema matches DBPaper columns exactly.
    Complex nested data (biblio, locations, authorships) should be stored
    in openalex_data or semantic_data JSONB fields.
    """
    # Core identifiers and metadata (matches DBPaper exactly)
    paper_id: str = Field(..., description="Unique paper ID (OpenAlex ID, DOI, or source-specific)")
    title: str
    authors: List[Author] = []  # Simplified for JSONB storage
    abstract: Optional[str] = None
    publication_date: Optional[datetime] = None
    venue: Optional[str] = None
    issn: Optional[str] = None 
    issn_l: Optional[str] = None
    
    # URLs
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    is_open_access: bool = False
    open_access_pdf: Optional[Dict[str, Any]] = None

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
    topics: Optional[List[Dict[str, Any]]] = None
    keywords: Optional[List[Dict[str, Any]]] = None
    concepts: Optional[List[Dict[str, Any]]] = None
    mesh_terms: Optional[List[Dict[str, Any]]] = None

    # Citation quality metrics
    citation_percentile: Optional[Dict[str, Any]] = None
    fwci: Optional[float] = None

    # Trust scores (computed)
    author_trust_score: Optional[float] = None
    institutional_trust_score: Optional[float] = None
    network_diversity_score: Optional[float] = None
    
    # Journal relationship
    journal_id: Optional[int] = None

    # Paper quality indicators
    is_retracted: bool = False
    language: Optional[str] = None

    # Author collaboration metadata
    corresponding_author_ids: Optional[List[str]] = None
    institutions_distinct_count: Optional[int] = None
    countries_distinct_count: Optional[int] = None

    # Processing metadata
    is_processed: bool = False
    processing_status: str = "pending"
    processing_error: Optional[str] = None

    # Timestamps (from DB)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_accessed_at: Optional[datetime] = None
    
    # Database ID (from DBPaper.id)
    id: Optional[int] = None

    class Config:
        from_attributes = True
        extra = "ignore"
        
class PaperPreprocess(Paper):
    """Paper data for preprocessing before DB insertion"""
    has_content: Optional[Dict[str, Any]] = Field(None, exclude=True)
    authorships: Optional[List[Dict[str, Any]]] = Field(None, exclude=True)
    semantic_authors: Optional[List[Dict[str, Any]]] = Field(None, exclude=True)


class PaperResponse(BaseModel):
    """
    External DTO for API responses to frontend.
    Can include computed fields, enriched data, and formatted information.
    """
    # Core identifiers
    paper_id: str
    title: str
    authors: List[Author] = []
    abstract: Optional[str] = None
    publication_date: Optional[datetime] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    
    # URLs
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    is_open_access: bool = False
    
    # Source
    source: str
    
    # External IDs for links
    openalex_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    doi: Optional[str] = None
    
    # Citation metrics
    citation_count: Optional[int] = 0
    influential_citation_count: Optional[int] = 0
    fwci: Optional[float] = None
    
    # Rich metadata
    topics: Optional[List[Dict[str, Any]]] = None
    keywords: Optional[List[Dict[str, Any]]] = None
    concepts: Optional[List[Dict[str, Any]]] = None
    
    # Trust scores
    author_trust_score: Optional[float] = None
    institutional_trust_score: Optional[float] = None
    
    # Quality indicators
    is_retracted: bool = False
    language: Optional[str] = None
    
    # Processing status
    is_processed: bool = False
    
    # Bibliographic info (extracted from openalex_data if needed)
    biblio: Optional[Dict[str, Any]] = None
    primary_location: Optional[Dict[str, Any]] = None
    
    # Computed/enriched fields
    relevance_score: Optional[float] = None
    ranking_score: Optional[Dict[str, Any]] = None
    
    # Database ID
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

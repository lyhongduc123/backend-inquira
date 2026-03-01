"""
Paper API schemas - Request/Response DTOs for REST API
Separation: API layer schemas, not for internal data transfer
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from app.core.model import CamelModel
from app.core.dtos import AuthorDTO
from app.authors.schemas import AuthorMetadata


class PaperUpdateRequest(CamelModel):
    """API request for updating a paper"""
    title: Optional[str] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    is_open_access: Optional[bool] = None

class SJRMetadata(CamelModel):
    """SJR journal metadata"""
    title: Optional[str] = None
    sjr_score: Optional[float] = None
    quartile: Optional[str] = None
    h_index: Optional[int] = None
    data_year: Optional[int] = None
    
    class Config:
        from_attributes = True
        ignore_extra = True

class PaperMetadata(CamelModel):
    """
    Lightweight paper metadata for frontend API responses.
    
    Used for:
    - Streaming paper citations during RAG (chat responses)
    - Paper snapshots in conversation messages
    - Citation/reference lists
    - Any context requiring minimal paper info without full details
    
    This is the primary paper format sent to frontend clients.
    For full paper details, use PaperDetailResponse.
    For ranked papers with scores, use RankedPaper (processor/schemas.py).
    """
    paper_id: str
    title: str
    abstract: Optional[str]
    authors: List[AuthorMetadata] = []
    year: Optional[int] = None
    publication_date: Optional[datetime]
    venue: Optional[str]
    journal: Optional[Any] = None
    url: Optional[str]
    pdf_url: Optional[str]
    citation_count: int
    influential_citation_count: Optional[int]
    reference_count: Optional[int]
    citation_styles: Optional[Dict[str, str]] = None
    author_trust_score: Optional[float]
    institutional_trust_score: Optional[float]
    fwci: Optional[float]
    is_open_access: bool
    is_retracted: bool
    topics: Optional[List[Dict[str, Any]]]
    keywords: Optional[List[Dict[str, Any]]]
    
    
    relevance_score: Optional[float] = None
    ranking_scores: Optional[Dict[str, float]] = None
    sjr_data: Optional[SJRMetadata] = None
    
    class Config:
        from_attributes = True
        ignore_extra = True


class PaperDetailResponse(CamelModel):
    """
    Full paper details for frontend users.
    Includes enriched data (authors, institutions, journal, citations) without computed fields.
    """
    # Database ID
    id: int
    
    # Core identifiers
    paper_id: str
    title: str
    abstract: str
    authors: List[AuthorDTO] = []
    journal: Optional[Dict[str, Any]] = None
    publication_date: Optional[datetime] = None
    venue: Optional[str] = None
    issn: Optional[List[str]] = None
    issn_l: Optional[str] = None
    
    # URLs and access
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    is_open_access: bool = False
    open_access_pdf: Optional[Dict[str, Any]] = None
    
    # Source tracking
    source: str
    external_ids: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    
    # Citation metrics
    citation_count: int = 0
    influential_citation_count: int = 0
    reference_count: int = 0
    citation_styles: Optional[Dict[str, str]] = None
    
    # Rich metadata
    topics: Optional[List[Dict[str, Any]]] = None
    keywords: Optional[List[Dict[str, Any]]] = None
    concepts: Optional[List[Dict[str, Any]]] = None
    mesh_terms: Optional[List[Dict[str, Any]]] = None
    
    # Citation quality metrics
    citation_percentile: Optional[Dict[str, Any]] = None
    fwci: Optional[float] = None
    
    # Trust scores
    author_trust_score: Optional[float] = None
    institutional_trust_score: Optional[float] = None
    network_diversity_score: Optional[float] = None
    
    # Quality indicators
    is_retracted: bool = False
    language: Optional[str] = None
    
    # Collaboration metadata
    corresponding_author_ids: Optional[List[str]] = None
    institutions_distinct_count: Optional[int] = None
    countries_distinct_count: Optional[int] = None
    
    # Processing status
    is_processed: bool
    processing_status: str
    processing_error: Optional[str] = None
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    last_accessed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ============= Citation & Reference Schemas =============

class CitingPaperData(CamelModel):
    """Nested paper data from Semantic Scholar citations API"""
    paper_id: str = Field(alias="paperId")
    corpus_id: Optional[int] = Field(None, alias="corpusId")
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[List[Dict[str, Any]]] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    citation_count: Optional[int] = Field(None, alias="citationCount")
    
    class Config:
        populate_by_name = True
        from_attributes = True


class CitingPaper(CamelModel):
    """A paper that cites the target paper (S2 API wrapper format)"""
    citing_paper: CitingPaperData = Field(alias="citingPaper")
    is_influential: Optional[bool] = Field(None, alias="isInfluential")
    contexts: Optional[List[str]] = None
    intents: Optional[List[str]] = None
    
    class Config:
        populate_by_name = True
        from_attributes = True


class ReferencedPaperData(CamelModel):
    """Nested paper data from Semantic Scholar references API"""
    paper_id: str = Field(alias="paperId")
    corpus_id: Optional[int] = Field(None, alias="corpusId")
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[List[Dict[str, Any]]] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    citation_count: Optional[int] = Field(None, alias="citationCount")
    
    class Config:
        populate_by_name = True
        from_attributes = True


class ReferencedPaper(CamelModel):
    """A paper referenced by the target paper (S2 API wrapper format)"""
    cited_paper: ReferencedPaperData = Field(alias="citedPaper")
    is_influential: Optional[bool] = Field(None, alias="isInfluential")
    contexts: Optional[List[str]] = None
    intents: Optional[List[str]] = None
    
    class Config:
        populate_by_name = True
        from_attributes = True


class PaginatedCitationsResponse(CamelModel):
    """Paginated response for papers citing the target paper"""
    offset: int
    next: Optional[int] = None
    total: Optional[int] = None
    data: List[CitingPaper]


class PaginatedReferencesResponse(CamelModel):
    """Paginated response for papers referenced by the target paper"""
    offset: int
    next: Optional[int] = None
    total: Optional[int] = None
    data: List[ReferencedPaper]


# ============= Backward Compatibility Aliases =============
# To minimize disruption during refactoring
PaperUpdate = PaperUpdateRequest
PaperDetail = PaperDetailResponse  

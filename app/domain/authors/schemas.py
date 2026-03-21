"""
Pydantic schemas for Author API
"""
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime
from pydantic import BaseModel, Field
from app.core.model import CamelModel
from app.domain.common.schemas import SJRMetadata


class AuthorBase(CamelModel):
    """Base author schema"""
    name: str
    display_name: Optional[str] = None
    orcid: Optional[str] = None


class AuthorCreate(AuthorBase):
    """Schema for creating a new author"""
    author_id: str
    openalex_id: Optional[str] = None
    h_index: Optional[int] = None
    i10_index: Optional[int] = None
    g_index: Optional[int] = None
    total_citations: Optional[int] = None
    total_papers: Optional[int] = None
    retracted_papers_count: Optional[int] = None
    has_retracted_papers: Optional[bool] = None
    external_ids: Optional[Dict[str, Any]] = None
    verified: bool = False
    
class AuthorMetadata(AuthorBase):
    """Lightweight author metadata schema"""
    author_id: str
    h_index: Optional[int] = None
    citation_count: Optional[int] = None
    paper_count: Optional[int] = None
    verified: bool = False

class Author(AuthorCreate):
    """Author mirroring database model"""
    id: int
    first_publication_year: Optional[int] = None
    last_known_institution_id: Optional[str] = None
    reputation_score: Optional[float] = None
    field_weighted_citation_impact: Optional[float] = None
    is_corresponding_author_frequently: Optional[bool] = None
    average_author_position: Optional[float] = None
    self_citation_rate: Optional[float] = None
    homepage_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_paper_indexed_at: Optional[datetime] = None

    class Config:
        from_attributes = True
    

class AuthorUpdate(CamelModel):
    """Schema for updating an author"""
    name: Optional[str] = None
    display_name: Optional[str] = None
    orcid: Optional[str] = None
    h_index: Optional[int] = None
    i10_index: Optional[int] = None
    g_index: Optional[int] = None
    total_citations: Optional[int] = None
    total_papers: Optional[int] = None
    retracted_papers_count: Optional[int] = None
    has_retracted_papers: Optional[bool] = None
    reputation_score: Optional[float] = None


class AuthorResponse(CamelModel):
    """Detailed author response"""
    id: int
    author_id: str
    openalex_id: Optional[str] = None
    name: str
    display_name: Optional[str] = None
    orcid: Optional[str] = None
    external_ids: Optional[Dict[str, Any]] = None
    h_index: Optional[int] = None
    i10_index: Optional[int] = None
    g_index: Optional[int] = None
    total_citations: Optional[int] = None
    total_papers: Optional[int] = None
    retracted_papers_count: Optional[int] = None
    has_retracted_papers: Optional[bool] = None
    verified: bool = False
    first_publication_year: Optional[int] = None
    last_known_institution_id: Optional[str] = None
    reputation_score: Optional[float] = None
    field_weighted_citation_impact: Optional[float] = None
    is_corresponding_author_frequently: Optional[bool] = None
    average_author_position: Optional[float] = None
    self_citation_rate: Optional[float] = None
    is_processed: bool = False
    is_conflict: bool = False
    homepage_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_paper_indexed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class AuthorListResponse(CamelModel):
    """List response for authors"""
    total: int
    page: int
    page_size: int
    authors: List[AuthorResponse]


class AuthorStatsResponse(CamelModel):
    """Author statistics"""
    total_authors: int
    verified_authors: int
    with_orcid: int
    with_retracted_papers: int
    average_h_index: Optional[float] = None
    average_citations: Optional[float] = None


class AuthorPaperSummary(CamelModel):
    """Lightweight paper summary for author detail page"""
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
    
    class Config:
        from_attributes = True
        extra = "ignore"
        
        


class QuartileBreakdown(CamelModel):
    """Paper count by journal quartile"""
    q1: int = 0
    q2: int = 0
    q3: int = 0
    q4: int = 0
    unknown: int = 0


class CoAuthor(CamelModel):
    """Co-author information"""
    author_id: str
    name: str
    h_index: Optional[int] = None
    total_citations: Optional[int] = None
    total_papers: Optional[int] = None
    is_enriched: bool = False
    collaboration_count: int


class CitingAuthor(CamelModel):
    """Author who has cited this author"""
    author_id: str
    name: str
    h_index: Optional[int] = None
    total_citations: Optional[int] = None
    total_papers: Optional[int] = None
    is_enriched: bool = False
    citation_count: int


class ReferencedAuthor(CamelModel):
    """Author that this author has referenced"""
    author_id: str
    name: str
    h_index: Optional[int] = None
    total_citations: Optional[int] = None
    total_papers: Optional[int] = None
    is_enriched: bool = False
    reference_count: int


class AuthorCollaborationListResponse(CamelModel):
    """List response for author collaborations"""
    total: int
    offset: int
    limit: int
    co_authors: List[CoAuthor]


class CitingAuthorsListResponse(CamelModel):
    """List response for authors citing this author"""
    total: int
    offset: int
    limit: int
    citing_authors: List[CitingAuthor]


class ReferencedAuthorsListResponse(CamelModel):
    """List response for authors referenced by this author"""
    total: int
    offset: int
    limit: int
    referenced_authors: List[ReferencedAuthor]


class AuthorPublicationsListResponse(CamelModel):
    """Paginated publications for an author."""
    total: int
    offset: int
    limit: int
    items: List[AuthorPaperSummary]


class AuthorDetailResponse(CamelModel):
    """
    Comprehensive author profile with career metrics and papers.
    Includes all computed fields from paper analysis.
    """
    # Basic info
    id: int
    author_id: str
    openalex_id: Optional[str] = None
    name: str
    display_name: Optional[str] = None
    orcid: Optional[str] = None
    external_ids: Optional[Dict[str, Any]] = None
    
    # Metrics
    h_index: Optional[int] = None
    i10_index: Optional[int] = None
    g_index: Optional[int] = None
    total_citations: Optional[int] = None
    total_papers: Optional[int] = None
    verified: bool = False
    
    # Career trajectory
    first_publication_year: Optional[int] = None
    last_known_institution_id: Optional[int] = None
    
    # Reputation scores
    reputation_score: Optional[float] = None
    field_weighted_citation_impact: Optional[float] = None
    retracted_papers_count: Optional[int] = None
    has_retracted_papers: Optional[bool] = None
    author_institutions: Optional[List[Dict[str, Any]]] = None
    
    is_corresponding_author_frequently: Optional[bool] = None
    average_author_position: Optional[float] = None
    
    # Red flags
    self_citation_rate: Optional[float] = None
    is_processed: bool = False
    is_conflict: bool = False
    
    homepage_url: Optional[str] = None
    url: Optional[str] = None  
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    last_paper_indexed_at: Optional[datetime] = None
    
    # Enrichment status
    is_enriched: bool = Field(
        default=False,
        description="Whether author has been enriched with full paper list"
    )
    
    class Config:
        from_attributes = True


class AuthorDetailWithPapersResponse(AuthorDetailResponse):
    """
    Author detail with papers, quartile breakdown, and co-authors.
    """
    papers: List[AuthorPaperSummary] = []
    quartile_breakdown: QuartileBreakdown
    co_authors: List[CoAuthor] = []
    counts_by_year: Optional[Dict[int, Dict[str, int]]] = Field(
        default=None,
        description="Yearly counts of papers and citations"
    )
    topics: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of topics associated with the author"
    )
    enrichment_status: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Current enrichment status (needs_enrichment/enriching/completed/failed)"
    )
    
    # Computed stats
    papers_by_year: Optional[Dict[int, int]] = Field(
        default=None,
        description="Paper count grouped by year"
    )


class AuthorEnrichmentRequest(BaseModel):
    """Request to enrich an author with papers"""
    force_refresh: bool = Field(
        default=False,
        description="Force re-fetch even if recently indexed"
    )
    limit: int = Field(
        default=500,
        ge=1,
        le=1000,
        description="Maximum papers to fetch"
    )


class AuthorEnrichmentResponse(BaseModel):
    """Response from author enrichment"""
    status: str
    papers_added: int
    papers_existing: int
    total_papers: int
    message: Optional[str] = None

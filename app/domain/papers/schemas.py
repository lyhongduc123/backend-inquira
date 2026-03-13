"""
Paper API schemas - Request/Response DTOs for REST API
Separation: API layer schemas, not for internal data transfer
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from app.core.model import CamelModel
from app.core.dtos import AuthorDTO
from app.domain.authors.schemas import AuthorMetadata
from app.domain.common.schemas import SJRMetadata


class PaperUpdateRequest(CamelModel):
    """API request for updating a paper"""
    title: Optional[str] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    is_open_access: Optional[bool] = None

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
    tldr: Optional[str] = None  # Semantic Scholar TLDR summary
    authors: List[AuthorMetadata] = []
    year: Optional[int] = None
    publication_date: Optional[datetime]
    venue: Optional[str]
    journal: Optional[Any] = None
    conference_id: Optional[int] = None
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
    fields_of_study: Optional[List[str]] = None  # Semantic Scholar fields for filtering
    
    
    relevance_score: Optional[float] = None
    ranking_scores: Optional[Dict[str, float]] = None
    sjr_data: Optional[Dict[str, Any]] = None  # Changed from SJRMetadata to Dict for proper serialization
    
    class Config:
        from_attributes = True
        ignore_extra = True
    
    @classmethod
    def from_db_model(cls, db_paper) -> "PaperMetadata":
        """Convert a DBPaper to PaperMetadata"""
        from app.domain.authors.schemas import AuthorMetadata
        
        if not db_paper:
            raise ValueError("DBPaper is None")
        authors = [AuthorMetadata.model_validate(author_paper.author) for author_paper in db_paper.paper_authors]
        year = db_paper.publication_date.year if db_paper.publication_date else None
        
        # Validate paper but exclude journal ORM relationship
        paper_metadata = cls.model_validate(db_paper, from_attributes=True)
        paper_metadata.journal = None  # Clear ORM object to avoid serialization issues
        paper_metadata.authors = authors
        paper_metadata.year = year
        
        # Convert SJRMetadata to dict for proper serialization
        if db_paper.journal:
            sjr_metadata = SJRMetadata.model_validate(db_paper.journal)
            paper_metadata.sjr_data = sjr_metadata.model_dump()
        else:
            paper_metadata.sjr_data = None
            
        return paper_metadata
    
    @classmethod
    def from_ranked_paper(cls, ranked_paper) -> "PaperMetadata":
        """
        Convert paper DBPaper to consistent metadata dictionary.
        This is the SINGLE SOURCE OF TRUTH for paper metadata format.
        Used for both streaming and snapshot storage in messages.

        Args:
            ranked_paper: RankedPaper model

        Returns:
            Consistent paper metadata dictionary with all fields
        """
        from app.domain.authors.schemas import AuthorMetadata
        
        # Extract data from either Pydantic schema or SQLAlchemy model
        paper = ranked_paper.paper
        year = paper.publication_date.year if paper.publication_date else None
        
        # Convert DBAuthorPaper ORM objects to dicts for validation
        authors = []
        for author_paper in paper.paper_authors:
            author_dict = {
                "author_id": author_paper.author.author_id if author_paper.author else None,
                "name": author_paper.author.name if author_paper.author else None,
                "author_position": author_paper.author_position,
            }
            authors.append(AuthorMetadata.model_validate(author_dict))
        
        # Validate paper but exclude the journal ORM relationship to avoid serialization issues
        paper_metadata = cls.model_validate(paper, from_attributes=True)
        paper_metadata.journal = None  # Clear ORM object
        paper_metadata.authors = authors
        paper_metadata.year = year
        paper_metadata.relevance_score = ranked_paper.relevance_score
        paper_metadata.ranking_scores = ranked_paper.ranking_scores
        
        # Convert SJRMetadata to dict for proper serialization
        if not paper.journal:
            paper_metadata.sjr_data = None
        else:
            sjr_metadata = SJRMetadata.model_validate(paper.journal)
            paper_metadata.sjr_data = sjr_metadata.model_dump()
            
        return paper_metadata


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
    tldr: Optional[str] = None  # Renamed from summary - Semantic Scholar TLDR
    year: Optional[int] = None  # Publication year from Semantic Scholar
    fields_of_study: Optional[List[str]] = None  # S2 fields for filtering
    s2_fields_of_study: Optional[List[Dict[str, str]]] = None  # Detailed S2 field metadata
    
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
    
class ComputeTagsRequest(CamelModel):
    paperId: str = Field(..., description="The unique identifier of the paper to compute tags for")
    content: str = Field(..., description="Text content to compute tags for (e.g., abstract)")
    candidate_labels: List[str] = Field(
        default=["Literature Review", "Meta-Analysis", "Case Study", "Methodology", "Dataset Paper", "Theoretical Paper"],
        description="Candidate tags for zero-shot classification (e.g., paper types or topics)"
    )
    category: str = Field(default="general")


# ============= Backward Compatibility Aliases =============
# To minimize disruption during refactoring
PaperUpdate = PaperUpdateRequest
PaperDetail = PaperDetailResponse  

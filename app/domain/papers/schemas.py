"""
Paper API schemas - Request/Response DTOs for REST API
Separation: API layer schemas, not for internal data transfer
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
from app.core.model import CamelModel
from app.core.dtos import AuthorDTO
from app.domain.authors.schemas import AuthorMetadata
from app.domain.common.schemas import SJRMetadata

from app.extensions.logger import create_logger

logger = create_logger(__name__)

class PaperFilterOptions(BaseModel):
    """Filtering options for paper search"""
    year_from: Optional[int] = Field(None, description="Filter papers published from this year (inclusive)")
    year_to: Optional[int] = Field(None, description="Filter papers published up to this year (inclusive)")
    open_access: Optional[bool] = Field(None, description="Filter for open access papers")
    min_citation_count: Optional[int] = Field(None, description="Filter for papers with at least this many citations")
    max_citation_count: Optional[int] = Field(None, description="Filter for papers with at most this many citations")
    fields_of_study: Optional[List[str]] = Field(None, description="Filter for specific fields of study (e.g., Computer Science)")
    publication_types: Optional[List[str]] = Field(None, description="Filter for specific publication types (e.g., Journal, Conference)")

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
    journal: Optional[SJRMetadata] = None
    conference_id: Optional[int] = None
    url: Optional[str]
    pdf_url: Optional[str]
    external_ids: Optional[Dict[str, Any]] = None
    citation_count: int
    influential_citation_count: Optional[int]
    reference_count: Optional[int]
    citation_styles: Optional[Dict[str, str]] = None
    author_trust_score: Optional[float]
    fwci: Optional[float]
    is_open_access: bool
    is_retracted: bool
    topics: Optional[List[Dict[str, Any]]]
    keywords: Optional[List[Dict[str, Any]]]
    fields_of_study: Optional[List[str]] = None  # Semantic Scholar fields for filtering
    publication_types: Optional[List[str]] = None
    paper_tags: Optional[List[Dict[str, Any]]] = None
    
    
    relevance_score: Optional[float] = None
    ranking_scores: Optional[Dict[str, float]] = None
    
    class Config:
        from_attributes = True
        ignore_extra = True

    @staticmethod
    def _safe_db_paper_payload(db_paper: Any) -> Dict[str, Any]:
        """Extract only DB column attributes to avoid relationship validation issues."""
        if isinstance(db_paper, dict):
            return db_paper

        try:
            from sqlalchemy import inspect as sa_inspect
            from sqlalchemy.orm import NO_VALUE
        except Exception:
            return db_paper.__dict__ if hasattr(db_paper, "__dict__") else {}

        state = sa_inspect(db_paper)
        payload: Dict[str, Any] = {}
        for attr in state.mapper.column_attrs:
            loaded_value = state.attrs[attr.key].loaded_value
            payload[attr.key] = None if loaded_value is NO_VALUE else loaded_value
        return payload

    
    @classmethod
    def from_db_model(cls, db_paper) -> "PaperMetadata":
        """Convert a DBPaper to PaperMetadata"""
        from app.domain.authors.schemas import AuthorMetadata
        
        if not db_paper:
            raise ValueError("DBPaper is None")
        authors = [
            AuthorMetadata.model_validate(author_paper.author, from_attributes=True)
            for author_paper in (db_paper.authors or [])
            if getattr(author_paper, "author", None)
        ]
        year = db_paper.publication_date.year if db_paper.publication_date else None

        paper_metadata = cls.model_validate(cls._safe_db_paper_payload(db_paper))
        paper_metadata.authors = authors
        paper_metadata.year = year
            
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
        from sqlalchemy import inspect as sa_inspect
        
        paper = ranked_paper.paper
        authors: List[AuthorMetadata] = []
        state = sa_inspect(paper)
        unloaded = set(getattr(state, "unloaded", set()))
        paper_payload = cls._safe_db_paper_payload(paper)
        publication_date = paper_payload.get("publication_date")
        year = publication_date.year if publication_date else None

        # Avoid triggering lazy-load in async contexts that are outside greenlet_spawn.
        if "authors" not in unloaded:
            for author_paper in (paper.authors or []):
                if not author_paper:
                    continue
                author_dict = {
                    "author_id": (
                        str(author_paper.author.author_id)
                        if author_paper.author and author_paper.author.author_id is not None
                        else None
                    ),
                    "name": author_paper.author.name if author_paper.author else None,
                    "author_position": author_paper.author_position,
                }
                authors.append(AuthorMetadata.model_validate(author_dict))
        
        paper_payload["paper_id"] = paper_payload.get("paper_id") or str(ranked_paper.paper_id or "")
        paper_payload["title"] = paper_payload.get("title") or ""
        paper_payload["citation_count"] = int(paper_payload.get("citation_count") or 0)
        paper_payload["is_open_access"] = bool(paper_payload.get("is_open_access") or False)
        paper_payload["is_retracted"] = bool(paper_payload.get("is_retracted") or False)

        paper_metadata = cls.model_validate(paper_payload)
        paper_metadata.journal = None  # Clear ORM object
        paper_metadata.authors = authors
        paper_metadata.year = year
        paper_metadata.relevance_score = ranked_paper.relevance_score
        paper_metadata.ranking_scores = ranked_paper.ranking_scores
        
        # Convert SJRMetadata to dict for proper serialization
        # if not paper.journal:
        #     paper_metadata.sjr_data = None
        # else:
        #     sjr_metadata = SJRMetadata.model_validate(paper.journal)
        #     paper_metadata.sjr_data = sjr_metadata.model_dump()
            
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
    journal: Optional[SJRMetadata] = None
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
    publication_types: Optional[List[str]] = None  # S2 publication types
    s2_fields_of_study: Optional[List[Dict[str, str]]] = None  # Detailed S2 field metadata
    paper_tags: Optional[List[Dict[str, Any]]] = None
    
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

    @model_validator(mode="before")
    @classmethod
    def _normalize_db_paper_input(cls, value: Any) -> Any:
        """
        Normalize ORM input safely to avoid async lazy-load access during validation.

        Pydantic `from_attributes` can trigger SQLAlchemy relationship loading when
        accessing `journal` on `DBPaper`. In async contexts this can raise
        MissingGreenlet if the relationship is not eagerly loaded.
        """
        if value is None or isinstance(value, dict):
            return value

        try:
            from app.models.papers import DBPaper
            from sqlalchemy import inspect as sa_inspect
            from sqlalchemy.orm import NO_VALUE
        except Exception:
            return value

        if not isinstance(value, DBPaper):
            return value

        state = sa_inspect(value)

        data: Dict[str, Any] = {
            attr.key: getattr(value, attr.key)
            for attr in state.mapper.column_attrs
        }

        journal_loaded_value = state.attrs.journal.loaded_value
        data["journal"] = None if journal_loaded_value is NO_VALUE else journal_loaded_value

        return data

    

# ============= Citation & Reference Schemas =============

class CitingPaperData(CamelModel):
    """Nested paper data from Semantic Scholar citations API"""
    paper_id: str = Field(...)
    corpus_id: Optional[int] = None
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
    citing_paper: CitingPaperData = Field(...)
    is_influential: Optional[bool] = Field(None)
    contexts: Optional[List[str]] = None
    intents: Optional[List[str]] = None
    
    class Config:
        populate_by_name = True
        from_attributes = True


class ReferencedPaperData(CamelModel):
    """Nested paper data from Semantic Scholar references API"""
    paper_id: str = Field(...)
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

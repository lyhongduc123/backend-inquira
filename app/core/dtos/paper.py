"""
Paper DTOs - Clean data transfer objects following single responsibility principle
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from app.core.dtos.author import AuthorDTO


class PaperDTO(BaseModel):
    """
    Internal paper DTO for data transfer between layers.
    Aligned with DBPaper model structure.
    Source of truth for paper data transfer.
    """
    # Core identifiers
    paper_id: str
    title: str
    abstract: Optional[str] = None
    embedding: Optional[List[float]] = None
    authors: List[AuthorDTO] = []
    publication_date: Optional[datetime] = None
    venue: Optional[str] = None
    issn: Optional[List[str]] = None
    issn_l: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    is_open_access: bool = False
    open_access_pdf: Optional[Dict[str, Any]] = None
    source: str = "SemanticScholar"
    external_ids: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    summary_embedding: Optional[List[float]] = None
    citation_count: Optional[int] = 0
    influential_citation_count: Optional[int] = 0
    reference_count: Optional[int] = 0
    citation_styles: Optional[Dict[str, str]] = None
    topics: Optional[List[Dict[str, Any]]] = None
    keywords: Optional[List[Dict[str, Any]]] = None
    concepts: Optional[List[Dict[str, Any]]] = None
    mesh_terms: Optional[List[Dict[str, Any]]] = None
    citation_percentile: Optional[Dict[str, Any]] = None
    fwci: Optional[float] = None
    author_trust_score: Optional[float] = None
    institutional_trust_score: Optional[float] = None
    network_diversity_score: Optional[float] = None
    journal_id: Optional[int] = None
    is_retracted: bool = False
    language: Optional[str] = None
    corresponding_author_ids: Optional[List[str]] = None
    institutions_distinct_count: Optional[int] = None
    countries_distinct_count: Optional[int] = None
    is_processed: bool = False
    processing_status: str = "pending"
    processing_error: Optional[str] = None
    
    # Database fields (optional, populated when reading from DB)
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_accessed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
        extra = "ignore"


class PaperEnrichedDTO(PaperDTO):
    """
    Extended paper DTO with enrichment data for preprocessing.
    Used in retrieval and processing pipeline.
    Authors field contains merged data from both Semantic Scholar and OpenAlex.
    """
    # Content availability (excluded from DB storage)
    has_content: Dict[str, bool] = Field(default_factory=dict, exclude=True)

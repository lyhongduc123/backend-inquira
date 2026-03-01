"""
Base schemas for retriever module.

Defines Pydantic models for normalized paper and author data across providers.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class AuthorSchema(BaseModel):
    """Author information from provider APIs"""

    name: str = Field(..., description="Author name")
    author_id: Optional[str] = Field(
        default=None, description="Unique author identifier"
    )
    citation_count: Optional[int] = Field(
        default=None, description="Number of citations"
    )
    h_index: Optional[int] = Field(default=None, description="Author h-index")
    paper_count: Optional[int] = Field(default=None, description="Number of papers")
    url: Optional[str] = Field(default=None, description="Author profile URL")
    homepage_url: Optional[str] = Field(
        default=None, description="URL to author's profile"
    )
    # OpenAlex-specific fields
    orcid: Optional[str] = Field(default=None, description="ORCID identifier")
    institutions: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Author's institutions"
    )
    affiliations: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Raw affiliation strings"
    )

    class Config:
        extra = "allow"  # Allow additional fields from providers


class NormalizedPaperResult(BaseModel):
    """
    Normalized paper result from any provider.

    This schema standardizes paper data across Semantic Scholar and OpenAlex,
    supporting rich metadata for author trust scoring and citation analysis.
    """

    # Core fields (required)
    paper_id: str = Field(..., description="Unique paper identifier")
    title: str = Field(..., description="Paper title")
    source: str = Field(
        ..., description="Provider name (e.g., 'semantic_scholar', 'openalex')"
    )
    abstract: Optional[str] = Field(default=None, description="Paper abstract")
    authors: List[AuthorSchema] = Field(
        default_factory=list, description="List of authors"
    )
    publication_date: Optional[str] = Field(
        default=None, description="Publication date (ISO format)"
    )
    venue: Optional[str] = Field(default=None, description="Publication venue")

    # URLs and access
    url: Optional[str] = Field(default=None, description="Paper URL")
    pdf_url: Optional[str] = Field(default=None, description="PDF URL (if available)")
    is_open_access: bool = Field(
        default=False, description="Whether paper is open access"
    )
    open_access_pdf: Optional[Dict[str, str]] = Field(
        default=None,
        description='Open access PDF metadata {"url": str, "status": str, "license": str}',
    )

    # Citation metrics
    citation_count: Optional[int] = Field(default=None, description="Citation count")
    influential_citation_count: Optional[int] = Field(
        default=None, description="Influential citation count (Semantic Scholar only)"
    )
    reference_count: Optional[int] = Field(
        default=None, description="Number of references cited"
    )
    citation_styles: Optional[Dict[str, str]] = Field(
        default=None, description="Citation styles in various formats"
    )
    external_ids: Optional[Dict[str, str | int]] = Field(
        default=None,
        description='External IDs {"DOI": str, "ArXiv": str, "OpenAlex": str, ...}',
    )

    # OpenAlex-specific enrichment fields
    topics: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Research topics with scores"
    )
    keywords: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Keywords with scores"
    )
    concepts: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Concepts with scores and levels"
    )
    mesh_terms: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="MeSH terms for biomedical papers"
    )
    has_content: Optional[Dict[str, Any]] = Field(
        default=None, description="Content availability metadata"
    )

    # Advanced citation metrics
    citation_percentile: Optional[Dict[str, Any]] = Field(
        default=None, description="Citation percentile rankings"
    )
    fwci: Optional[float] = Field(
        default=None, description="Field-weighted citation impact"
    )
    is_retracted: Optional[bool] = Field(default=False, description="Retraction status")
    language: Optional[str] = Field(default=None, description="Paper language code")

    # Publication metadata
    biblio: Optional[Dict[str, Any]] = Field(
        default=None, description="Bibliographic info (volume, issue, pages) - values can be strings or None"
    )
    primary_location: Optional[Dict[str, Any]] = Field(
        default=None, description="Primary publication location"
    )
    locations: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="All publication locations"
    )
    best_oa_location: Optional[Dict[str, Any]] = Field(
        default=None, description="Best open access location"
    )

    # Author collaboration metadata (for author reputation scoring)
    corresponding_author_ids: Optional[List[str]] = Field(
        default=None, description="IDs of corresponding authors"
    )
    institutions_distinct_count: Optional[int] = Field(
        default=None, description="Number of unique institutions"
    )
    countries_distinct_count: Optional[int] = Field(
        default=None, description="Number of unique countries"
    )

    openalex_data: Optional[Dict[str, Any]] = Field(
        default=None, description="OpenAlex enrichment data"
    )
    semantic_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Semantic Scholar data"
    )

    class Config:
        extra = "allow"  # Allow additional provider-specific fields
        json_schema_extra = {
            "example": {
                "paper_id": "W2123456789",
                "title": "Example Paper Title",
                "source": "openalex",
                "abstract": "This is an example abstract...",
                "authors": [
                    {
                        "name": "John Doe",
                        "author_id": "A123456",
                        "h_index": 25,
                        "citation_count": 1500,
                    }
                ],
                "publication_date": "2024-01-15",
                "venue": "Nature",
                "citation_count": 42,
                "is_open_access": True,
            }
        }


class NormalizedAuthorResult(BaseModel):
    """
    Normalized author result from any provider.

    This schema standardizes author data across Semantic Scholar and OpenAlex,
    supporting author profile enrichment and paper fetching.
    """

    # Core fields (required)
    author_id: str = Field(..., description="Unique author identifier (primary)")
    name: str = Field(..., description="Author display name")
    source: str = Field(
        ..., description="Provider name (e.g., 'semantic_scholar', 'openalex')"
    )

    # Alternative identifiers
    openalex_id: Optional[str] = Field(
        default=None, description="OpenAlex author ID (stored separately)"
    )
    orcid: Optional[str] = Field(default=None, description="ORCID identifier")

    # Metrics
    h_index: Optional[int] = Field(default=None, description="Author h-index")
    citation_count: Optional[int] = Field(
        default=None, description="Total citation count"
    )
    paper_count: Optional[int] = Field(
        default=None, description="Total number of papers"
    )

    # Affiliation information
    institutions: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Author's institutions"
    )
    last_known_institution: Optional[Dict[str, Any]] = Field(
        default=None, description="Most recent institution"
    )

    # External IDs and URLs
    external_ids: Optional[Dict[str, str]] = Field(
        default=None,
        description='External IDs {"openalex": str, "orcid": str, "semantic_scholar": str, ...}',
    )
    profile_url: Optional[str] = Field(
        default=None, description="URL to author's profile"
    )

    # Paper list (for enrichment)
    papers: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Author's papers (when fetching paper list)"
    )

    class Config:
        extra = "allow"  # Allow additional provider-specific fields
        json_schema_extra = {
            "example": {
                "author_id": "A5023888391",
                "name": "Jane Smith",
                "source": "openalex",
                "h_index": 42,
                "citation_count": 8500,
                "paper_count": 125,
                "orcid": "0000-0002-1234-5678",
            }
        }

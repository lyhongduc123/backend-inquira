from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class OAMeta(BaseModel):
    """Metadata for OpenAlex API response."""

    count: int = Field(..., description="Total number of results")
    db_response_time_ms: float = Field(..., description="Database response time in ms")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Number of results per page")


class OAResponse(BaseModel):
    """Schema for OpenAlex API response."""

    meta: OAMeta = Field(..., description="Response metadata")
    results: List[Dict[str, Any]] = Field(..., description="List of paper results")


class OAAuthorResponse(BaseModel):
    """Schema for OpenAlex Author API response."""

    id: str = Field(..., description="OpenAlex Author ID")
    display_name: str = Field(..., description="Author's display name")
    display_name_alternatives: Optional[List[str]] = Field(
        None, description="Alternative display names"
    )
    orcid: Optional[str] = Field(None, description="Author's ORCID")
    works_count: int = Field(..., description="Number of works by the author")
    cited_by_count: int = Field(
        ..., description="Number of citations received by the author"
    )
    summary_stats: Optional[Dict[str, Any]] = Field(
        None, description="Summary statistics about the author's works"
    )
    topics: Optional[List[Dict[str, Any]]] = Field(
        None, description="List of topics associated with the author"
    )
    affiliation: Optional[Dict[str, Any]] = Field(
        None, description="Author's affiliation details"
    )
    counts_by_year: Optional[List[Dict[str, Any]]] = Field(
        None, description="Yearly counts of works and citations"
    )

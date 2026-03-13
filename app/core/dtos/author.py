"""
Author DTOs - Single source of truth for author data transfer objects
"""
from typing import Optional
from pydantic import BaseModel


class AuthorDTO(BaseModel):
    """
    Lightweight author DTO for internal data transfer.
    Used in paper schemas and retrieval operations.
    Matches data from Semantic Scholar and OpenAlex.
    """
    name: str
    author_id: Optional[str] = None
    openalex_id: Optional[str] = None  # OpenAlex ID stored separately
    citation_count: Optional[int] = None
    h_index: Optional[int] = None
    paper_count: Optional[int] = None
    orcid: Optional[str] = None
    url: Optional[str] = None  # Author profile URL (from Semantic Scholar)
    homepage_url: Optional[str] = None  # Author homepage (from Semantic Scholar)
    
    # OpenAlex-specific fields (from merged data)
    institutions: Optional[list] = None
    affiliations: Optional[list] = None
    
    class Config:
        extra = "ignore"

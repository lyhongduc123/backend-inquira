from typing import List, TypedDict, Dict, Optional


class AuthorDict(TypedDict, total=False):
    """Author information from provider APIs"""
    name: str                    # Author name
    author_id: Optional[str]     # Unique author identifier
    citation_count: Optional[int]  # Number of citations
    h_index: Optional[int]       # Author h-index


class NormalizedResult(TypedDict, total=False):
    """Normalized paper result from any provider"""
    paper_id: str                # Unique paper identifier
    title: str                   # Paper title
    abstract: Optional[str]      # Paper abstract
    authors: List[AuthorDict]    # List of authors
    publication_date: Optional[str]  # Publication date (ISO format)
    venue: Optional[str]         # Publication venue
    url: Optional[str]           # Paper URL
    pdf_url: Optional[str]       # PDF URL (if available)
    is_open_access: bool         # Whether paper is open access
    open_access_pdf: Optional[Dict[str, str]]  # Open access PDF metadata {"url": str, "status": str, "license": str}
    citation_count: Optional[int]  # Citation count
    influential_citation_count: Optional[int]  # Influential citation count
    reference_count: Optional[int]  # Number of references cited
    external_ids: Optional[Dict[str, str]]  # {"DOI": str, "ArXiv": str, ...}
    source: str                  # Provider name (e.g., "semantic_scholar")
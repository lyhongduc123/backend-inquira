"""
Retriever schemas package.

Exports all schema models for the retriever module.
"""
from .base import AuthorSchema, NormalizedPaperResult, NormalizedAuthorResult
from .openalex import OAMeta, OAResponse
from .semantic_scholar import S2RelevanceResponse, S2AuthorPapersResponse, S2PaperCitationsResponse, S2PaperReferencesResponse

__all__ = [
    "AuthorSchema",
    "NormalizedPaperResult",
    "NormalizedAuthorResult",
    "OAMeta",
    "OAResponse",
    "S2RelevanceResponse",
    "S2AuthorPapersResponse",
    "S2PaperCitationsResponse",
    "S2PaperReferencesResponse",
]

"""
Papers module for CRUD operations on papers
"""
from .service import PaperService
from .repository import PaperRepository, LoadOptions, SearchFilterOptions
from .schemas import (
    PaperDetailResponse,
    PaperUpdateRequest,
    PaperMetadata,
    SJRMetadata,
    PaginatedCitationsResponse,
    PaginatedReferencesResponse,
)
from .linking_service import PaperLinkingService
from .journal_service import JournalService

__all__ = [
    "PaperService",
    "PaperRepository",
    "LoadOptions",
    "SearchFilterOptions",
    "PaperDetailResponse",
    "PaperUpdateRequest",
    "PaperMetadata",
    "SJRMetadata",
    "PaginatedCitationsResponse",
    "PaginatedReferencesResponse",
    "PaperLinkingService",
    "JournalService",
]

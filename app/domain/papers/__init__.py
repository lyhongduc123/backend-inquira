"""
Papers module for CRUD operations on papers
"""
from .service import PaperService
from .repository import PaperRepository, LoadOptions
from .schemas import (
    PaperDetailResponse,
    PaperUpdateRequest,
    PaperMetadata,
    SJRMetadata,
    PaginatedCitationsResponse,
    PaginatedReferencesResponse,
)
from .enrichment_service import PaperEnrichmentService
from .journal_service import JournalService

__all__ = [
    "PaperService",
    "PaperRepository",
    "LoadOptions",
    "PaperDetailResponse",
    "PaperUpdateRequest",
    "PaperMetadata",
    "SJRMetadata",
    "PaginatedCitationsResponse",
    "PaginatedReferencesResponse",
    "PaperEnrichmentService",
    "JournalService",
]

"""
Authors module for managing author entities and relationships.
"""
from .service import AuthorService
from .repository import AuthorRepository
from .schemas import AuthorMetadata, AuthorDetailResponse, AuthorDetailWithPapersResponse

__all__ = [
    "AuthorService",
    "AuthorRepository",
    "AuthorMetadata",
    "AuthorDetailResponse",
    "AuthorDetailWithPapersResponse",
]

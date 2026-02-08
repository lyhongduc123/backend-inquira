"""
Authors module for managing author entities and relationships.
"""
from app.authors.repository import AuthorRepository
from app.authors.service import AuthorService

__all__ = [
    "AuthorRepository",
    "AuthorService",
]

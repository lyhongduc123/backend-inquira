"""
Retriever schemas package.

Exports all schema models for the retriever module.
"""
from .base import AuthorSchema, NormalizedResult

__all__ = [
    "AuthorSchema",
    "NormalizedResult",
]

"""Compatibility exports for legacy DTO imports.

New code should import paper and author transfer types from their domain modules.
"""
from .paper import PaperDTO, PaperEnrichedDTO
from .author import AuthorDTO

__all__ = [
    'PaperDTO',
    'PaperEnrichedDTO',
    'AuthorDTO',
]

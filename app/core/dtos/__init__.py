"""Core DTOs package - Single source of truth for data transfer objects"""
from .paper import PaperDTO, PaperEnrichedDTO
from .author import AuthorDTO

__all__ = [
    'PaperDTO',
    'PaperEnrichedDTO',
    'AuthorDTO',
]
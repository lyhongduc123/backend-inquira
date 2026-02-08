"""
Institutions module for managing institution entities.
"""
from app.institutions.repository import InstitutionRepository
from app.institutions.service import InstitutionService

__all__ = [
    "InstitutionRepository",
    "InstitutionService",
]

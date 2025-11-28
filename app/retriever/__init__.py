from typing import List, Dict, Any, Optional
from enum import Enum
from app.core.config import settings

from .provider import (
    SemanticScholarProvider,
    ArxivProvider,
    OpenAlexProvider,
    GoogleScholarProvider,
    RetrievalConfig,
    RetrievalMode
)

from .paper_schemas import Paper, PaperChunk, Author, Citation
from .paper_retriever import PaperRetriever
from .paper_repository import PaperRepository
from .paper_service import PaperRetrievalService, RetrievalServiceType

# Export all components
__all__ = [
    'PaperRetrievalService',
    'RetrievalServiceType',
    
    # Providers
    'SemanticScholarProvider',
    'ArxivProvider',
    'OpenAlexProvider',
    'GoogleScholarProvider',
    
    # Configuration
    'RetrievalConfig',
    'RetrievalMode',
    
    # Data models
    'Paper',
    'PaperChunk',
    'Author',
    'Citation',
    
    # Utilities
    'PaperRetriever',
    'PaperRepository',
]
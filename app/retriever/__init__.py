"""
Retriever module for paper search and retrieval.

Provides:
- Multi-provider paper search (Semantic Scholar, OpenAlex)
- Normalized result schemas
- Hybrid search with metadata enrichment
"""
from .provider import (
    SemanticScholarProvider,
    OpenAlexProvider,
    RetrievalConfig,
    RetrievalMode,
    BaseRetrievalProvider,
)
from .schemas import NormalizedResult, AuthorSchema
from .paper_schemas import Paper, PaperChunk, Author, Citation
from .paper_retriever import PaperRetriever
from .paper_service import PaperRetrievalService, RetrievalServiceType

# Export all components
__all__ = [
    # Main service
    'PaperRetrievalService',
    'RetrievalServiceType',
    
    # Providers
    'SemanticScholarProvider',
    'OpenAlexProvider',
    'BaseRetrievalProvider',
    
    # Configuration
    'RetrievalConfig',
    'RetrievalMode',
    
    # Schemas
    'NormalizedResult',
    'AuthorSchema',
    
    # Data models
    'Paper',
    'PaperChunk',
    'Author',
    'Citation',
    
    # Utilities
    'PaperRetriever',
]
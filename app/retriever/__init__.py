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
    BaseRetrievalProvider,
)
from .schemas import NormalizedPaperResult, NormalizedAuthorResult, AuthorSchema
from .external_retriever import ExternalPaperRetriever
from .service import RetrievalService, RetrievalServiceType

# Export all components
__all__ = [
    # Main service
    'RetrievalService',
    'RetrievalServiceType',
    
    # Providers
    'SemanticScholarProvider',
    'OpenAlexProvider',
    'BaseRetrievalProvider',
    
    # Configuration
    'RetrievalConfig',

    # Schemas
    'NormalizedPaperResult',
    'NormalizedAuthorResult',
    'AuthorSchema',
    
    # Utilities
    'ExternalPaperRetriever',
]
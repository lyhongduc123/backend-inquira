from typing import List, Dict, Any, Optional
from enum import Enum
from app.core.config import settings

# Export provider-based architecture
from .provider import (
    SemanticScholarProvider,
    ArxivProvider,
    OpenAlexProvider,
    GoogleScholarProvider,
    RetrievalConfig,
    RetrievalMode
)

# Export data models and utilities
from .paper_schemas import Paper, PaperChunk, Author, Citation
from .paper_retriever import PaperRetriever
from .chunker import TextChunker
from .embeddings import EmbeddingService
from .paper_repository import PaperRepository

class RetrievalServiceType(str, Enum):
    SEMANTIC = 'semantic'
    ARXIV = 'arxiv'
    SCHOLAR = 'scholar'
    OPENALEX = 'openalex'

class RetrieverService:
    """
    Unified Retriever Service using the new provider architecture.
    
    This service manages multiple retrieval providers and provides a unified
    interface for searching papers across different sources.
    
    Features:
    - Multiple providers (Semantic Scholar, arXiv, OpenAlex, Google Scholar)
    - Configurable caching and full-text retrieval
    - Async/await support
    - Standard normalized results
    """
    
    def __init__(self, db_session=None, enable_caching: bool = False, enable_full_text: bool = False):
        """
        Initialize the retriever service with providers.
        
        Args:
            db_session: Optional database session for caching
            enable_caching: If True, enable database caching for Semantic Scholar
            enable_full_text: If True, enable full-text retrieval for arXiv
        """
        self.db_session = db_session
        self.enable_caching = enable_caching
        self.enable_full_text = enable_full_text
        
        # Initialize providers with appropriate configurations
        self.providers: Dict[RetrievalServiceType, Any] = {
            RetrievalServiceType.SEMANTIC: SemanticScholarProvider(
                api_url=settings.SEMANTIC_API_URL,
                config=RetrievalConfig(
                    mode=RetrievalMode.ENHANCED if enable_caching else RetrievalMode.SIMPLE,
                    enable_caching=enable_caching,
                    enable_full_text=enable_full_text,
                    max_results=100
                ),
                db_session=db_session
            ),
            RetrievalServiceType.ARXIV: ArxivProvider(
                api_url=settings.ARXIV_API_URL,
                config=RetrievalConfig(
                    mode=RetrievalMode.FULL if enable_full_text else RetrievalMode.SIMPLE,
                    enable_full_text=enable_full_text,
                    max_results=100
                )
            ),
            RetrievalServiceType.SCHOLAR: GoogleScholarProvider(
                api_url=settings.SCHOLAR_URL,
                config=RetrievalConfig(
                    mode=RetrievalMode.SIMPLE,
                    max_results=20
                )
            ),
            RetrievalServiceType.OPENALEX: OpenAlexProvider(
                api_url=settings.OPENALEX_URL,
                config=RetrievalConfig(
                    mode=RetrievalMode.ENHANCED,
                    max_results=20
                )
            )
        }

    async def search(
        self,
        search_services: List[RetrievalServiceType],
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for papers using the specified retrieval providers.

        Args:
            search_services: List of providers to use
            query: Search query string
            limit: Maximum results per provider

        Returns:
            List of normalized paper dictionaries
        """
        results: List[Dict[str, Any]] = []
        for service_type in search_services:
            provider = self.providers.get(service_type)
            if not provider:
                print(f"Provider {service_type} not found.")
                continue

            try:
                provider_results = await provider.search_papers(query, limit=limit)
                results.extend(provider_results)
            except Exception as e:
                print(f"Error fetching from {service_type.value}: {e}")

        return results

    async def search_all(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for papers using all available retrieval providers.

        Args:
            query: Search query string
            limit: Maximum results per provider

        Returns:
            List of normalized paper dictionaries from all providers
        """
        results = []
        for provider in self.providers.values():
            try:
                provider_results = await provider.search_papers(query, limit=limit)
                results.extend(provider_results)
            except Exception as e:
                print(f"Error fetching from {provider.__class__.__name__}: {e}")
        return results
    
    async def search_with_caching(
        self,
        query: str,
        db_session,
        search_services: List[RetrievalServiceType] = [RetrievalServiceType.SEMANTIC],
        limit: int = 10,
        auto_process: bool = True
    ):
        """Search for papers with database caching and vector search.
        
        This method:
        1. Searches the specified service for papers
        2. Checks if papers are cached in database
        3. If not cached and auto_process=True, retrieves full-text and processes
        4. Returns database objects with embeddings for vector search
        
        Args:
            query: Search query
            db_session: Database session
            service_type: Which retrieval service to use
            limit: Number of papers to retrieve
            auto_process: Whether to auto-process papers (fetch PDF, chunk, embed)
            
        Returns:
            List of DBPaper objects from database
        """
        from .paper_service import PaperRetrievalService
        
        paper_service = PaperRetrievalService(db_session)
        return await paper_service.search_and_retrieve_papers(
            query=query,
            limit=limit,
            auto_process=auto_process
        )

retriever = RetrieverService()

# Export all components
__all__ = [
    # Main service
    'retriever',
    'RetrieverService',
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
    'TextChunker',
    'EmbeddingService',
    'PaperRepository',
]
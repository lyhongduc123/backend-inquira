"""
Abstract base classes for paper retrieval providers.

This module defines the interfaces that all retrieval providers must implement.
Supports both simple (metadata-only) and enhanced (full-text) retrieval.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from app.modules.httpclient import HTTPClient
from app.retriever.provider.base_schemas import NormalizedResult


class RetrievalMode(str, Enum):
    """Retrieval mode for providers"""
    SIMPLE = "simple"  # Metadata only
    ENHANCED = "enhanced"  # With full-text and caching
    FULL = "full"  # Complete pipeline (chunking, embedding, etc.)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval providers"""
    mode: RetrievalMode = RetrievalMode.SIMPLE
    enable_caching: bool = False
    enable_full_text: bool = False
    max_results: int = 10
    timeout: float = 30.0


class BaseRetrievalProvider(ABC):
    """
    Abstract base class for all retrieval providers.
    
    All providers must implement:
    - search_papers(): Retrieve paper metadata
    - get_paper_details() (optional): Get full details for specific paper
    - supports_full_text(): Whether provider supports full-text retrieval
    """

    def __init__(self, api_url: str, config: Optional[RetrievalConfig] = None):
        """
        Initialize retrieval provider.
        
        Args:
            api_url: Base URL for the provider's API
            config: Optional configuration for retrieval behavior
        """
        self.api_url = api_url
        self.config = config or RetrievalConfig()
        self.client = HTTPClient()
        self._name = self.__class__.__name__.replace("Provider", "").replace("Retrieval", "")

    @property
    def name(self) -> str:
        """Get provider name"""
        return self._name

    @abstractmethod
    async def search_papers(
        self,
        query: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search for papers using the provider's API.
        
        Args:
            query: Search query string
            limit: Maximum number of results (uses config default if None)
            offset: Offset for pagination
            
        Returns:
            List of paper metadata dictionaries
            
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(f"{self.name} must implement search_papers()")

    async def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific paper.
        
        Optional method - providers can override to support detailed retrieval.
        
        Args:
            paper_id: Unique identifier for the paper
            
        Returns:
            Paper details dictionary, or None if not supported/found
        """
        return None

    def supports_full_text(self) -> bool:
        """
        Check if provider supports full-text retrieval.
        
        Returns:
            True if provider can retrieve full paper text
        """
        return False

    def supports_caching(self) -> bool:
        """
        Check if provider supports database caching.
        
        Returns:
            True if provider can cache results
        """
        return False

    @abstractmethod
    def normalize_result(self, raw_result: Dict[str, Any]) -> NormalizedResult:
        """
        Normalize provider-specific result to standard format.
        
        Standard format:
        {
            "id": str,              # Unique paper ID
            "title": str,           # Paper title
            "abstract": str,        # Abstract/summary
            "authors": List[Dict],  # [{"name": str, "id": str}]
            "year": int,            # Publication year
            "venue": str,           # Publication venue
            "url": str,             # Paper URL
            "pdfUrl": str,         # PDF URL (if available)
            "citationCount": int,  # Citation count (if available)
            "externalIds": Dict,   # {"DOI": str, "ArXiv": str, ...}
            "source": str,          # Provider name
        }
        
        Args:
            raw_result: Provider-specific result dictionary
            
        Returns:
            Normalized result dictionary
        """
        raise NotImplementedError(f"{self.name} must implement normalize_result()")

    async def search_and_normalize(
        self,
        query: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[NormalizedResult]:
        """
        Search papers and normalize results to standard format.
        
        Args:
            query: Search query
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of normalized paper dictionaries
        """
        results = await self.search_papers(query, limit, offset)
        normalized = [self.normalize_result(r) for r in results]
        return normalized


class BaseFullTextProvider(BaseRetrievalProvider):
    """
    Abstract base for providers that support full-text retrieval.
    
    Extends BaseRetrievalProvider with full-text methods.
    """

    def supports_full_text(self) -> bool:
        """Override to indicate full-text support"""
        return True

    @abstractmethod
    async def retrieve_full_text(
        self,
        paper_id: str,
        pdf_url: Optional[str] = None
    ) -> Optional[str]:
        """
        Retrieve full text of a paper.
        
        Args:
            paper_id: Unique paper identifier
            pdf_url: Optional direct PDF URL
            
        Returns:
            Full text string, or None if retrieval failed
        """
        raise NotImplementedError(f"{self.name} must implement retrieve_full_text()")

    async def is_open_access(self, paper_id: str) -> bool:
        """
        Check if paper is open access.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            True if paper is openly accessible
        """
        return False


class BaseCachedProvider(BaseFullTextProvider):
    """
    Abstract base for providers with database caching support.
    
    Adds caching capabilities to full-text providers.
    """

    def __init__(
        self,
        api_url: str,
        config: Optional[RetrievalConfig] = None,
        db_session=None
    ):
        """
        Initialize cached provider.
        
        Args:
            api_url: API base URL
            config: Retrieval configuration
            db_session: Optional database session for caching
        """
        super().__init__(api_url, config)
        self.db_session = db_session

    def supports_caching(self) -> bool:
        """Override to indicate caching support"""
        return True

    @abstractmethod
    async def get_from_cache(self, query: str) -> Optional[List[NormalizedResult]]:
        """
        Retrieve results from cache.
        
        Args:
            query: Search query
            
        Returns:
            Cached results, or None if not in cache
        """
        raise NotImplementedError(f"{self.name} must implement get_from_cache()")

    @abstractmethod
    async def save_to_cache(
        self,
        query: str,
        results: List[NormalizedResult]
    ) -> bool:
        """
        Save results to cache.
        
        Args:
            query: Search query
            results: Results to cache
            
        Returns:
            True if successfully cached
        """
        raise NotImplementedError(f"{self.name} must implement save_to_cache()")

    async def search_with_cache(
        self,
        query: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[NormalizedResult]:
        """
        Search with automatic caching.
        
        Checks cache first, falls back to API if needed.
        
        Args:
            query: Search query
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of normalized paper dictionaries
        """
        # Try cache first
        if self.config.enable_caching and self.db_session:
            cached = await self.get_from_cache(query)
            if cached is not None:
                return cached[offset:offset + (limit or self.config.max_results)]

        # Fetch from API
        results = await self.search_and_normalize(query, limit, offset)

        # Save to cache
        if self.config.enable_caching and self.db_session and results:
            await self.save_to_cache(query, results)

        return results


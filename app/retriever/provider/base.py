"""
Abstract base class for paper retrieval providers.

Defines the interface that all retrieval providers must implement.
Simplified to support only metadata retrieval and normalization.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from enum import Enum
from dataclasses import dataclass

from app.retriever.schemas import NormalizedPaperResult

@dataclass
class RetrievalConfig:
    """Configuration for retrieval providers"""
    max_results: int = 100
    timeout: float = 30.0


class BaseRetrievalProvider(ABC):
    """
    Abstract base class for all retrieval providers.

    Providers must implement:
    - search_papers(): Retrieve raw paper metadata from API
    - normalize_result(): Convert provider-specific format to NormalizedResult

    The base class provides:
    - search_and_normalize(): Convenience method combining search + normalization
    - Configuration management
    """

    def __init__(
        self,
        api_url: str,
        config: Optional[RetrievalConfig] = None,
    ):
        """
        Initialize retrieval provider.

        Args:
            api_url: Base URL for the provider's API
            api_key: Optional API key for authentication
            config: Optional configuration for retrieval behavior
        """
        self.api_url = api_url
        self.config = config or RetrievalConfig()
        self._name = self.__class__.__name__.replace("Provider", "")

    @property
    def name(self) -> str:
        """Get provider name"""
        return self._name

    @abstractmethod
    async def search_papers(
        self,
        query: str,
        limit: Optional[int] = None,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for papers using the provider's API.

        Args:
            query: Search query string
            limit: Maximum number of results (uses config default if None)
            offset: Offset for pagination
            filters: Optional filters (yearRange, category, openAccessOnly, etc.)

        Returns:
            List of raw paper metadata dictionaries from the provider

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(f"{self.name} must implement search_papers()")

    @abstractmethod
    def normalize_result(self, raw_result: Dict[str, Any]) -> NormalizedPaperResult:
        """
        Normalize provider-specific result to standard format.

        Converts provider's raw API response into a standardized NormalizedResult
        schema that can be used across all providers.

        Args:
            raw_result: Provider-specific result dictionary

        Returns:
            NormalizedPaperResult Pydantic model with standardized fields

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(f"{self.name} must implement normalize_result()")

    async def search_and_normalize(
        self,
        query: str,
        limit: Optional[int] = None,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[NormalizedPaperResult]:
        """
        Search papers and normalize results to standard format.

        Convenience method that combines search_papers() and normalize_result().

        Args:
            query: Search query
            limit: Maximum results
            offset: Pagination offset
            filters: Optional filters (yearRange, category, openAccessOnly, etc.)

        Returns:
            List of normalized paper results
        """
        raw_results = await self.search_papers(query, limit, offset, filters=filters)
        normalized = [self.normalize_result(r) for r in raw_results]
        return normalized

    async def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific paper.

        Optional method - providers can override to support detailed retrieval by ID.

        Args:
            paper_id: Unique identifier for the paper

        Returns:
            Paper details dictionary, or None if not supported/found
        """
        return None

class RetrievalProvider(Protocol):
    """
    Protocol for retrieval providers.
    """

    async def search_papers(
        self,
        query: str,
        limit: Optional[int] = None,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        ...

    def normalize_result(self, raw_result: Dict[str, Any]) -> NormalizedPaperResult:
        ...
    
    async def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        ...
        
    async def search_and_normalize(
        self,
        query: str,
        limit: Optional[int] = None,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[NormalizedPaperResult]:
        ...
from typing import List, Dict, Any, Protocol
from enum import Enum
from app.core.config import settings
from .services import (ScholarRetrievalService, SemanticRetrievalService, ArxivRetrievalService, OpenAlexRetrievalService)
from .base import BaseRetrievalService

class RetrievalServiceType(str, Enum):
    SEMANTIC = 'semantic'
    ARXIV = 'arxiv'
    SCHOLAR = 'scholar'
    OPENALEX = 'openalex'

class RetrieverService:
    """
    Main Retriever Service that manages multiple retrieval providers
    """
    def __init__(self):
        self.services: Dict[RetrievalServiceType, BaseRetrievalService] = {
            RetrievalServiceType.SEMANTIC: SemanticRetrievalService(api_url=settings.SEMANTIC_API_URL),
            RetrievalServiceType.ARXIV: ArxivRetrievalService(api_url=settings.ARXIV_API_URL),
            RetrievalServiceType.SCHOLAR: ScholarRetrievalService(api_url=settings.SCHOLAR_URL),
            RetrievalServiceType.OPENALEX: OpenAlexRetrievalService(api_url=settings.OPENALEX_URL)
        }

    async def search(self, search_services: List[RetrievalServiceType], query: str) -> List[Dict[str, Any]]:
        """Search for papers using the specified retrieval services.

        Args:
            search_services (List[RetrievalServiceType]): The retrieval services to use.
            query (str): The search query.

        Returns:
            (List[Dict[str, Any]]): The search results from the specified retrieval services.
        """
        results: List[Dict[str, Any]] = []
        for service_type in search_services:
            service = self.services.get(service_type)
            if not service:
                print(f"Service {service_type} not found.")
                continue

            try:
                service_results = service.fetch(query)
                results.extend(service_results)
            except Exception as e:
                print(f"Error fetching from {service_type.value}: {e}")

        return results

    def search_all(self, query: str) -> List[Dict[str, Any]]:
        """Search for papers using all available retrieval services.

        Args:
            query (str): The search query.

        Returns:
            (List[Dict[str, Any]]): The search results from all available retrieval services.
        """
        results = []
        for service in self.services.values():
            try:
                service_results = service.fetch(query)
                results.extend(service_results)
            except Exception as e:
                print(f"Error fetching from {service.__class__.__name__}: {e}")
        return results

retriever = RetrieverService()
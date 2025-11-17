from abc import ABC, abstractmethod
from typing import Any, Dict, List
from app.modules.httpclient import HTTPClient

class BaseRetrievalService(ABC):
    """Abstract base class for all retriever providers."""

    def __init__(self, api_url: str, client: HTTPClient = HTTPClient()):
        self.api_url = api_url
        self.client = client

    @abstractmethod
    def fetch(self, query: str) -> List[Dict[str, Any]]:
        """Fetch data from the provider based on a query.

        Args:
            query (str): The search query.

        Returns:
            (List[Dict[str, Any]]): A list of search results as dictionaries.
        """
        pass

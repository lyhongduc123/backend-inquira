from .base import BaseRetrievalService
from typing import Any, Dict, List
from app.core.config import settings
from app.modules.crawler import crawler
import xml.etree.ElementTree as ET
import requests

class SemanticRetrievalService(BaseRetrievalService):
    def fetch(self, query: str) -> List[Dict[str, Any]]:
        """Fetches search results from the Semantic Scholar API and returns them as a list of dicts."""
        headers = {
            "x-api-key": settings.SEMANTIC_API_KEY
        }
        response = self.client.get(f"{self.api_url}/paper/search", headers=headers, params={
            "query": query,
            "fields": "url,abstract,authors,title,year,isOpenAccess,openAccessPdf,influentialCitationCount,citationCount",
        })
        if response is None:
            return []
        return response.json().get("data", [])  

class ArxivRetrievalService(BaseRetrievalService):
    def fetch(self, query: str) -> List[Dict[str, Any]]:
        """Fetches search results from the arXiv API and returns them as a list of dicts."""
        params = {
            "search_query": query,
            "start": 0,
            "max_results": 10,
        }
        response = self.client.get(f"{self.api_url}query", params=params)
        if response is None:
            return []
        
        # Parse XML instead of JSON
        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        results = []
        for entry in root.findall("atom:entry", ns):
            title = entry.findtext("atom:title", default="", namespaces=ns).strip()
            summary = entry.findtext("atom:summary", default="", namespaces=ns).strip()
            paper_id = entry.findtext("atom:id", default="", namespaces=ns).strip()

            results.append({
                "id": paper_id,
                "title": title,
                "summary": summary,
            })

        return results
    
class ScholarRetrievalService(BaseRetrievalService):
    def fetch(self, query: str) -> List[Dict[str, Any]]:
        """Fetches search results from the Scholar API and returns them as a list of dicts."""
        html_content = crawler.fetch_page(f"{self.api_url}/search?q={query}")
        parsed_data = crawler.parse_content(html_content)
        return parsed_data
    
class OpenAlexRetrievalService(BaseRetrievalService):
    def fetch(self, query: str) -> List[Dict[str, Any]]:

        response = self.client.get(f"{self.api_url}/works", params={
            "search": query,
            "per-page": 20,
        })
        if response is None:
            return []
        return response.json().get("results", [])
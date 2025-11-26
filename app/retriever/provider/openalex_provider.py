"""
OpenAlex retrieval provider.

Implements BaseRetrievalProvider for OpenAlex API.
"""
import httpx
from typing import List, Dict, Any, Optional
from app.extensions.logger import create_logger
from .base import BaseRetrievalProvider, RetrievalConfig
from .base_schemas import NormalizedResult, AuthorDict

logger = create_logger(__name__)


class OpenAlexProvider(BaseRetrievalProvider):
    """
    OpenAlex retrieval provider.
    
    Features:
    - Open academic graph
    - Rich metadata
    - Citation relationships
    """

    def __init__(self, api_url: str, config: Optional[RetrievalConfig] = None):
        super().__init__(api_url, config)
        self.timeout = config.timeout if config else 30.0

    async def search_papers(
        self,
        query: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search papers via OpenAlex API.
        
        Args:
            query: Search query
            limit: Max results
            offset: Pagination offset
            
        Returns:
            List of raw API results
        """
        limit = limit or self.config.max_results
        
        params = {
            "search": query,
            "per-page": min(limit, 200),
            "page": (offset // limit) + 1 if limit > 0 else 1
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.api_url}/works", params=params)
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                logger.info(f"[{self.name}] Retrieved {len(results)} papers for: {query[:50]}...")
                return results
                
        except httpx.HTTPError as e:
            logger.error(f"[{self.name}] API error: {e}")
            return []
        except Exception as e:
            logger.error(f"[{self.name}] Search error: {e}")
            return []

    def normalize_result(self, raw_result: Dict[str, Any]) -> NormalizedResult:
        """
        Normalize OpenAlex result to standard format.
        
        Args:
            raw_result: Raw API response
            
        Returns:
            Normalized paper dictionary
        """
        # Extract authors
        authorships = raw_result.get("authorships", []) or []
        authors: List[AuthorDict] = [
            AuthorDict(
                name=auth.get("author", {}).get("display_name", "Unknown"),
                author_id=auth.get("author", {}).get("id")
            )
            for auth in authorships
        ]
        
        # Extract IDs
        ids = raw_result.get("ids", {}) or {}
        doi = ids.get("doi", "").replace("https://doi.org/", "") if ids.get("doi") else None
        
        # Extract open access info
        open_access = raw_result.get("open_access", {}) or {}
        pdf_url = open_access.get("oa_url")
        
        # Extract publication date
        year = raw_result.get("publication_year")
        publication_date = f"{year}-01-01" if year else None
        
        return NormalizedResult(
            paper_id=raw_result.get("id", ""),
            title=raw_result.get("title", ""),
            abstract=raw_result.get("abstract"),
            authors=authors,
            publication_date=publication_date,
            venue=raw_result.get("host_venue", {}).get("display_name"),
            url=raw_result.get("doi") or raw_result.get("id"),
            pdf_url=pdf_url,
            citation_count=raw_result.get("cited_by_count"),
            influential_citation_count=None,  # OpenAlex doesn't provide this
            reference_count=raw_result.get("referenced_works_count"),
            external_ids={
                "DOI": doi,
                "OpenAlex": raw_result.get("id", "")
            } if doi else {"OpenAlex": raw_result.get("id", "")},
            source=self.name
        )

    async def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get paper details by OpenAlex ID.
        
        Args:
            paper_id: OpenAlex work ID
            
        Returns:
            Paper details dictionary
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # OpenAlex IDs start with 'W' or full URL
                if not paper_id.startswith("http"):
                    paper_id = f"W{paper_id}" if not paper_id.startswith("W") else paper_id
                    url = f"{self.api_url}/works/{paper_id}"
                else:
                    url = paper_id
                
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPError as e:
            logger.error(f"[{self.name}] Error fetching paper {paper_id}: {e}")
            return None

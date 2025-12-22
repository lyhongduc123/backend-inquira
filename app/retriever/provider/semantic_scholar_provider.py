"""
Semantic Scholar retrieval provider.

Implements BaseCachedProvider for Semantic Scholar API with caching support.
"""
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.core.config import settings
from app.extensions.logger import create_logger
from .base import BaseCachedProvider, RetrievalConfig
from ..paper_repository import PaperRepository
from ..paper_schemas import Paper, Author
from ..paper_retriever import PaperRetriever
from .base_schemas import NormalizedResult, AuthorDict

logger = create_logger(__name__)


class SemanticScholarProvider(BaseCachedProvider):
    """
    Semantic Scholar retrieval provider with caching.
    
    Features:
    - Semantic relevance search
    - Open access PDF detection
    - Citation counts
    - Database caching support
    """

    def __init__(self, api_url: str, config: Optional[RetrievalConfig] = None, db_session=None):
        super().__init__(api_url, config, db_session)
        self.api_key = settings.SEMANTIC_API_KEY
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.timeout = config.timeout if config else 30.0

    async def search_papers(
        self,
        query: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search papers via Semantic Scholar API.
        
        Args:
            query: Search query
            limit: Max results (default from config)
            offset: Pagination offset
            
        Returns:
            List of raw API response dictionaries
        """
        limit = limit or self.config.max_results
        
        fields = [
            "paperId", "title", "abstract", "year", "authors",
            "venue", "publicationDate", "citationCount", "influentialCitationCount", "referenceCount", "url",
            "openAccessPdf", "isOpenAccess", "externalIds"
        ]
        
        params = {
            "query": query,
            "limit": min(limit, 20),
            "offset": offset,
            "fields": ",".join(fields)
        }
        
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/paper/search",
                    params=params,
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()
                
                results = data.get("data", [])
                logger.info(f"[{self.name}] Retrieved {len(results)} papers for: {query[:50]}...")
                return results
                
        except httpx.HTTPError as e:
            logger.error(f"[{self.name}] API error: {e}")
            return []
        except Exception as e:
            logger.error(f"[{self.name}] Search error: {e}")
            return []

    async def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed paper information by ID.
        
        Args:
            paper_id: Semantic Scholar paper ID
            
        Returns:
            Paper details dictionary or None
        """
        fields = [
            "paperId", "title", "abstract", "year", "authors",
            "venue", "publicationDate", "citationCount", "url",
            "openAccessPdf", "isOpenAccess", "externalIds", "references", "citations", "influentialCitationCount"
        ]
        
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/paper/{paper_id}",
                    params={"fields": ",".join(fields)},
                    headers=headers
                )
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPError as e:
            logger.error(f"[{self.name}] Error fetching paper {paper_id}: {e}")
            return None
        
    async def get_snippet(
        self,
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Get a brief snippet for the query from Semantic Scholar.
        
        Args:
            query: Search query 
        Returns:
            Snippet text or None
        """
        limit = 10
        
        params = {
            "query": query,
            "limit": limit,
        }
        
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/snippet/search",
                    params=params,
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()
                
                results = data.get("data", [])
                return results
                
        except httpx.HTTPError as e:
            logger.error(f"[{self.name}] API error: {e}")
            return []
        except Exception as e:
            logger.error(f"[{self.name}] Search error: {e}")
            return []
        

    def normalize_result(self, raw_result: Dict[str, Any]) -> NormalizedResult:
        """
        Normalize Semantic Scholar result to standard format.
        
        Args:
            raw_result: Raw API response
            
        Returns:
            Normalized paper dictionary
        """
        # Extract external IDs
        external_ids = raw_result.get("externalIds", {}) or {}
        arxiv_id = external_ids.get("ArXiv")
        doi = external_ids.get("DOI")
        
        # Extract open access PDF
        open_access_pdf = raw_result.get("openAccessPdf") or {}
        pdf_url = open_access_pdf.get("url") if isinstance(open_access_pdf, dict) else None
        is_open_access = raw_result.get("isOpenAccess", False) or bool(pdf_url)
        
        # Store full open access metadata if available
        open_access_metadata = None
        if isinstance(open_access_pdf, dict) and open_access_pdf.get("url"):
            open_access_metadata = {
                "url": str(open_access_pdf.get("url", "")),
                "status": str(open_access_pdf.get("status", "")),
                "license": str(open_access_pdf.get("license", ""))
            }
        
        # Extract authors
        authors_raw = raw_result.get("authors", []) or []
        authors: List[AuthorDict] = [
            AuthorDict(
                name=author.get("name", ""),
                author_id=author.get("authorId"),
                citation_count=author.get("citationCount"),
                h_index=author.get("hIndex")
            )
            for author in authors_raw
        ]
        
        return NormalizedResult(
            paper_id=raw_result.get("paperId", ""),
            title=raw_result.get("title", ""),
            abstract=raw_result.get("abstract"),
            authors=authors,
            publication_date=raw_result.get("publicationDate"),
            venue=raw_result.get("venue"),
            url=raw_result.get("url"),
            pdf_url=pdf_url,
            is_open_access=is_open_access,
            open_access_pdf=open_access_metadata,
            citation_count=raw_result.get("citationCount"),
            influential_citation_count=raw_result.get("influentialCitationCount"),
            reference_count=len(raw_result.get("references", [])) if raw_result.get("references") else None,
            external_ids={
                "DOI": doi or "",
                "ArXiv": arxiv_id or ""
            } if (doi or arxiv_id) else None,
            source=self.name
        )

    async def retrieve_full_text(
        self,
        paper_id: str,
        pdf_url: Optional[str] = None
    ) -> Optional[str]:
        """
        Retrieve full text from open access PDF.
        
        Args:
            paper_id: Paper ID
            pdf_url: Direct PDF URL
            
        Returns:
            Full text or None
        """
        # If no PDF URL provided, try to get it from paper details
        if not pdf_url:
            details = await self.get_paper_details(paper_id)
            if details:
                open_access = details.get("openAccessPdf") or {}
                pdf_url = open_access.get("url") if isinstance(open_access, dict) else None
        
        if not pdf_url:
            logger.debug(f"[{self.name}] No PDF URL for paper {paper_id}")
            return None
        
        # Use PaperRetriever for actual PDF download and extraction
        try:
            retriever = PaperRetriever()
            # text = await retriever.get_paper_text(pdf_url)
            text = ""
            return text
        except Exception as e:
            logger.error(f"[{self.name}] Error retrieving full text: {e}")
            return None

    async def is_open_access(self, paper_id: str) -> bool:
        """Check if paper has open access PDF."""
        details = await self.get_paper_details(paper_id)
        if not details:
            return False
        
        open_access = details.get("openAccessPdf") or {}
        return bool(open_access.get("url")) if isinstance(open_access, dict) else False
    
    async def get_from_cache(self, query: str) -> Optional[List[NormalizedResult]]:
        """
        Retrieve results from database cache.
        
        Note: Currently returns None as we don't have query-based caching.
        Papers are cached individually by external_id when saved.
        """
        # TODO: Implement query-based caching with a queries table
        return None

    def _db_paper_to_dict(self, db_paper) -> Dict[str, Any]:
        """Convert database paper object to dict."""
        return {
            "id": str(db_paper.paper_id),
            "title": str(getattr(db_paper, 'title', '')),
            "abstract": str(getattr(db_paper, 'abstract', '')),
            "authors": [],  # TODO: Parse from DB
            "year": getattr(db_paper, 'year', None),
            "venue": str(getattr(db_paper, 'venue', '')),
            "url": str(getattr(db_paper, 'url', '')),
            "pdf_url": str(getattr(db_paper, 'pdf_url', '')),
            "citation_count": getattr(db_paper, 'citation_count', 0),
            "source": self.name,
        }

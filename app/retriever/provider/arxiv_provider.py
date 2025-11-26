"""
arXiv retrieval provider.

Implements BaseFullTextProvider for arXiv API with full-text support.
"""
import httpx
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from app.extensions.logger import create_logger
from .base import BaseFullTextProvider, RetrievalConfig
from .base_schemas import NormalizedResult, AuthorDict
from ..paper_retriever import PaperRetriever

logger = create_logger(__name__)


class ArxivProvider(BaseFullTextProvider):
    """
    arXiv retrieval provider with full-text support.
    
    Features:
    - arXiv API search
    - PDF full-text extraction
    - Multi-source retrieval (arXiv, bioRxiv)
    - Open access by default
    """

    def __init__(self, api_url: str, config: Optional[RetrievalConfig] = None):
        super().__init__(api_url, config)
        self.timeout = config.timeout if config else 60.0
        self._paper_retriever = None

    def _get_paper_retriever(self):
        """Lazy load PaperRetriever for full-text extraction."""
        if self._paper_retriever is None:
            self._paper_retriever = PaperRetriever()
        return self._paper_retriever

    async def search_papers(
        self,
        query: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search papers via arXiv API.
        
        Args:
            query: Search query
            limit: Max results
            offset: Pagination start
            
        Returns:
            List of raw API results
        """
        limit = limit or self.config.max_results
        
        params = {
            "search_query": query,
            "start": offset,
            "max_results": limit
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(f"{self.api_url}query", params=params)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                
                results = []
                for entry in root.findall("atom:entry", ns):
                    result = {
                        "id": entry.findtext("atom:id", default="", namespaces=ns).strip(),
                        "title": entry.findtext("atom:title", default="", namespaces=ns).strip(),
                        "summary": entry.findtext("atom:summary", default="", namespaces=ns).strip(),
                        "published": entry.findtext("atom:published", default="", namespaces=ns).strip(),
                        "updated": entry.findtext("atom:updated", default="", namespaces=ns).strip(),
                        "authors": [
                            {"name": author.findtext("atom:name", default="", namespaces=ns).strip()}
                            for author in entry.findall("atom:author", ns)
                        ],
                        "links": [
                            {
                                "href": link.get("href", ""),
                                "type": link.get("type", ""),
                                "title": link.get("title", "")
                            }
                            for link in entry.findall("atom:link", ns)
                        ],
                    }
                    results.append(result)
                
                logger.info(f"[{self.name}] Retrieved {len(results)} papers for: {query[:50]}...")
                return results
                
        except httpx.HTTPError as e:
            logger.error(f"[{self.name}] API error: {e}")
            return []
        except ET.ParseError as e:
            logger.error(f"[{self.name}] XML parse error: {e}")
            return []
        except Exception as e:
            logger.error(f"[{self.name}] Search error: {e}")
            return []

    def normalize_result(self, raw_result: Dict[str, Any]) -> NormalizedResult:
        """
        Normalize arXiv result to standard format.
        
        Args:
            raw_result: Raw API response
            
        Returns:
            Normalized paper dictionary
        """
        # Extract arXiv ID from URL
        paper_id = raw_result.get("id", "")
        arxiv_id = None
        if "arxiv.org/abs/" in paper_id:
            arxiv_id = paper_id.split("arxiv.org/abs/")[-1]
        
        # Find PDF link
        pdf_url = None
        for link in raw_result.get("links", []):
            if link.get("title") == "pdf" or "pdf" in link.get("type", ""):
                pdf_url = link.get("href")
                break
        
        # If no PDF link found, construct from arXiv ID
        if not pdf_url and arxiv_id:
            retriever = self._get_paper_retriever()
            pdf_url = retriever.get_pdf_url_from_arxiv_id(arxiv_id)
        
        # Format authors
        authors: List[AuthorDict] = [
            AuthorDict(name=a.get("name", "Unknown"), author_id=None)
            for a in raw_result.get("authors", [])
        ]
        
        # Extract publication date
        published = raw_result.get("published", "")
        publication_date = published[:10] if published else None  # ISO date format
        
        return NormalizedResult(
            paper_id=paper_id,
            title=raw_result.get("title", ""),
            abstract=raw_result.get("summary"),
            authors=authors,
            publication_date=publication_date,
            venue="arXiv",
            url=paper_id,
            pdf_url=pdf_url,
            citation_count=0,  # arXiv doesn't provide citation counts
            influential_citation_count=None,
            reference_count=None,
            external_ids={"ArXiv": arxiv_id} if arxiv_id else None,
            source=self.name
        )

    async def retrieve_full_text(
        self,
        paper_id: str,
        pdf_url: Optional[str] = None
    ) -> Optional[str]:
        """
        Retrieve full text from arXiv PDF.
        
        Args:
            paper_id: arXiv paper ID or URL
            pdf_url: Direct PDF URL (optional)
            
        Returns:
            Full text string or None
        """
        retriever = self._get_paper_retriever()
        
        # Extract arXiv ID if needed
        arxiv_id = retriever.extract_arxiv_id(paper_id)
        
        # Get PDF URL if not provided
        if not pdf_url and arxiv_id:
            pdf_url = retriever.get_pdf_url_from_arxiv_id(arxiv_id)
        
        if not pdf_url:
            logger.warning(f"[{self.name}] No PDF URL for {paper_id}")
            return None
        
        try:
            text = await retriever.get_paper_text(pdf_url)
            if text:
                logger.info(f"[{self.name}] Retrieved {len(text)} chars for {paper_id}")
            return text
        except Exception as e:
            logger.error(f"[{self.name}] Full-text retrieval error: {e}")
            return None

    async def is_open_access(self, paper_id: str) -> bool:
        """All arXiv papers are open access."""
        return True

    async def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get paper details by arXiv ID.
        
        Args:
            paper_id: arXiv ID or URL
            
        Returns:
            Paper details dictionary
        """
        retriever = self._get_paper_retriever()
        arxiv_id = retriever.extract_arxiv_id(paper_id)
        
        if not arxiv_id:
            return None
        
        # Search for specific arXiv ID
        results = await self.search_papers(f"id:{arxiv_id}", limit=1)
        
        if results:
            return results[0]
        
        return None

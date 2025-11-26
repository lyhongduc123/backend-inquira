"""
Google Scholar retrieval provider.

Implements BaseRetrievalProvider for Google Scholar (via web scraping).
"""
from typing import List, Dict, Any, Optional
from app.modules.crawler import crawler
from app.extensions.logger import create_logger
from .base import BaseRetrievalProvider, RetrievalConfig
from .base_schemas import NormalizedResult, AuthorDict

logger = create_logger(__name__)


class GoogleScholarProvider(BaseRetrievalProvider):
    """
    Google Scholar retrieval provider (web scraping).
    
    Note: Uses web scraping, may be rate-limited.
    Consider using official API or proxy if available.
    """

    def __init__(self, api_url: str, config: Optional[RetrievalConfig] = None):
        super().__init__(api_url, config)

    async def search_papers(
        self,
        query: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search papers via Google Scholar scraping.
        
        Args:
            query: Search query
            limit: Max results (ignored for now)
            offset: Pagination offset
            
        Returns:
            List of scraped results
        """
        try:
            # Use the existing crawler module
            html_content = crawler.fetch_page(f"{self.api_url}/scholar?q={query}&start={offset}")
            parsed_data = crawler.parse_content(html_content)
            
            logger.info(f"[{self.name}] Retrieved {len(parsed_data)} papers for: {query[:50]}...")
            return parsed_data or []
            
        except Exception as e:
            logger.error(f"[{self.name}] Scraping error: {e}")
            return []

    def normalize_result(self, raw_result: Dict[str, Any]) -> NormalizedResult:
        """
        Normalize Scholar result to standard format.
        
        Note: Schema depends on crawler implementation.
        Adjust based on actual crawler output.
        
        Args:
            raw_result: Raw crawler output
            
        Returns:
            Normalized paper dictionary
        """
        # Parse authors
        authors = self._parse_authors(raw_result.get("authors", ""))
        
        # Extract year and create publication date
        year = self._extract_year(raw_result.get("year", raw_result.get("published", "")))
        publication_date = f"{year}-01-01" if year else None
        
        return NormalizedResult(
            paper_id=raw_result.get("id", raw_result.get("url", "")),
            title=raw_result.get("title", ""),
            abstract=raw_result.get("snippet", raw_result.get("abstract")),
            authors=authors,
            publication_date=publication_date,
            venue=raw_result.get("venue", raw_result.get("source")),
            url=raw_result.get("url"),
            pdf_url=raw_result.get("pdf_url", raw_result.get("pdf")),
            citation_count=self._parse_citations(raw_result.get("citations", 0)),
            influential_citation_count=None,  # Google Scholar doesn't provide this
            reference_count=None,
            external_ids=None,
            source=self.name
        )

    def _parse_authors(self, authors_str: Any) -> List[AuthorDict]:
        """Parse authors from string or list."""
        if isinstance(authors_str, list):
            return [AuthorDict(name=str(a), author_id=None) for a in authors_str]
        elif isinstance(authors_str, str):
            # Split by common separators
            author_names = [a.strip() for a in authors_str.replace(";", ",").split(",")]
            return [AuthorDict(name=name, author_id=None) for name in author_names if name]
        return []

    def _extract_year(self, year_input: Any) -> Optional[int]:
        """Extract year as integer."""
        if isinstance(year_input, int):
            return year_input
        elif isinstance(year_input, str):
            # Try to extract 4-digit year
            import re
            match = re.search(r'\b(19|20)\d{2}\b', year_input)
            if match:
                return int(match.group())
        return None

    def _parse_citations(self, citations_input: Any) -> int:
        """Parse citation count."""
        if isinstance(citations_input, int):
            return citations_input
        elif isinstance(citations_input, str):
            # Extract number from string like "Cited by 123"
            import re
            match = re.search(r'\d+', citations_input)
            if match:
                return int(match.group())
        return 0

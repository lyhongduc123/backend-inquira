"""
Semantic Scholar retrieval provider.

Implements BaseRetrievalProvider for Semantic Scholar API.
"""

import httpx
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.extensions.logger import create_logger
from app.retriever.schemas import NormalizedResult, AuthorSchema
from .base import BaseRetrievalProvider, RetrievalConfig
from ..paper_retriever import PaperRetriever

logger = create_logger(__name__)


class SemanticScholarProvider(BaseRetrievalProvider):
    """
    Semantic Scholar retrieval provider.

    Features:
    - Semantic relevance search
    - Open access PDF detection
    - Author h-index and citation metrics
    - Influential citation counts
    """

    def __init__(
        self,
        api_url: str,
        config: Optional[RetrievalConfig] = None,
    ):
        super().__init__(api_url, config)
        self.api_key = settings.SEMANTIC_API_KEY

    async def search_papers(
        self,
        query: str,
        limit: Optional[int] = None,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search papers via Semantic Scholar API.

        Args:
            query: Search query
            limit: Max results (default from config)
            offset: Pagination offset
            filters: Optional filters (yearRange, category, openAccessOnly, excludePreprints, topJournalsOnly)

        Returns:
            List of raw API response dictionaries
        """
        limit = limit or self.config.max_results

        fields = [
            "paperId",
            "title",
            "abstract",
            "year",
            "authors",
            "authors.citationCount",
            "authors.hIndex",
            "authors.paperCount",
            "authors.url",
            "venue",
            "publicationDate",
            "citationCount",
            "influentialCitationCount",
            "referenceCount",
            "url",
            "isOpenAccess",
            "openAccessPdf",
            "citationStyles",
            "externalIds",
        ]

        params = {
            "query": query,
            "limit": min(limit, 20),
            "offset": offset,
            "fields": ",".join(fields),
        }

        if filters:
            if "yearRange" in filters and filters["yearRange"]:
                year_range = filters["yearRange"]
                if "min" in year_range and year_range["min"]:
                    params["year"] = f"{year_range['min']}-"
                if "max" in year_range and year_range["max"]:
                    if "year" in params:
                        params["year"] = f"{year_range['min']}-{year_range['max']}"
                    else:
                        params["year"] = f"-{year_range['max']}"
            if "category" in filters and filters["category"]:
                params["fieldsOfStudy"] = ",".join(filters["category"])
            if filters.get("openAccessOnly"):
                params["isOpenAccess"] = ""

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(
                    f"{self.api_url}/paper/search", params=params, headers=headers
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

    async def get_paper_details(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed paper information by ID.

        Args:
            paper_id: Semantic Scholar paper ID

        Returns:
            Paper details dictionary or None
        """
        fields = [
            "paperId",
            "title",
            "abstract",
            "year",
            "authors",
            "authors.citationCount",
            "authors.hIndex",
            "authors.paperCount",
            "authors.url",
            "venue",
            "publicationDate",
            "citationCount",
            "url",
            "openAccessPdf",
            "isOpenAccess",
            "externalIds",
            "references",
            "citations",
            "influentialCitationCount",
        ]

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(
                    f"{self.api_url}/paper/{paper_id}",
                    params={"fields": ",".join(fields)},
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.error(f"[{self.name}] Error fetching paper {paper_id}: {e}")
            return None

    async def get_snippet(self, query: str) -> List[Dict[str, Any]]:
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
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(
                    f"{self.api_url}/snippet/search", params=params, headers=headers
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

        # Extract open access PDF
        open_access_pdf = raw_result.get("openAccessPdf") or {}
        pdf_url = (
            open_access_pdf.get("url") if isinstance(open_access_pdf, dict) else None
        )
        is_open_access = raw_result.get("isOpenAccess", False) or bool(pdf_url)

        # Store full open access metadata if available
        open_access_metadata = None
        if isinstance(open_access_pdf, dict) and open_access_pdf.get("url"):
            open_access_metadata = {
                "url": str(open_access_pdf.get("url", "")),
                "status": str(open_access_pdf.get("status", "")),
                "license": str(open_access_pdf.get("license", "")),
            }

        # Extract authors
        authors_raw = raw_result.get("authors", []) or []
        authors: List[AuthorSchema] = [
            AuthorSchema(
                name=author.get("name", ""),
                author_id=author.get("authorId"),
                citation_count=author.get("citationCount"),
                h_index=author.get("hIndex"),
                paper_count=author.get("paperCount"),
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
            reference_count=(
                len(raw_result.get("references", []))
                if raw_result.get("references")
                else None
            ),
            citation_styles=raw_result.get("citationStyles"),
            external_ids=external_ids,
            source=self.name,
        )

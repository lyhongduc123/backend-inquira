"""
Semantic Scholar retrieval provider.

Implements BaseRetrievalProvider for Semantic Scholar API.
"""

from urllib.parse import urlencode

import httpx
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.extensions.logger import create_logger
from app.retriever.schemas import NormalizedPaperResult, AuthorSchema
from .base import BaseRetrievalProvider, RetrievalConfig
from ..schemas import (
    S2AuthorPapersResponse,
    S2PaperCitationsResponse,
    S2PaperReferencesResponse,
)

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
            filters: Optional filters (year range, category, open access)

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
            "authors.name",
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
            "tldr",  # NEW: TL;DR summary
            "fieldsOfStudy",  # NEW: Basic fields of study
            "s2FieldsOfStudy",  # NEW: Enriched fields with source
            "references",  # NEW: References list
            "references.paperId",  # NEW: Reference paper IDs
            "references.title",  # NEW: Reference titles
        ]

        params = {
            "query": query,
            "limit": min(limit, 100),
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
            raise e
        except Exception as e:
            logger.error(f"[{self.name}] Search error: {e}")
            raise e

    async def get_bulk_paper(
        self,
        query: str,
        token: Optional[str] = None,
        fields_of_study: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get bulk paper details for a search query.

        Args:
            query: Search query
            token: Continuation token for pagination
            fields_of_study: Optional list of fields of study to filter

        Returns:
            List of paper details dictionaries
        """

        fields = [
            "paperId",
            "title",
            "abstract",
            "year",
            "authors",
            "authors.name",
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
            "fieldsOfStudy",
            "s2FieldsOfStudy",
        ]

        params = {
            "query": query,
            "fields": ",".join(fields),
            "sort": "citationCount:desc",
        }

        if token:
            params["token"] = token

        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        query_str = urlencode(params) + "&openAccessPdf"

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(
                    f"{self.api_url}/paper/search/bulk",
                    params=query_str,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

                results = data.get("data", [])
                return results

        except httpx.HTTPError as e:
            logger.error(f"[{self.name}] API error: {e}")
            raise e
        except Exception as e:
            logger.error(f"[{self.name}] Search error: {e}")
            raise e

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
            "authors.name",
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

    async def get_multiple_papers_details(
        self, paper_ids: List[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get details for multiple papers by their IDs.

        Args:
            paper_ids: List of Semantic Scholar paper IDs
        Returns:
            List of paper details dictionaries
        """
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
            "authors.name",
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
            "tldr",
            "fieldsOfStudy",
            "s2FieldsOfStudy",
            "citationStyles",
        ]

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    f"{self.api_url}/paper/batch",
                    params={"fields": ",".join(fields)},
                    json={"ids": paper_ids},
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"[{self.name}] Error fetching batch papers: {e}")
            try:
                logger.error(f"Response content: {e.response.json()}")
            except Exception:
                logger.error(f"Response content: {e.response.text}")
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

    async def get_citations(
        self,
        paper_id: str,
        offset: int = 0,
        limit: int = 100,
        fields: Optional[str] = None,
    ) -> S2PaperCitationsResponse:
        """
        Get papers that cite this paper.

        Args:
            paper_id: S2 paper ID
            offset: Pagination offset
            limit: Max results (up to 1000)
            fields: Optional comma-separated S2 fields

        Returns:
            {
                "offset": 0,
                "next": 100,
                "data": [...citing papers...]
            }
        """
        default_fields = "paperId,corpusId,title,abstract,authors,year,citationCount,venue,isInfluential,contexts,intents"
        field_param = fields or default_fields

        params = {"offset": offset, "limit": limit, "fields": field_param}

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(
                    f"{self.api_url}/paper/{paper_id}/citations",
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()
                json_response = response.json()
                return S2PaperCitationsResponse(
                    offset=offset,
                    next=json_response.get("next", 0),
                    data=json_response.get("data", []),
                )

        except httpx.HTTPError as e:
            logger.error(f"[{self.name}] Error fetching citations for {paper_id}: {e}")
            return S2PaperCitationsResponse(offset=offset, next=0, data=[])

    async def get_references(
        self,
        paper_id: str,
        offset: int = 0,
        limit: int = 100,
        fields: Optional[str] = None,
    ) -> S2PaperReferencesResponse:
        """
        Get papers referenced by this paper.

        Args:
            paper_id: S2 paper ID
            offset: Pagination offset
            limit: Max results (up to 1000)
            fields: Optional comma-separated S2 fields

        Returns:
            {
                "offset": 0,
                "next": 100,
                "data": [...referenced papers...]
            }
        """
        default_fields = "paperId,corpusId,title,abstract,authors,year,citationCount,venue,isInfluential,contexts,intents"
        field_param = fields or default_fields

        params = {"offset": offset, "limit": limit, "fields": field_param}

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(
                    f"{self.api_url}/paper/{paper_id}/references",
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()
                json_response = response.json()
                return S2PaperReferencesResponse(
                    offset=offset,
                    next=json_response.get("next", 0),
                    data=json_response.get("data", []),
                )

        except httpx.HTTPError as e:
            logger.error(f"[{self.name}] Error fetching references for {paper_id}: {e}")
            return S2PaperReferencesResponse(offset=offset, next=0, data=[])
        
    async def get_multiple_authors(self, author_ids: list[str]) -> Optional[Dict[str, Any]]:
        """
        Get details for multiple authors by their IDs.

        Args:
            author_ids: List of Semantic Scholar author IDs
        Returns:
            Dictionary mapping author IDs to their details
        """
        fields = [
            "authorId",
            "externalIds",
            "name",
            "citationCount",
            "hIndex",
            "paperCount",
            "url",
        ]

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
            
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    f"{self.api_url}/author/batch",
                    params={"fields": ",".join(fields)},
                    json={"ids": author_ids},
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                return {author["authorId"]: author for author in data.get("data", [])}
        except httpx.HTTPStatusError as e:
            logger.error(f"[{self.name}] Error fetching batch authors: {e}")
            try:
                logger.error(f"Response content: {e.response.json()}")
            except Exception:
                logger.error(f"Response content: {e.response.text}")
            return None

    async def get_author_papers(
        self,
        author_id: str,
        offset: int = 0,
        limit: int = 100,
        fields: Optional[str] = None,
    ) -> S2AuthorPapersResponse:
        """
        Get papers by an author from Semantic Scholar API.

        Args:
            author_id: Semantic Scholar author ID
            offset: Pagination offset
            limit: Max results per page (max 1000 per S2 API)
            fields: Comma-separated field list

        Returns:
            S2AuthorPapersResponse containing author papers and pagination info:
            {
                "offset": 0,
                "next": 100,
                "data": [...papers...]
            }
        """
        default_fields = [
            "paperId",
            "title",
            "abstract",
            "authors",
            "authors.name",
            "year",
            "venue",
            "publicationDate",
            "citationCount",
            "url",
            "openAccessPdf",
            "isOpenAccess",
            "externalIds",
            "influentialCitationCount",
            "citationStyles",
        ]
        params_fields = fields.split(",") if fields else default_fields

        params = {
            "offset": offset,
            "limit": min(limit, 100),
            "fields": ",".join(params_fields),
        }

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.get(
                    f"{self.api_url}/author/{author_id}/papers",
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
                return S2AuthorPapersResponse(
                    offset=offset, next=data.get("next"), data=data.get("data", [])
                )

        except httpx.HTTPError as e:
            logger.error(
                f"[{self.name}] Error fetching papers for author {author_id}: {e}"
            )
            return S2AuthorPapersResponse(offset=offset, next=None, data=[])

    def normalize_result(self, raw_result: Dict[str, Any]) -> NormalizedPaperResult:
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
                homepage_url=author.get("url"),
                url=author.get("url"),
            )
            for author in authors_raw
        ]

        # Extract TLDR
        tldr_data = raw_result.get("tldr")
        tldr = None
        if tldr_data and isinstance(tldr_data, dict):
            text = tldr_data.get("text")
            # Only create tldr dict if text has an actual value
            if text:
                tldr = {"model": tldr_data.get("model"), "text": text}

        # Extract year
        year = raw_result.get("year")

        # Extract fields of study
        fields_of_study = raw_result.get("fieldsOfStudy", [])
        s2_fields_of_study = raw_result.get("s2FieldsOfStudy", [])

        # Extract references
        references = raw_result.get("references", [])

        return NormalizedPaperResult(
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
            # Semantic Scholar specific fields
            tldr=tldr,
            year=year,
            fields_of_study=fields_of_study if fields_of_study else None,
            s2_fields_of_study=s2_fields_of_study if s2_fields_of_study else None,
            references=references if references else None,
        )

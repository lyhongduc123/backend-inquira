"""
OpenAlex retrieval provider.

Implements BaseRetrievalProvider for OpenAlex API.
"""

from app.retriever.schemas.openalex import OAAuthorResponse
import httpx
from typing import List, Dict, Any, Optional
from app.extensions.logger import create_logger
from app.retriever.schemas import (
    NormalizedPaperResult,
    AuthorSchema,
    NormalizedAuthorResult,
)
from app.utils.identifier_normalization import normalize_external_ids
from .base import BaseRetrievalProvider, RetrievalConfig

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
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search papers via OpenAlex API. Provides keyword-based search.

        Args:
            query: Search query
            limit: Max results
            offset: Pagination offset
            filters: Optional filters (not used by OpenAlex provider)

        Returns:
            List of raw API results
        """
        limit = limit or self.config.max_results

        params = {
            "search": query,
            "per-page": min(limit, 200),
            "page": (offset // limit) + 1 if limit > 0 else 1,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.api_url}/works", params=params)
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                logger.info(
                    f"[{self.name}] Retrieved {len(results)} papers for: {query[:50]}..."
                )
                return results

        except httpx.HTTPError as e:
            logger.error(f"[{self.name}] API error: {e}")
            raise e
        except Exception as e:
            logger.error(f"[{self.name}] Search error: {e}")
            raise e
        
    # async def search_semantic_papers(
    #     self,
    #     query: str,
    #     limit: Optional[int] = None,
    #     offset: int = 0,
    #     filters: Optional[Dict[str, Any]] = None,
    # ) -> List[Dict[str, Any]]:
    #     limit = limit or self.config.max_results

    #     params = {
    #         "search": query,
    #         "per-page": min(limit, 200),
    #         "page": (offset // limit) + 1 if limit > 0 else 1,
    #     }

    #     try:
    #         async with httpx.AsyncClient(timeout=self.timeout) as client:
    #             response = await client.get(f"{self.api_url}/works?sort=", params=params)
    #             response.raise_for_status()
    #             data = response.json()

    #             results = data.get("results", [])
    #             logger.info(
    #                 f"[{self.name}] Retrieved {len(results)} papers for: {query[:50]}..."
    #             )
    #             return results

    #     except httpx.HTTPError as e:
    #         logger.error(f"[{self.name}] API error: {e}")
    #         raise e
    #     except Exception as e:
    #         logger.error(f"[{self.name}] Search error: {e}")
    #         raise e
        
        
    async def get_papers_by_dois(self, dois: List[str], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get multiple papers by their DOIs.

        Args:
            dois: List of DOIs

        Returns:
            List of normalized paper results
        """
        results = []

        try:
            dois_str = "|".join([doi for doi in dois if doi])
            params = {"filter": f"doi:{dois_str}", "per-page": min(limit, 100) if limit else 100}
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.api_url}/works", params=params)
                response.raise_for_status()
                data = response.json()
                items = data.get("results", [])
                return items

        except httpx.HTTPError as e:
            logger.error(f"[{self.name}] API error for DOI {dois}: {e}")
        except Exception as e:
            logger.error(f"[{self.name}] Error fetching DOI {dois}: {e}")

        return results

    async def get_institutions_by_ids(
        self, institution_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get multiple institutions by their OpenAlex IDs.
        Args:
            institution_ids: List of OpenAlex institution IDs
        Returns:
            List of institution details
        results = []
        """
        results = []
        try:
            ids_str = "|".join([inst_id for inst_id in institution_ids if inst_id])
            params = {"filter": f"id:{ids_str}"}
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.api_url}/institutions", params=params
                )
                response.raise_for_status()
                data = response.json()
                items = data.get("results", [])
                return items

        except httpx.HTTPError as e:
            logger.error(
                f"[{self.name}] API error for Institution IDs {institution_ids}: {e}"
            )
        except Exception as e:
            logger.error(
                f"[{self.name}] Error fetching Institution IDs {institution_ids}: {e}"
            )

        return results

    async def get_multiple_authors(self, author_ids: List[str], limit: Optional[int] = None) -> list[OAAuthorResponse]:
        """
        Get multiple authors by their OpenAlex IDs.
        Args:
            author_ids: List of OpenAlex author IDs
        Returns:
            List of institution details
        results = []
        """
        results = []
        try:
            ids_str = "|".join(author_ids)
            params = {"filter": f"ids.openalex:{ids_str}", "per-page": min(limit, 100) if limit else 100}
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.api_url}/authors", params=params
                )
                response.raise_for_status()
                data = response.json()
                items = data.get("results", [])
                return [OAAuthorResponse(**item) for item in items]

        except httpx.HTTPError as e:
            logger.error(
                f"[{self.name}] API error for Author IDs {author_ids}: {e}"
            )
        except Exception as e:
            logger.error(
                f"[{self.name}] Error fetching Author IDs {author_ids}: {e}"
            )

        return []

    def normalize_result(self, raw_result: Dict[str, Any]) -> NormalizedPaperResult:
        """
        Normalize OpenAlex result to standard format.

        OpenAlex provides rich metadata including:
        - Detailed author information with institutions and affiliations
        - Topics, keywords, concepts with relevance scores
        - Citation metrics including percentiles and FWCI
        - MeSH terms for biomedical papers
        - Multiple publication locations and open access info

        Args:
            raw_result: Raw OpenAlex API response

        Returns:
            Normalized paper dictionary with comprehensive metadata
        """
        authorships = raw_result.get("authorships", []) or []
        authors: List[AuthorSchema] = []

        for auth in authorships:
            author_info = auth.get("author", {}) or {}
            author_dict = AuthorSchema(
                name=author_info.get("display_name", "Unknown"),
                author_id=author_info.get("id").removeprefix("https://openalex.org/") if author_info.get("id") else None,  # type: ignore
                orcid=author_info.get("orcid"),
                institutions=auth.get("institutions", []),
                affiliations=auth.get("affiliations", []),
            )
            authors.append(author_dict)

        # Extract IDs
        ids = raw_result.get("ids", {}) or {}
        doi = (
            ids.get("doi", "").replace("https://doi.org/", "")
            if ids.get("doi")
            else None
        )
        pmid = (
            ids.get("pmid", "").replace("https://pubmed.ncbi.nlm.nih.gov/", "")
            if ids.get("pmid")
            else None
        )

        # Build external IDs dictionary
        external_ids = {}
        if doi:
            external_ids["DOI"] = doi
        if pmid:
            external_ids["PubMed"] = pmid
        if ids.get("mag"):
            external_ids["MAG"] = str(ids.get("mag"))
        external_ids["OpenAlex"] = raw_result.get("id", "")

        # Extract open access info
        open_access = raw_result.get("open_access", {}) or {}
        is_oa = open_access.get("is_oa", False)
        pdf_url = open_access.get("oa_url")

        open_access_metadata = None
        if is_oa and pdf_url:
            open_access_metadata = {
                "url": str(pdf_url),
                "status": str(open_access.get("oa_status", "")),
                "license": "",
            }

        primary_location = raw_result.get("primary_location", {}) or {}
        primary_source = primary_location.get("source", {}) or {}
        venue = primary_source.get("display_name")
        year = raw_result.get("publication_year")
        pub_date = raw_result.get("publication_date")
        if not pub_date and year:
            pub_date = f"{year}-01-01"

        citation_count = raw_result.get("cited_by_count", 0)
        citation_percentile = raw_result.get("citation_normalized_percentile")
        topics = raw_result.get("topics", [])
        keywords = raw_result.get("keywords", [])
        concepts = raw_result.get("concepts", [])
        has_content = raw_result.get("has_content", {})

        mesh_terms = raw_result.get("mesh", [])
        biblio = raw_result.get("biblio", {})
        fwci = raw_result.get("fwci")
        is_retracted = raw_result.get("is_retracted", False)
        language = raw_result.get("language")

        # Most likely null, but include for completeness
        corresponding_author_ids = raw_result.get("corresponding_author_ids", [])
        institutions_distinct_count = raw_result.get("institutions_distinct_count")
        countries_distinct_count = raw_result.get("countries_distinct_count")

        locations = raw_result.get("locations", [])
        best_oa_location = raw_result.get("best_oa_location", {})

        url = raw_result.get("doi") or raw_result.get("id")
        external_ids = normalize_external_ids(external_ids)

        return NormalizedPaperResult(
            paper_id=raw_result.get("id", ""),
            title=raw_result.get("title", "") or raw_result.get("display_name", ""),
            abstract=raw_result.get("abstract"),
            authors=authors,
            publication_date=pub_date,
            venue=venue,
            url=url,
            pdf_url=pdf_url,
            is_open_access=is_oa,
            open_access_pdf=open_access_metadata,
            citation_count=citation_count,
            influential_citation_count=None,
            reference_count=raw_result.get("referenced_works_count"),
            external_ids=external_ids,
            source=self.name,
            # OpenAlex-specific fields
            topics=topics,
            keywords=keywords,
            concepts=concepts,
            mesh_terms=mesh_terms,
            citation_percentile=citation_percentile,
            has_content=has_content,
            fwci=fwci,
            is_retracted=is_retracted,
            language=language,
            biblio=biblio,
            primary_location=primary_location,
            locations=locations,
            best_oa_location=best_oa_location,
            corresponding_author_ids=corresponding_author_ids,
            institutions_distinct_count=institutions_distinct_count,
            countries_distinct_count=countries_distinct_count,
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
                    paper_id = (
                        f"W{paper_id}" if not paper_id.startswith("W") else paper_id
                    )
                    url = f"{self.api_url}/works/{paper_id}"
                else:
                    url = paper_id

                response = await client.get(url)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.error(f"[{self.name}] Error fetching paper {paper_id}: {e}")
            return None

    async def get_author_works(
        self, author_id: str, page: int = 1, per_page: int = 100
    ) -> Dict[str, Any]:
        """
        Get works (papers) by an author from OpenAlex API.

        Args:
            author_id: OpenAlex author ID (e.g., A5023888391)
            page: Page number (1-indexed)
            per_page: Results per page (max 200)

        Returns:
            Dict containing author works and pagination info:
            {
                "results": [...papers...],
                "meta": {"count": 60, "page": 1, "per_page": 100}
            }
        """
        try:
            # Ensure author_id has 'A' prefix
            if not author_id.startswith("A") and not author_id.startswith("http"):
                author_id = f"A{author_id}"

            params = {
                "filter": f"author.id:{author_id}",
                "per-page": min(per_page, 200),
                "page": page,
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.api_url}/works", params=params)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.error(
                f"[{self.name}] Error fetching works for author {author_id}: {e}"
            )
            return {"results": [], "meta": {"count": 0}}

    async def get_author_details(self, author_id: str) -> Optional[Dict[str, Any]]:
        """
        Get author details by OpenAlex Author ID.

        Args:
            author_id: OpenAlex author ID (e.g., A5023888391)

        Returns:
            Author details dictionary
        """
        try:
            # Ensure author_id has 'A' prefix
            if not author_id.startswith("A") and not author_id.startswith("http"):
                author_id = f"A{author_id}"

            url = f"{self.api_url}/authors/{author_id}"

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.error(f"[{self.name}] Error fetching author {author_id}: {e}")
            return None

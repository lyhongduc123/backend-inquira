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
        Search papers via OpenAlex API. Provides keyword-based search.
        
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
        # Extract authorships with detailed information
        authorships = raw_result.get("authorships", []) or []
        authors: List[AuthorDict] = []
        
        for auth in authorships:
            author_info = auth.get("author", {}) or {}
            author_dict = AuthorDict(
                name=author_info.get("display_name", "Unknown"),
                author_id=author_info.get("id"),
                orcid=author_info.get("orcid"),
                institutions=auth.get("institutions", []),
                affiliations=auth.get("affiliations", [])
            )
            authors.append(author_dict)
        
        # Extract IDs
        ids = raw_result.get("ids", {}) or {}
        doi = ids.get("doi", "").replace("https://doi.org/", "") if ids.get("doi") else None
        pmid = ids.get("pmid", "").replace("https://pubmed.ncbi.nlm.nih.gov/", "") if ids.get("pmid") else None
        
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
        
        # Build open access metadata
        open_access_metadata = None
        if is_oa and pdf_url:
            open_access_metadata = {
                "url": str(pdf_url),
                "status": str(open_access.get("oa_status", "")),
                "license": ""  # OpenAlex doesn't provide license in open_access field
            }
        
        # Extract primary location (journal/venue info)
        primary_location = raw_result.get("primary_location", {}) or {}
        primary_source = primary_location.get("source", {}) or {}
        venue = primary_source.get("display_name")
        
        # Extract publication date
        year = raw_result.get("publication_year")
        pub_date = raw_result.get("publication_date")
        if not pub_date and year:
            pub_date = f"{year}-01-01"
        
        # Extract citation metrics
        citation_count = raw_result.get("cited_by_count", 0)
        
        # Extract citation percentile information
        citation_percentile = raw_result.get("citation_normalized_percentile")
        
        # Extract topics (OpenAlex provides scored research topics)
        topics = raw_result.get("topics", [])
        
        # Extract keywords (with scores)
        keywords = raw_result.get("keywords", [])
        
        # Extract concepts (hierarchical with scores and levels)
        concepts = raw_result.get("concepts", [])
        
        # Extract MeSH terms for biomedical papers
        mesh_terms = raw_result.get("mesh", [])
        
        # Extract bibliographic info
        biblio = raw_result.get("biblio", {})
        
        # Extract FWCI (field-weighted citation impact)
        fwci = raw_result.get("fwci")
        
        # Extract retraction status
        is_retracted = raw_result.get("is_retracted", False)
        
        # Extract language
        language = raw_result.get("language")
        
        # Extract author collaboration metadata
        corresponding_author_ids = raw_result.get("corresponding_author_ids", [])
        institutions_distinct_count = raw_result.get("institutions_distinct_count")
        countries_distinct_count = raw_result.get("countries_distinct_count")
        
        # Extract all locations (for multi-venue publications)
        locations = raw_result.get("locations", [])
        
        # Determine URL
        url = raw_result.get("doi") or raw_result.get("id")
        
        return NormalizedResult(
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
            influential_citation_count=None,  # OpenAlex doesn't provide this (Semantic Scholar only)
            reference_count=raw_result.get("referenced_works_count"),
            external_ids=external_ids,
            source=self.name,
            # OpenAlex-specific fields
            topics=topics,
            keywords=keywords,
            concepts=concepts,
            mesh_terms=mesh_terms,
            citation_percentile=citation_percentile,
            fwci=fwci,
            is_retracted=is_retracted,
            language=language,
            biblio=biblio,
            primary_location=primary_location,
            locations=locations,
            corresponding_author_ids=corresponding_author_ids,
            institutions_distinct_count=institutions_distinct_count,
            countries_distinct_count=countries_distinct_count
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

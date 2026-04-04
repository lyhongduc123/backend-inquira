from typing import (
    TypeVar,
    Type,
    List,
    Literal,
    Optional,
    Dict,
    Any,
    Tuple,
    Union,
    TYPE_CHECKING,
)
from enum import Enum
from litellm import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.chunks.schemas import ChunkRetrieved
from app.extensions.logger import create_logger
from app.core.config import settings
from app.retriever.provider import (
    SemanticScholarProvider,
    OpenAlexProvider,
    RetrievalConfig,
    RetrievalProvider,
)
from app.retriever.schemas import NormalizedPaperResult
from app.processor.services.embeddings import EmbeddingService, get_embedding_service
from app.domain.chunks import ChunkService, ChunkRepository
from app.core.dtos.paper import PaperEnrichedDTO
from app.utils.transformers import batch_normalized_to_papers
from app.utils.string import surname

from .retriever import PaperRetriever
from .result_logger import save_retrieval_results
from .schemas.openalex import OAAuthorResponse

if TYPE_CHECKING:
    from app.domain.papers.service import PaperService

logger = create_logger(__name__)


class RetrievalServiceType(str, Enum):
    """Supported retrieval service types"""

    SEMANTIC = "semantic"
    OPENALEX = "openalex"


@dataclass
class ResolvedPaperContent:
    kind: Literal["tei_xml", "pdf_bytes"]
    content: Union[str, bytes]


class RetrievalService:
    """
    Unified retrieval service.

    Handles:
    - Multiple providers (Semantic Scholar, OpenAlex)
    - Hybrid search combining semantic relevance with metadata enrichment
    - Paper deduplication and merging
    - Author metadata enrichment
    """

    def __init__(
        self,
        db: AsyncSession,
        paper_retriever: Optional[PaperRetriever] = None,
        embedding_service: Optional[EmbeddingService] = None,
        chunk_service: Optional[ChunkService] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        """
        Initialize RetrievalService with dependency injection.

        Args:
            db: Database session
            paper_retriever: Optional paper retriever (for fetching from external APIs)
            embedding_service: Optional embedding service
            chunk_service: Optional chunk service (uses repository internally)
            config: Optional retrieval configuration
        """
        self.paper_retriever = paper_retriever or PaperRetriever()
        self.embedding_service = embedding_service or get_embedding_service()
        if chunk_service:
            self.chunk_service = chunk_service
        else:
            chunk_repository = ChunkRepository(db)
            self.chunk_service = ChunkService(chunk_repository)

        retrieval_config = config or RetrievalConfig(max_results=100, timeout=30.0)
        self.providers: Dict[RetrievalServiceType, RetrievalProvider] = {
            RetrievalServiceType.SEMANTIC: SemanticScholarProvider(
                api_url=settings.SEMANTIC_API_URL,
                config=retrieval_config,
            ),
            RetrievalServiceType.OPENALEX: OpenAlexProvider(
                api_url=settings.OPENALEX_API_URL, config=retrieval_config
            ),
        }

    async def search(
        self,
        query: str,
        limit: int,
        services: List[RetrievalServiceType],
        save_results: bool = False,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[PaperEnrichedDTO]:
        """
        Search for papers across specified services

        Args:
            query: Search query
            limit: Number of papers to retrieve per service
            services: List of retrieval services to use
            save_results: If True, save raw retrieval results to JSON for debugging
            filters: Optional filters (yearRange, category, openAccessOnly, excludePreprints, topJournalsOnly)

        Returns:
            List of Paper objects
        """
        results: List[NormalizedPaperResult] = []
        for service_type in services:
            provider = self.providers.get(service_type)
            if not provider:
                logger.warning(f"Provider for service {service_type} not found")
                continue

            try:
                service_papers = await provider.search_and_normalize(
                    query, limit, filters=filters
                )
                results.extend(service_papers)
                logger.info(
                    f"Retrieved {len(service_papers)} papers from {service_type}"
                )
            except Exception as e:
                logger.error(f"Error retrieving papers from {service_type}: {e}")

        # Optionally save raw results for analysis
        if save_results and results:
            try:
                save_retrieval_results(results, query=query, provider=str(services))
            except Exception as e:
                logger.warning(f"Failed to save retrieval results: {e}")

        papers = batch_normalized_to_papers(results)
        return papers

    async def hybrid_search(
        self,
        query: str,
        semantic_limit: int = 50,
        final_limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        enable_enrichment: bool = True,
    ) -> Tuple[List[PaperEnrichedDTO], Dict[str, Any]]:
        """
        Hybrid search combining Semantic Scholar semantic search with OpenAlex metadata enrichment.

        Workflow:
        1. Semantic Scholar semantic search (better query understanding)
        2. Extract DOIs/OpenAlex IDs from results
        3. Batch fetch OpenAlex metadata to enrich papers (FWCI, institutions, topics)
        4. Optional: OpenAlex keyword search for additional results
        5. Merge and deduplicate by DOI
        6. Return combined results with comprehensive metadata

        Args:
            query: Search query
            semantic_limit: Max results from Semantic Scholar (default 100)
            openalex_limit: Max additional results from OpenAlex keyword search (default 50)
            final_limit: Max final results to return (default 100)
            filters: Optional filters (year_min, year_max, fields)
            enable_enrichment: Whether to enrich with OpenAlex metadata

        Returns:
            Tuple of (papers, metadata)
            - papers: List of Paper objects with enriched metadata
            - metadata: Search metadata (counts, sources, etc.)
        """
        logger.info(f"[HybridSearch] Starting hybrid search for: {query[:50]}...")
        semantic_provider = self.providers.get(RetrievalServiceType.SEMANTIC)
        semantic_results = []

        if semantic_provider:
            try:
                logger.info(
                    f"[HybridSearch] Fetching {semantic_limit} papers from Semantic Scholar..."
                )
                raw_semantic = await semantic_provider.search_papers(
                    query, limit=semantic_limit, filters=filters
                )
                semantic_results = [
                    semantic_provider.normalize_result(r) for r in raw_semantic
                ]
                logger.info(
                    f"[HybridSearch] Retrieved {len(semantic_results)} papers from Semantic Scholar"
                )
            except Exception as e:
                logger.error(f"[HybridSearch] Semantic Scholar search error: {e}")
                raise e

        enriched_papers = []
        if enable_enrichment and semantic_results:
            enriched_papers = await self._enrich_with_openalex(semantic_results)
            logger.info(
                f"[HybridSearch] Enriched {len(enriched_papers)}/{len(semantic_results)} papers with OpenAlex metadata"
            )
        else:
            enriched_papers = semantic_results

        papers = batch_normalized_to_papers(enriched_papers)

        metadata = {
            "semantic_scholar_count": len(semantic_results),
            "openalex_enriched_count": len(enriched_papers),
            "final_returned": len(papers),
        }
        return papers, metadata

    async def get_multiple_papers(self, paper_ids: List[str]) -> List[PaperEnrichedDTO]:
        semantic_provider = self.get_provider_as(
            RetrievalServiceType.SEMANTIC, SemanticScholarProvider
        )

        normalized_results = []
        try:
            result = await semantic_provider.get_multiple_papers_details(paper_ids)
            if result:
                normalized_results = [
                    semantic_provider.normalize_result(r) for r in result
                ]
        except Exception as e:
            logger.error(f"Error fetching papers: {e}")
            return []

        enriched_papers = await self._enrich_with_openalex(normalized_results)
        papers = batch_normalized_to_papers(enriched_papers)

        return papers

    async def get_paper_citations(
        self, paper_id: str, limit: int= 100, offset: int = 0
    ) -> Dict[str, Any]:
        semantic_provider = self.get_provider_as(
            RetrievalServiceType.SEMANTIC, SemanticScholarProvider
        )
       
        result = await semantic_provider.get_citations(
            paper_id, limit=limit, offset=offset
        )
        return result.model_dump() if result else {}
    
    async def get_paper_references(
        self, paper_id: str, limit: int= 100, offset: int = 0
    ) -> Dict[str, Any]:
        semantic_provider = self.get_provider_as(
            RetrievalServiceType.SEMANTIC, SemanticScholarProvider
        )
       
        result = await semantic_provider.get_references(
            paper_id, limit=limit, offset=offset
        )
        logger.debug(f"Fetched {len(result.data) if result else 0} references for paper {paper_id}")
        return result.model_dump() if result else {}
   

    async def get_author(self, oa_id: str) -> Optional[OAAuthorResponse]:
        openalex_provider: OpenAlexProvider = self.get_provider_as(
            RetrievalServiceType.OPENALEX, OpenAlexProvider
        )
        try:
            raw_author = await openalex_provider.get_author_details(oa_id)
            if not raw_author:
                return None

            return OAAuthorResponse(**raw_author)
        except Exception as e:
            logger.error(f"Error fetching OpenAlex author {oa_id}: {e}")
            return None

    async def get_author_papers(self, author_id: str) -> List[PaperEnrichedDTO]:
        """
        Fetch detailed author information from specified provider.

        Args:
            author_id: Unique author identifier (e.g., Semantic Scholar ID)
        Returns:
            List of enriched Paper objects for the author
        """
        semantic_provider: SemanticScholarProvider = self.providers[RetrievalServiceType.SEMANTIC]  # type: ignore[assignment]

        normalized_results = []
        try:
            result = await semantic_provider.get_author_papers(author_id)
            if result:
                normalized_results = [
                    semantic_provider.normalize_result(r) for r in result.data
                ]
        except Exception as e:
            logger.error(f"Error fetching Semantic Scholar author {author_id}: {e}")
            return []

        enriched_papers = await self._enrich_with_openalex(normalized_results)
        papers = batch_normalized_to_papers(enriched_papers)

        return papers

    async def _enrich_with_openalex(
        self, normalized_semantic_results: List[NormalizedPaperResult]
    ) -> List[NormalizedPaperResult]:
        """
        Enrich Semantic Scholar results with OpenAlex metadata via DOI/OpenAlex ID lookup.

        Args:
            semantic_results: Normalized results from Semantic Scholar

        Returns:
            Enriched results with OpenAlex metadata merged
        """
        openalex_provider: OpenAlexProvider = self.providers[RetrievalServiceType.OPENALEX]  

        dois = []
        doi_to_semantic = {}

        for result in normalized_semantic_results:
            external_ids = result.external_ids or {}
            if not external_ids:
                continue
            elif "DOI" in external_ids:
                doi = external_ids["DOI"].strip().lower() # type: ignore
                dois.append(doi)
                doi_to_semantic[doi] = result
        openalex_data = {}
        if dois:
            openalex_id_results = []
            try:
                openalex_id_results = await openalex_provider.get_papers_by_dois(dois, limit=len(dois))
            except Exception as e:
                logger.error(f"Error fetching OpenAlex data for enrichment: {e}")
            for oa_result in openalex_id_results:
                doi = oa_result.get("doi")
                if isinstance(doi, str):
                    doi = doi.removeprefix("https://doi.org/").strip().lower()
                    openalex_data[doi] = oa_result

        logger.debug(
            f"Fetched {len(openalex_data)} OpenAlex records for enrichment"
        )

        enriched = []
        for doi, normalized_semantic_result in doi_to_semantic.items():
            openalex_result = openalex_data.get(doi)

            if openalex_result:
                normalized_openalex_result = openalex_provider.normalize_result(
                    openalex_result
                )
                merged = self._merge_semantic_and_openalex(
                    normalized_semantic_result, normalized_openalex_result
                )
                enriched.append(merged)
            else:
                enriched.append(normalized_semantic_result)

        return enriched

    def _merge_semantic_and_openalex(
        self,
        semantic_result: NormalizedPaperResult,
        openalex_result: NormalizedPaperResult,
    ) -> NormalizedPaperResult:
        """
        Merge Semantic Scholar and OpenAlex normalized results.
        Combines the best of both: author h-index from S2, institutions from OA authorships.

        Args:
            semantic_result: Normalized result from Semantic Scholar
            openalex_result: Normalized result from OpenAlex

        Returns:
            Merged NormalizedResult with unified author data
        """
        merged_model = semantic_result.model_copy()

        merged_model.fwci = openalex_result.fwci
        merged_model.is_retracted = openalex_result.is_retracted
        merged_model.citation_percentile = openalex_result.citation_percentile
        merged_model.language = openalex_result.language
        merged_model.topics = openalex_result.topics
        merged_model.keywords = openalex_result.keywords
        merged_model.concepts = openalex_result.concepts
        merged_model.mesh_terms = openalex_result.mesh_terms
        merged_model.has_content = openalex_result.has_content
        merged_model.biblio = openalex_result.biblio
        merged_model.primary_location = openalex_result.primary_location
        merged_model.locations = openalex_result.locations
        merged_model.best_oa_location = openalex_result.best_oa_location

        # Author collaboration metadata
        merged_model.corresponding_author_ids = openalex_result.corresponding_author_ids
        merged_model.institutions_distinct_count = openalex_result.institutions_distinct_count
        merged_model.countries_distinct_count = openalex_result.countries_distinct_count
        
        merged_authors = []
        semantic_authors = semantic_result.authors or []
        oa_authors = openalex_result.authors or []

        for i, s2_author in enumerate(semantic_authors):
            oa_author = self._find_matching_author(s2_author, oa_authors, i)
            merged_author = s2_author.model_copy()
            if oa_author:
                merged_author.institutions = oa_author.institutions
                merged_author.affiliations = oa_author.affiliations
                merged_author.openalex_id = oa_author.author_id
                merged_author.orcid = oa_author.orcid
            merged_authors.append(merged_author)

        merged_model.authors = merged_authors
        if openalex_result.venue:
            merged_model.venue = semantic_result.venue or openalex_result.venue

        if not merged_model.external_ids:
            merged_model.external_ids = {}

        merged_model.external_ids["OpenAlex"] = (
            openalex_result.paper_id.removeprefix("https://openalex.org/")
        )

        return merged_model

    async def resolve_paper_content(
        self, paper: PaperEnrichedDTO
    ) -> Optional[ResolvedPaperContent]:
        """
        Fetch PDF bytes for a given paper.

        Args:
            paper: Paper object
        Returns:
            PDF content as bytes, or None if not available
        """
        if paper.has_content.get("grobid_xml", True):
            tei_xml = await self.get_tei_xml(paper)
            if tei_xml:
                return ResolvedPaperContent(kind="tei_xml", content=tei_xml)

        pdf = await self.get_pdf_paper(paper)
        if pdf:
            return ResolvedPaperContent(kind="pdf_bytes", content=pdf)

        return None

    async def get_tei_xml(self, paper: PaperEnrichedDTO) -> Optional[str]:
        """
        Get TEI XML for a paper from OpenAlex.

        TEI XML provides structured full-text extraction from GROBID,
        superior to PDF parsing for academic papers.

        Args:
            paper: PaperEnrichedDTO object with openalex_id

        Returns:
            TEI XML string, or None if not available
        """
        try:
            # Extract OpenAlex ID from external_ids (case-insensitive)
            openalex_id = None
            if paper.external_ids:
                openalex_id = paper.external_ids.get(
                    "OpenAlex"
                ) or paper.external_ids.get("openalex")

            if not openalex_id:
                logger.info(
                    f"Paper {paper.paper_id} has no OpenAlex ID, skipping TEI XML retrieval"
                )
                return None

            logger.info(
                f"Attempting to download TEI XML for OpenAlex ID: {openalex_id}"
            )
            tei_xml = await self.paper_retriever.download_tei_from_openalex(openalex_id)

            if tei_xml:
                logger.info(f"Successfully retrieved TEI XML for {openalex_id}")
                return tei_xml
            else:
                logger.info(f"No TEI XML available for {openalex_id}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving TEI XML for {paper.paper_id}: {e}")
            return None

    async def get_pdf_paper(self, paper: PaperEnrichedDTO) -> Optional[bytes]:
        """
        Get PDF bytes for a paper.

        First tries the pdf_url from the paper metadata (already extracted from API).
        Falls back to access_info lookup if needed.

        Args:
            paper: PaperPreprocess object with metadata

        Returns:
            PDF content as bytes, or None if not available
        """
        try:
            # S2 API-provided PDF URL
            if paper.pdf_url:
                logger.info(
                    f"Attempting to download PDF from API-provided URL: {paper.pdf_url}"
                )
                pdfBytes = await self.paper_retriever.download_pdf(
                    paper.pdf_url, check_open_access=False
                )
                if pdfBytes:
                    return pdfBytes
                else:
                    logger.warning(f"Failed to download from API URL: {paper.pdf_url}")
            else:
                logger.info("Paper has no API-provided PDF URL, checking OpenAlex...")
                openalex_id = None
                if paper.external_ids:
                    openalex_id = paper.external_ids.get(
                        "OpenAlex"
                    ) or paper.external_ids.get("openalex")

                if not openalex_id:
                    logger.info(
                        f"Paper {paper.paper_id} has no OpenAlex ID, skipping TEI XML retrieval"
                    )
                    return None

                pdfBytes = await self.paper_retriever.download_pdf_from_openalex(
                    openalex_id=openalex_id
                )
                if pdfBytes:
                    return pdfBytes

            logger.info(
                "Paper is not open-access and has no PDF URL, cannot retrieve full-text"
            )
            return None
        except Exception as e:
            logger.error(f"Error retrieving PDF for paper {paper.paper_id}: {e}")
            return None

    async def get_relevant_chunks(
        self, query: str, paper_ids: Optional[List[str]] = None, limit: int = 40
    ) -> List[ChunkRetrieved]:
        """
        Get relevant chunks for a query with relevance scores attached

        Args:
            query: Query text
            paper_ids: Optional list of paper IDs to restrict search
            limit: Number of chunks to return

        Returns:
            List of relevant chunks with relevance_score attribute set
        """
        query_embedding = await self.embedding_service.create_embedding(query, task="search_query")

        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []

        chunks_with_scores = await self.chunk_service.search_similar_chunks(
            query_embedding, limit=limit, paper_ids=paper_ids
        )

        return chunks_with_scores

    T = TypeVar("T", bound=RetrievalProvider)

    def get_provider_as(
        self, service_type: RetrievalServiceType, provider_class: Type[T]
    ) -> T:
        provider = self.providers.get(service_type)
        if not isinstance(provider, provider_class):
            raise TypeError(
                f"Provider {service_type} is not of type {provider_class.__name__}"
            )
        return provider

    def _find_matching_author(self, s2_author, oa_authors, index):
        s2_last = surname(s2_author.name)
        
        if index < len(oa_authors):
            oa_author = oa_authors[index]
            if surname(oa_author.name) == s2_last:
                return oa_author

        # Try nearby positions (handles missing authors)
        for shift in (-1, 1):
            new_i = index + shift
            if 0 <= new_i < len(oa_authors):
                oa_author = oa_authors[new_i]
                if surname(oa_author.name) == s2_last:
                    return oa_author

        return None
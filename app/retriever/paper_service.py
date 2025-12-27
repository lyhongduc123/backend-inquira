from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.papers import DBPaper, DBPaperChunk
from app.extensions.logger import create_logger
from app.core.config import settings
from app.retriever.provider import SemanticScholarProvider, RetrievalConfig, RetrievalMode
from app.processor.services.embeddings import EmbeddingService
from .provider.openalex_provider import OpenAlexProvider
from .provider.scholar_provider import GoogleScholarProvider
from .paper_retriever import PaperRetriever
from .paper_repository import PaperRepository
from .paper_schemas import Paper, PaperChunk, Author
from .provider.arxiv_provider import ArxivProvider
from .provider.base import BaseRetrievalProvider
from .provider.base_schemas import NormalizedResult
from .utils import batch_normalized_to_papers
from .result_logger import save_retrieval_results, save_paper_analysis
import httpx
from collections import defaultdict

logger = create_logger(__name__)


class RetrievalServiceType(str, Enum):
    SEMANTIC = 'semantic'
    ARXIV = 'arxiv'
    SCHOLAR = 'scholar'
    OPENALEX = 'openalex'


class PaperRetrievalService:
    """
    Unified paper retrieval service.
    
    Handles:
    - Multiple providers (Semantic Scholar, arXiv, OpenAlex, Google Scholar)
    - Full-text retrieval
    """
    
    def __init__(self, db: AsyncSession):
        self.paper_retriever = PaperRetriever()
        self.embedding_service = EmbeddingService()
        
        config = RetrievalConfig(
            mode=RetrievalMode.ENHANCED,
            enable_caching=True,
            enable_full_text=True,
            max_results=100
        )
        self.providers: Dict[RetrievalServiceType, BaseRetrievalProvider] = {
            RetrievalServiceType.SEMANTIC: SemanticScholarProvider(
                api_url=settings.SEMANTIC_API_URL,
                config=config,
                db_session=db
            ),
            RetrievalServiceType.ARXIV: ArxivProvider(
                api_url=settings.ARXIV_API_URL,
                config=config
            ),
            RetrievalServiceType.SCHOLAR: GoogleScholarProvider(
                api_url=settings.SCHOLAR_URL,
                config=config
            ),
            RetrievalServiceType.OPENALEX: OpenAlexProvider(
                api_url=settings.OPENALEX_URL,
                config=config
            )
        }
        self.embedding_service = EmbeddingService()
        self.repository = PaperRepository(db)
        
    async def search(
        self, 
        query: str, 
        limit: int, 
        services: List[RetrievalServiceType],
        save_results: bool = False
    ) -> List[Paper]:
        """
        Search for papers across specified services
        
        Args:
            query: Search query
            limit: Number of papers to retrieve per service
            services: List of retrieval services to use
            save_results: If True, save raw retrieval results to JSON for debugging
            
        Returns:
            List of Paper objects
        """
        results: List[NormalizedResult] = []
        for service_type in services:
            provider = self.providers.get(service_type)
            if not provider:
                logger.warning(f"Provider for service {service_type} not found")
                continue
            
            try:
                service_papers = await provider.search_and_normalize(query, limit)
                results.extend(service_papers)
                logger.info(f"Retrieved {len(service_papers)} papers from {service_type}")
            except Exception as e:
                logger.error(f"Error retrieving papers from {service_type}: {e}")
        
        # Optionally save raw results for analysis
        if save_results and results:
            try:
                save_retrieval_results(results, query=query, provider=str(services))
                save_paper_analysis(results)
            except Exception as e:
                logger.warning(f"Failed to save retrieval results: {e}")
        
        papers = batch_normalized_to_papers(results)
        return papers
    
    async def hybrid_search(
        self,
        query: str,
        semantic_limit: int = 100,
        openalex_limit: int = 50,
        final_limit: int = 25,
        filters: Optional[Dict[str, Any]] = None,
        enable_enrichment: bool = True
    ) -> Tuple[List[Paper], Dict[str, Any]]:
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
            final_limit: Max final results to return (default 25)
            filters: Optional filters (year_min, year_max, fields)
            enable_enrichment: Whether to enrich with OpenAlex metadata
            
        Returns:
            Tuple of (papers, metadata)
            - papers: List of Paper objects with enriched metadata
            - metadata: Search metadata (counts, sources, etc.)
        """
        logger.info(f"[HybridSearch] Starting hybrid search for: {query[:50]}...")
        
        # Step 1: Semantic Scholar semantic search
        semantic_provider = self.providers.get(RetrievalServiceType.SEMANTIC)
        semantic_results = []
        
        if semantic_provider:
            try:
                logger.info(f"[HybridSearch] Fetching {semantic_limit} papers from Semantic Scholar...")
                raw_semantic = await semantic_provider.search_papers(query, limit=semantic_limit)
                semantic_results = [semantic_provider.normalize_result(r) for r in raw_semantic]
                logger.info(f"[HybridSearch] Retrieved {len(semantic_results)} papers from Semantic Scholar")
            except Exception as e:
                logger.error(f"[HybridSearch] Semantic Scholar search error: {e}")
        
        # Step 2 & 3: Extract identifiers and enrich with OpenAlex
        enriched_papers = []
        if enable_enrichment and semantic_results:
            enriched_papers = await self._enrich_with_openalex(semantic_results)
            logger.info(f"[HybridSearch] Enriched {len(enriched_papers)} papers with OpenAlex metadata")
        else:
            enriched_papers = semantic_results
        
        # Step 4: Optional OpenAlex keyword search for additional results
        openalex_provider = self.providers.get(RetrievalServiceType.OPENALEX)
        openalex_results = []
        
        if openalex_provider and openalex_limit > 0:
            try:
                logger.info(f"[HybridSearch] Fetching {openalex_limit} additional papers from OpenAlex...")
                raw_openalex = await openalex_provider.search_papers(query, limit=openalex_limit)
                openalex_results = [openalex_provider.normalize_result(r) for r in raw_openalex]
                logger.info(f"[HybridSearch] Retrieved {len(openalex_results)} papers from OpenAlex keyword search")
            except Exception as e:
                logger.error(f"[HybridSearch] OpenAlex search error: {e}")
        
        # Step 5: Merge and deduplicate
        merged_results = self._merge_and_deduplicate(enriched_papers, openalex_results)
        logger.info(f"[HybridSearch] Merged to {len(merged_results)} unique papers")
        
        # Convert to Paper objects
        papers = batch_normalized_to_papers(merged_results[:final_limit])
        
        # Prepare metadata
        metadata = {
            'total_before_dedup': len(enriched_papers) + len(openalex_results),
            'total_after_dedup': len(merged_results),
            'semantic_scholar_count': len(semantic_results),
            'openalex_enriched_count': len(enriched_papers),
            'openalex_keyword_count': len(openalex_results),
            'duplicates_removed': (len(enriched_papers) + len(openalex_results)) - len(merged_results),
            'final_returned': len(papers),
            'sources': {
                'semantic_scholar': len(semantic_results) > 0,
                'openalex': len(openalex_results) > 0,
                'enrichment_enabled': enable_enrichment
            }
        }
        
        logger.info(f"[HybridSearch] Completed. Returning {len(papers)} papers")
        return papers, metadata
    
    async def _enrich_with_openalex(
        self,
        semantic_results: List[NormalizedResult]
    ) -> List[NormalizedResult]:
        """
        Enrich Semantic Scholar results with OpenAlex metadata via DOI/OpenAlex ID lookup.
        
        Args:
            semantic_results: Normalized results from Semantic Scholar
            
        Returns:
            Enriched results with OpenAlex metadata merged
        """
        openalex_provider = self.providers.get(RetrievalServiceType.OPENALEX)
        if not openalex_provider:
            logger.warning("[HybridSearch] OpenAlex provider not available for enrichment")
            return semantic_results
        
        # Extract DOIs and OpenAlex IDs
        dois = []
        openalex_ids = []
        id_to_semantic = {}
        
        for result in semantic_results:
            external_ids = result.get('external_ids', {})
            if not external_ids:
                continue
            
            # Prioritize OpenAlex ID
            if 'OpenAlex' in external_ids:
                oa_id = external_ids['OpenAlex']
                openalex_ids.append(oa_id)
                id_to_semantic[oa_id] = result
            # Fallback to DOI
            elif 'DOI' in external_ids:
                doi = external_ids['DOI']
                dois.append(doi)
                id_to_semantic[f"doi:{doi}"] = result
        
        logger.info(f"[HybridSearch] Enrichment: {len(openalex_ids)} OpenAlex IDs, {len(dois)} DOIs")
        
        # Batch fetch from OpenAlex
        openalex_data_map = {}
        
        # Fetch by OpenAlex IDs
        if openalex_ids:
            oa_results = await self._batch_fetch_openalex_by_ids(openalex_ids)
            for oa_data in oa_results:
                oa_id = oa_data.get('id', '')
                openalex_data_map[oa_id] = oa_data
        
        # Fetch by DOIs
        if dois:
            doi_results = await self._batch_fetch_openalex_by_dois(dois)
            for oa_data in doi_results:
                doi = oa_data.get('doi', '').replace('https://doi.org/', '')
                if doi:
                    openalex_data_map[f"doi:{doi}"] = oa_data
        
        logger.info(f"[HybridSearch] Fetched {len(openalex_data_map)} OpenAlex records for enrichment")
        
        # Merge data
        enriched = []
        for identifier, semantic_result in id_to_semantic.items():
            openalex_data = openalex_data_map.get(identifier)
            
            if openalex_data:
                # Normalize OpenAlex data
                normalized_oa = openalex_provider.normalize_result(openalex_data)
                # Merge with semantic data
                merged = self._merge_semantic_and_openalex(semantic_result, normalized_oa)
                enriched.append(merged)
            else:
                # No OpenAlex data found, keep original
                enriched.append(semantic_result)
        
        return enriched
    
    async def _batch_fetch_openalex_by_ids(
        self,
        openalex_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Batch fetch OpenAlex works by OpenAlex IDs.
        
        Args:
            openalex_ids: List of OpenAlex IDs
            
        Returns:
            List of raw OpenAlex work dictionaries
        """
        results = []
        chunk_size = 50  # OpenAlex allows up to 50 per request
        
        try:
            for i in range(0, len(openalex_ids), chunk_size):
                chunk = openalex_ids[i:i + chunk_size]
                
                # Build filter with pipe-separated IDs
                filter_query = '|'.join(chunk)
                params = {
                    'filter': f'openalex_id:{filter_query}',
                    'per-page': chunk_size
                }
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(
                        f"{settings.OPENALEX_URL}/works",
                        params=params
                    )
                    response.raise_for_status()
                    data = response.json()
                    works = data.get('results', [])
                    results.extend(works)
                    
                    logger.debug(f"[HybridSearch] Fetched {len(works)} works by OpenAlex IDs")
                
        except Exception as e:
            logger.error(f"[HybridSearch] Error batch fetching by OpenAlex IDs: {e}")
        
        return results
    
    async def _batch_fetch_openalex_by_dois(
        self,
        dois: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Batch fetch OpenAlex works by DOIs.
        
        Args:
            dois: List of DOIs
            
        Returns:
            List of raw OpenAlex work dictionaries
        """
        results = []
        chunk_size = 50
        
        try:
            for i in range(0, len(dois), chunk_size):
                chunk = dois[i:i + chunk_size]
                
                # Build filter with pipe-separated DOIs
                # OpenAlex expects full DOI URLs
                doi_urls = [f'https://doi.org/{doi}' for doi in chunk]
                filter_query = '|'.join(doi_urls)
                params = {
                    'filter': f'doi:{filter_query}',
                    'per-page': chunk_size
                }
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(
                        f"{settings.OPENALEX_URL}/works",
                        params=params
                    )
                    response.raise_for_status()
                    data = response.json()
                    works = data.get('results', [])
                    results.extend(works)
                    
                    logger.debug(f"[HybridSearch] Fetched {len(works)} works by DOIs")
                
        except Exception as e:
            logger.error(f"[HybridSearch] Error batch fetching by DOIs: {e}")
        
        return results
    
    def _merge_semantic_and_openalex(
        self,
        semantic_result: NormalizedResult,
        openalex_result: NormalizedResult
    ) -> NormalizedResult:
        """
        Merge Semantic Scholar and OpenAlex normalized results.
        Combines the best of both: author h-index from S2, FWCI/institutions from OA.
        
        Args:
            semantic_result: Normalized result from Semantic Scholar
            openalex_result: Normalized result from OpenAlex
            
        Returns:
            Merged NormalizedResult
        """
        # Start with semantic result as base (has better author metrics)
        merged: Dict[str, Any] = dict(semantic_result)
        
        # Add/update OpenAlex-specific fields
        if openalex_result:
            # Citation count - use max
            merged['citation_count'] = max(
                semantic_result.get('citation_count') or 0,
                openalex_result.get('citation_count') or 0
            )
            
            # OpenAlex metadata
            merged['fwci'] = openalex_result.get('fwci')
            merged['cited_by_percentile_year'] = openalex_result.get('cited_by_percentile_year')
            merged['citation_normalized_percentile'] = openalex_result.get('citation_normalized_percentile')
            merged['is_retracted'] = openalex_result.get('is_retracted', False)
            merged['is_paratext'] = openalex_result.get('is_paratext', False)
            merged['indexed_in'] = openalex_result.get('indexed_in', [])
            
            # Topics and concepts (OpenAlex has better taxonomy)
            merged['topics'] = openalex_result.get('topics', [])
            merged['concepts'] = openalex_result.get('concepts', [])
            merged['primary_topic'] = openalex_result.get('primary_topic')
            
            # Authorships with institutions (from OpenAlex)
            merged['authorships'] = openalex_result.get('authorships', [])
            merged['institutions_distinct_count'] = openalex_result.get('institutions_distinct_count', 0)
            merged['countries_distinct_count'] = openalex_result.get('countries_distinct_count', 0)
            
            # Venue info (OpenAlex has more details)
            if openalex_result.get('venue'):
                merged['venue'] = openalex_result.get('venue')
            
            # Update external IDs
            if 'external_ids' not in merged:
                merged['external_ids'] = {}
            oa_external_ids = openalex_result.get('external_ids')
            if oa_external_ids and isinstance(oa_external_ids, dict):
                merged_ext_ids = merged.get('external_ids')
                if isinstance(merged_ext_ids, dict):
                    merged_ext_ids.update(oa_external_ids)
            
            # Store raw data for comprehensive scoring
            merged['openalex_data'] = openalex_result.get('raw_data')
        
        # Keep semantic data
        merged['semantic_data'] = semantic_result.get('raw_data')
        
        # Cast back to NormalizedResult (type checker will accept this)
        from typing import cast
        return cast(NormalizedResult, merged)
    
    def _merge_and_deduplicate(
        self,
        semantic_results: List[NormalizedResult],
        openalex_results: List[NormalizedResult]
    ) -> List[NormalizedResult]:
        """
        Merge results from both sources and remove duplicates by DOI.
        Semantic Scholar results take priority.
        
        Args:
            semantic_results: Results from Semantic Scholar (already enriched)
            openalex_results: Results from OpenAlex keyword search
            
        Returns:
            Deduplicated list of results
        """
        papers_by_doi = {}
        papers_by_title = {}
        
        # Add semantic results first (priority)
        for result in semantic_results:
            external_ids = result.get('external_ids') or {}
            doi = external_ids.get('DOI', '').lower() if isinstance(external_ids, dict) else ''
            title = self._normalize_title(result.get('title', ''))
            
            if doi:
                papers_by_doi[doi] = result
            elif title:
                papers_by_title[title] = result
        
        # Add OpenAlex results (skip if duplicate)
        for result in openalex_results:
            external_ids = result.get('external_ids') or {}
            doi = external_ids.get('DOI', '').lower() if isinstance(external_ids, dict) else ''
            title = self._normalize_title(result.get('title', ''))
            
            # Check if already exists
            if doi and doi in papers_by_doi:
                continue
            if title and title in papers_by_title:
                continue
            
            # New paper, add it
            if doi:
                papers_by_doi[doi] = result
            elif title:
                papers_by_title[title] = result
        
        # Combine all unique papers
        all_papers = list(papers_by_doi.values()) + list(papers_by_title.values())
        
        # Remove any remaining duplicates by title
        seen_titles = set()
        unique_papers = []
        for paper in all_papers:
            title = self._normalize_title(paper.get('title', ''))
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for deduplication."""
        if not title:
            return ''
        # Remove punctuation and extra spaces
        normalized = title.lower().strip()
        normalized = normalized.replace('.', '').replace(',', '').replace(':', '').replace(';', '')
        normalized = ' '.join(normalized.split())
        return normalized
    
    # async def search_and_retrieve_papers(
    #     self,
    #     query: str,
    #     limit: int = 20,
    #     auto_process: bool = True
    # ) -> List[DBPaper]:
        """
        Search for papers and optionally auto-process them
        
        Args:
            query: Search query
            limit: Number of papers to retrieve
            auto_process: Whether to automatically process papers (fetch full-text, chunk, embed)
            
        Returns:
            List of DBPaper objects
        """
        # logger.info(f"Searching papers for query: {query[:100]}...")
        
        # raw_papers = await self.semantic_provider.search_papers(query, limit=limit)
        
        # if not raw_papers:
        #     logger.warning(f"No papers found for query: {query}")
        #     return []
        
        # papers = []
        # for raw_paper in raw_papers:
        #     try:
        #         normalized = self.semantic_provider.normalize_result(raw_paper)
        #         paper = self._convert_normalized_to_paper(normalized)
        #         papers.append(paper)
        #     except Exception as e:
        #         logger.error(f"Error converting paper: {e}")
        #         continue
        
        # # Check which papers already exist in database
        # db_papers = []
        # for paper in papers:
        #     if not paper.external_id:
        #         continue
                
        #     print(f"Processing paper: {paper.title} ({paper.external_id}) source: {paper.source}")
        #     existing = await self.repository.get_paper_by_external_id(
        #         paper.external_id, paper.source
        #     )
            
        #     if existing:
        #         logger.info(f"Paper {paper.paper_id} already exists in database")
         
        #         await self.repository.update_last_accessed(str(existing.paper_id))
        #         db_papers.append(existing)
                
        #         # Process if not yet processed and auto_process is enabled
        #         # Access actual values from SQLAlchemy columns
        #         is_processed = bool(existing.is_processed)
        #         pdf_url = getattr(existing, 'pdf_url', None)
        #         pdf_url_str = str(pdf_url) if pdf_url is not None else None
        #         if auto_process and not is_processed and pdf_url_str:
        #             logger.info(f"Auto-processing existing paper {str(existing.paper_id)}")
        #             await self._process_paper(existing)
        #     else:
        #         db_paper = await self.repository.create_paper(paper)
        #         db_papers.append(db_paper)
                
        #         # Process if auto_process is enabled
        #         if auto_process:
        #             logger.info(f"Auto-processing new paper {str(db_paper.paper_id)}")
        #             await self._process_paper(db_paper)
        
        # return db_papers
        
    async def get_paper_if_exists(self, external_ids: Dict[str, str], source: str) -> Optional[DBPaper]:
        """
        Check if a paper exists in the database by external IDs and source
        
        Args:
            external_ids: External paper IDs dict (e.g., {"DOI": "...", "ArXiv": "..."})
            source: Source/provider name
        Returns:
            DBPaper object if exists, else None
        """
        paper = await self.repository.get_paper_by_external_ids(external_ids, source)
        return paper
    
        
    async def get_pdf_paper(
        self,
        paper: Paper
    ) -> Optional[bytes]:
        """
        Get PDF bytes for a paper.
        
        First tries the pdf_url from the paper metadata (already extracted from API).
        Falls back to access_info lookup if needed.
        
        Args:
            paper: Paper object with metadata
            
        Returns:
            PDF content as bytes, or None if not available
        """
        try:
            # First, try using the pdf_url already extracted from the API
            if paper.pdf_url:
                logger.info(f"Attempting to download PDF from API-provided URL: {paper.pdf_url}")
                pdfBytes = await self.paper_retriever.download_pdf(paper.pdf_url, check_open_access=False)
                if pdfBytes:
                    return pdfBytes
                else:
                    logger.warning(f"Failed to download from API URL: {paper.pdf_url}")
            
            # Fallback: check access info for alternative PDF URLs
            result = self.paper_retriever.get_access_info(paper.dict())
            if result.get("is_open_access"):
                fallback_pdf_url = result.get("pdf_url")
                if fallback_pdf_url and fallback_pdf_url != paper.pdf_url:
                    logger.info(f"Trying fallback PDF URL from access_info: {fallback_pdf_url}")
                    pdfBytes = await self.paper_retriever.download_pdf(str(fallback_pdf_url))
                    if pdfBytes:
                        return pdfBytes
                    else:
                        logger.warning(f"Failed to download from fallback URL: {fallback_pdf_url}")
                elif not fallback_pdf_url:
                    logger.warning("Open-access paper has no pdf_url in access info")
            else:
                if not paper.pdf_url:
                    logger.info("Paper is not open-access and has no PDF URL, cannot retrieve full-text")
            
            return None
        except Exception as e:
            logger.error(f"Error retrieving PDF for paper {paper.paper_id}: {e}")
            return None
            
    async def get_relevant_chunks(
        self,
        query: str,
        paper_ids: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[DBPaperChunk]:
        """
        Get relevant chunks for a query
        
        Args:
            query: Query text
            paper_ids: Optional list of paper IDs to restrict search
            limit: Number of chunks to return
            
        Returns:
            List of relevant chunks
        """
        # Generate query embedding
        query_embedding = await self.embedding_service.create_embedding(query)
        
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []
        
        # Search for similar chunks
        chunks = await self.repository.search_similar_chunks(
            query_embedding, limit=limit, paper_ids=paper_ids
        )
        
        return chunks
    
    async def get_paper_summaries(
        self,
        query: str,
        limit: int = 10
    ) -> List[DBPaper]:
        """
        Get relevant paper summaries for a query
        
        Args:
            query: Query text
            limit: Number of papers to return
            
        Returns:
            List of relevant papers with summaries
        """
        # Generate query embedding
        query_embedding = await self.embedding_service.create_embedding(query)
        
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []
        
        # Search for similar papers
        papers = await self.repository.search_similar_papers(
            query_embedding, limit=limit
        )
        
        return papers
    
    def get_provider(self, service_type: RetrievalServiceType) -> Optional[BaseRetrievalProvider]:
        """
        Get the provider instance for a given service type
        
        Args:
            service_type: RetrievalServiceType enum value
        Returns:
            BaseRetrievalProvider instance or None
        """
        return self.providers.get(service_type)
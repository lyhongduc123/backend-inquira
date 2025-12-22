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
"""
Bulk preprocessing service using Semantic Scholar bulk search API.

Workflow:
1. Use POST /paper/search/bulk to get papers matching criteria
2. Enrich Semantic Scholar results with OpenAlex metadata
3. Check if paper exists in database (skip if yes)
4. Create papers and process through RAG pipeline (extract, chunk, embed)
5. Handle pagination with continuation tokens
6. Track progress with state management
"""
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.extensions.logger import create_logger
from app.papers.repository import PaperRepository
from app.papers.service import PaperService
from app.retriever.service import RetrievalService, RetrievalServiceType
from app.processor.services import transformer
from app.processor.paper_processor import PaperProcessor
from app.core.config import settings
from app.models.preprocessing_state import DBPreprocessingState
from app.chunks.repository import ChunkRepository

logger = create_logger(__name__)


class PreprocessingService:
    """
    Service for bulk preprocessing using Semantic Scholar bulk search.
    
    Uses the bulk search API to retrieve papers and process them through
    the RAG pipeline (PDF download, chunking, embedding).
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.repository = PaperRepository(db_session)
        self.paper_service = PaperService(self.repository)
        
        # Use RetrievalService for unified retrieval and enrichment
        self.retriever = RetrievalService(db=db_session)
        
        # Use PaperProcessor for RAG pipeline (extraction, chunking, embedding)
        chunk_repository = ChunkRepository(db_session)
        self.processor = PaperProcessor(
            repository=self.repository,
            chunk_repository=chunk_repository,
            retrieval_service=self.retriever
        )
        
        # HTTP client for bulk search API
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=30.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        
        self.semantic_base_url = settings.SEMANTIC_API_URL
        self.semantic_api_key = settings.SEMANTIC_API_KEY
    
    async def close(self):
        """Close HTTP client connections."""
        await self.http_client.aclose()
    
    async def process_bulk_search(
        self,
        job_id: str,
        search_query: str,
        target_count: int,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        fields_of_study: Optional[List[str]] = None,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Process papers from Semantic Scholar bulk search API.
        
        Workflow:
        1. Fetch papers from bulk search API with pagination
        2. Enrich with OpenAlex metadata via RetrievalService
        3. Check if papers exist in database (skip if yes)
        4. Create papers and process through RAG pipeline
        5. Track progress with state management
        
        Args:
            job_id: Unique job identifier for tracking
            search_query: Search query string
            target_count: Number of papers to process
            year_min: Minimum publication year
            year_max: Maximum publication year
            fields_of_study: List of fields to filter by
            resume: Whether to resume from previous state
            
        Returns:
            Statistics about the preprocessing job
        """
        logger.info(f"Starting bulk search preprocessing job {job_id}")
        logger.info(f"Query: {search_query}, Target: {target_count}")
        
        # Get or create job state
        state = await self._get_or_create_state(job_id, target_count, resume)
        
        if state.is_completed:
            logger.info(f"Job {job_id} already completed")
            return self._state_to_stats(state)
        
        if state.is_running:
            logger.warning(f"Job {job_id} is already running")
            return self._state_to_stats(state)
        
        # Mark as running
        state.is_running = True
        state.status_message = "Starting bulk search..."  # type: ignore
        await self.db_session.commit()
        
        try:
            continuation_token = state.continuation_token if resume else None
            
            while state.processed_count < target_count:
                # Re-fetch state to check for pause
                stmt = select(DBPreprocessingState).where(DBPreprocessingState.job_id == job_id)
                result = await self.db_session.execute(stmt)
                state = result.scalar_one()
                
                if state.is_paused:
                    logger.info(f"Job {job_id} paused by user request")
                    state.is_running = False
                    state.status_message = "Paused by user"  # type: ignore
                    await self.db_session.commit()
                    return self._state_to_stats(state)
                
                # Fetch batch from bulk search
                batch_size = min(100, target_count - state.processed_count)
                result = await self._fetch_bulk_search(
                    query=search_query,
                    limit=batch_size,
                    token=continuation_token,
                    year_min=year_min,
                    year_max=year_max,
                    fields_of_study=fields_of_study
                )
                
                if not result or not result.get('data'):
                    logger.info("No more results from bulk search")
                    break
                
                papers_data = result['data']
                continuation_token = result.get('token')
                
                # Update state with token
                state.continuation_token = continuation_token  # type: ignore
                await self.db_session.commit()
                
                # Process batch (normalize, enrich, create, process through RAG)
                await self._process_search_batch(papers_data, state, target_count)
                
                # Re-fetch state after processing
                stmt = select(DBPreprocessingState).where(DBPreprocessingState.job_id == job_id)
                result = await self.db_session.execute(stmt)
                state = result.scalar_one()
                
                # Check if we should stop
                if not continuation_token or state.processed_count >= target_count:
                    break
                
                # Update status
                state.status_message = f"Processed: {state.processed_count}/{target_count} | Skipped: {state.skipped_count} | Errors: {state.error_count}"  # type: ignore
                await self.db_session.commit()
            
            # Re-fetch final state
            stmt = select(DBPreprocessingState).where(DBPreprocessingState.job_id == job_id)
            result = await self.db_session.execute(stmt)
            state = result.scalar_one()
            
            # Mark as completed
            state.is_completed = True
            state.is_running = False
            state.completed_at = datetime.now()  # type: ignore
            state.status_message = f"Completed: {state.processed_count} papers processed"  # type: ignore
            await self.db_session.commit()
            
            logger.info(f"Job {job_id} finished: {self._state_to_stats(state)}")
            return self._state_to_stats(state)
            
        except Exception as e:
            logger.error(f"Fatal error in preprocessing job {job_id}: {e}", exc_info=True)
            # Re-fetch state to update error status
            stmt = select(DBPreprocessingState).where(DBPreprocessingState.job_id == job_id)
            result = await self.db_session.execute(stmt)
            state = result.scalar_one_or_none()
            if state:
                state.is_running = False
                state.status_message = f"Error: {str(e)}"  # type: ignore
                await self.db_session.commit()
            raise
    
    async def _process_search_batch(
        self,
        papers_data: List[Dict[str, Any]],
        state: DBPreprocessingState,
        target_count: int
    ) -> None:
        """
        Process a batch of papers from bulk search results.
        
        Workflow:
        1. Filter for papers with openAccessPdf only
        2. Batch fetch full details from S2 (including author h-index, citation counts)
        3. Normalize enriched S2 results
        4. Enrich with OpenAlex metadata (using RetrievalService)
        5. Transform to Paper schema
        6. Check if already exists in database
        7. Create paper and process through RAG pipeline
        
        Args:
            papers_data: List of paper data from bulk search
            state: Job state to update
            target_count: Target paper count
        """
        # Filter for papers with open access PDF only
        oa_papers = [p for p in papers_data if p.get('isOpenAccess') and p.get('openAccessPdf')]
        
        if not oa_papers:
            logger.info("No open access papers in this batch")
            return
        
        # Extract paper IDs for batch fetch
        paper_ids = [p.get('paperId') for p in oa_papers if p.get('paperId')]
        
        if not paper_ids:
            logger.warning("No valid paper IDs in batch")
            return
        
        logger.info(f"Fetching full details for {len(paper_ids)} open access papers from S2")
        
        # Step 1: Batch fetch full paper details from S2 (with author metrics)
        detailed_papers = await self._fetch_paper_details_batch(paper_ids)
        
        if not detailed_papers:
            logger.warning("No detailed papers returned from S2 batch fetch")
            return
        
        logger.info(f"Retrieved {len(detailed_papers)} detailed papers from S2")
        
        # Step 2: Normalize Semantic Scholar results (now with full author data)
        semantic_provider = self.retriever.providers[RetrievalServiceType.SEMANTIC]
        
        normalized_papers = []
        for paper_data in detailed_papers:
            try:
                normalized = semantic_provider.normalize_result(paper_data)
                normalized_papers.append(normalized)
            except Exception as e:
                logger.error(f"Error normalizing paper: {e}")
                state.error_count += 1
                continue
        
        if not normalized_papers:
            logger.warning("No papers to process from this batch")
            return
        
        enriched_papers = await self.retriever._enrich_with_openalex(normalized_papers)
        logger.info(f"Enriched {len(enriched_papers)} papers with OpenAlex metadata")
        papers = transformer.batch_normalized_to_papers(enriched_papers)
        for paper in papers:
            if state.processed_count >= target_count:
                break
            
            # Re-fetch state to check for pause
            stmt = select(DBPreprocessingState).where(DBPreprocessingState.job_id == state.job_id)
            result = await self.db_session.execute(stmt)
            state = result.scalar_one()
            
            if state.is_paused:
                logger.info(f"Job {state.job_id} paused during batch processing")
                return
            
            try:
                paper_id = str(paper.paper_id)
                existing = await self.repository.get_paper_by_id(paper_id)
                if existing:
                    state.skipped_count += 1
                    state.current_index += 1
                    continue
                
                db_paper = await self.paper_service.create_paper_from_schema(paper)
                logger.info(f"Created paper {paper_id}")
                
                try:
                    success = await self.processor.process_single_paper(paper)
                    if success:
                        state.processed_count += 1
                        logger.info(f"Successfully processed {paper_id}")
                    else:
                        state.error_count += 1
                        logger.warning(f"Failed to process {paper_id}")
                    
                    state.current_index += 1
                    
                    # Commit after EVERY paper to free memory immediately
                    await self.db_session.commit()
                    
                    # Expunge all objects from session to free memory
                    self.db_session.expunge_all()
                    
                    # Re-fetch state
                    stmt_refresh = select(DBPreprocessingState).where(DBPreprocessingState.job_id == state.job_id)
                    result_refresh = await self.db_session.execute(stmt_refresh)
                    state = result_refresh.scalar_one()
                    
                    if state.processed_count % 5 == 0:
                        logger.info(f"Progress: {state.processed_count}/{target_count}")
                        
                except Exception as e:
                    logger.error(f"Error processing paper {paper_id} through RAG pipeline: {e}")
                    state.error_count += 1
                    state.current_index += 1
                    # Commit error state and clear session
                    await self.db_session.commit()
                    self.db_session.expunge_all()
                    # Re-fetch state
                    stmt_refresh = select(DBPreprocessingState).where(DBPreprocessingState.job_id == state.job_id)
                    result_refresh = await self.db_session.execute(stmt_refresh)
                    state = result_refresh.scalar_one()
                    
            except Exception as e:
                logger.error(f"Error processing paper from search: {e}")
                state.error_count += 1
                state.current_index += 1
    
    async def _fetch_bulk_search(
        self,
        query: str,
        limit: int = 100,
        token: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        fields_of_study: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch papers using Semantic Scholar bulk search API.
        
        Response format:
        {
            "total": 15117,
            "token": "continuation_token",
            "data": [{paper}, ...]
        }
        
        Args:
            query: Search query
            limit: Max results per request (max 1000)
            token: Continuation token for pagination
            year_min: Minimum publication year
            year_max: Maximum publication year
            fields_of_study: List of fields to filter
            
        Returns:
            Search result dict with total, token, and data
        """
        url = f"{self.semantic_base_url}/paper/search/bulk"
        
        params = {
            "query": query,
            "limit": min(limit, 1000),
            "fields": "paperId,title,abstract,year,authors,venue,publicationDate,citationCount,influentialCitationCount,referenceCount,url,openAccessPdf,isOpenAccess,externalIds",
            "openAccessPdf": ""
        }
        
        if token:
            params["token"] = token
        if year_min:
            params["year"] = f"{year_min}-"
        if year_max:
            if year_min:
                params["year"] = f"{year_min}-{year_max}"
            else:
                params["year"] = f"-{year_max}"
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        
        headers = {}
        if self.semantic_api_key:
            headers["x-api-key"] = self.semantic_api_key
        
        try:
            logger.info(f"Fetching bulk search: query='{query}', limit={limit}")
            response = await self.http_client.get(url, params=params, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            total = result.get('total', 0)
            data_count = len(result.get('data', []))
            logger.info(f"Bulk search returned {data_count} papers (total matches: {total})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching bulk search: {e}")
            return None
    
    async def _fetch_paper_details_batch(self, paper_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch detailed paper information in batch from Semantic Scholar.
        
        This enriches the bulk search results with full author metrics:
        - Authors with h-index, citation counts, paper counts
        - Detailed citation metrics
        - Full external IDs
        
        Args:
            paper_ids: List of Semantic Scholar paper IDs
            
        Returns:
            List of detailed paper dictionaries
        """
        if not paper_ids:
            return []
        
        url = f"{self.semantic_base_url}/graph/v1/paper/batch"
        
        # Request comprehensive fields including author metrics
        fields = [
            "paperId", "title", "abstract", "year", "publicationDate",
            "authors", "authors.authorId", "authors.name", "authors.hIndex", 
            "authors.citationCount", "authors.paperCount", "authors.url",
            "venue", "citationCount", "influentialCitationCount", "referenceCount",
            "url", "openAccessPdf", "isOpenAccess", "externalIds",
            "fieldsOfStudy", "s2FieldsOfStudy", "publicationTypes"
        ]
        
        params = {"fields": ",".join(fields), "openAccessPdf": ""}
        
        headers = {"Content-Type": "application/json"}
        if self.semantic_api_key:
            headers["x-api-key"] = self.semantic_api_key
        
        payload = {"ids": paper_ids}
        
        try:
            response = await self.http_client.post(
                url,
                json=payload,
                params=params,
                headers=headers,
                timeout=60.0
            )
            response.raise_for_status()
            results = response.json()
            
            # Filter out None results (papers not found)
            valid_results = [r for r in results if r is not None]
            logger.info(f"Fetched {len(valid_results)} paper details out of {len(paper_ids)}")
            
            return valid_results
            
        except Exception as e:
            logger.error(f"Error batch fetching paper details: {e}")
            return []
    
    async def _get_or_create_state(
        self,
        job_id: str,
        target_count: int,
        resume: bool
    ) -> DBPreprocessingState:
        """Get existing state or create new one."""
        stmt = select(DBPreprocessingState).where(DBPreprocessingState.job_id == job_id)
        result = await self.db_session.execute(stmt)
        state = result.scalar_one_or_none()
        
        if state and not resume:
            # Reset state for fresh start
            state.current_index = 0
            state.processed_count = 0
            state.skipped_count = 0
            state.error_count = 0
            state.target_count = target_count
            state.is_completed = False
            state.is_running = False
            state.is_paused = False
            state.completed_at = None  # type: ignore
            state.continuation_token = None  # type: ignore
        elif not state:
            # Create new state
            state = DBPreprocessingState(
                job_id=job_id,
                target_count=target_count,
                current_index=0,
                processed_count=0,
                skipped_count=0,
                error_count=0,
                is_completed=False,
                is_running=False,
                is_paused=False
            )
            self.db_session.add(state)
        else:
            # Resuming - clear pause flag if paused
            if state.is_paused:
                state.is_paused = False
                state.status_message = "Resuming from pause..."  # type: ignore
        
        await self.db_session.commit()
        await self.db_session.refresh(state)
        return state
    
    
    def _state_to_stats(self, state: DBPreprocessingState) -> Dict[str, Any]:
        """Convert state to statistics dict with computed metrics."""
        papers_per_second = 0.0
        eta_seconds = None
        
        if state.created_at and state.updated_at and state.is_running:
            elapsed = (state.updated_at - state.created_at).total_seconds()
            if elapsed > 0 and state.processed_count > 0:
                papers_per_second = state.processed_count / elapsed
                remaining = state.target_count - state.processed_count
                if papers_per_second > 0:
                    eta_seconds = int(remaining / papers_per_second)
        
        return {
            'job_id': state.job_id,
            'current_index': state.current_index,
            'processed_count': state.processed_count,
            'skipped_count': state.skipped_count,
            'error_count': state.error_count,
            'target_count': state.target_count,
            'is_completed': state.is_completed,
            'is_running': state.is_running,
            'is_paused': getattr(state, 'is_paused', False),
            'status_message': getattr(state, 'status_message', None),
            'current_file': getattr(state, 'current_file', None),
            'continuation_token': getattr(state, 'continuation_token', None),
            'papers_per_second': round(papers_per_second, 2),
            'eta_seconds': eta_seconds,
            'progress_percent': round((state.processed_count / state.target_count * 100), 1) if state.target_count > 0 else 0,
            'created_at': str(state.created_at) if state.created_at else None,
            'updated_at': str(state.updated_at) if state.updated_at else None,
            'completed_at': str(state.completed_at) if state.completed_at else None
        }

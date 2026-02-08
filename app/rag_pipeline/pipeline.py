# app/rag/pipeline.py

import asyncio
import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm import get_llm_service
from app.processor.paper_processor import PaperProcessor
from app.retriever.paper_service import PaperRetrievalService, RetrievalServiceType
from app.papers.repository import PaperRepository
from app.chunks.repository import ChunkRepository
from app.processor.services.ranking import RankingService

from app.rag_pipeline.schemas import RAGPipelineEvent, RAGResult
from app.extensions.logger import create_logger

from app.rag_pipeline.utils import deduplicate_papers
from app.processor.services import transformer

if TYPE_CHECKING:
    from app.processor.paper_processor import PaperProcessor


logger = create_logger(__name__)


class RAGPipeline:
    """
    Orchestrates the Retrieval-Augmented Generation pipeline.
    Responsibilities:
    - Break down user question
    - Retrieve & cache papers
    - Auto-process papers (PDF → chunk → embed)
    - Vector search to find relevant chunks
    - Return structured RAGResult(papers, chunks)

    Uses dependency injection for better testability and shared instances.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        repository: Optional[PaperRepository] = None,
        retriever: Optional[PaperRetrievalService] = None,
        processor: Optional[PaperProcessor] = None,
        llm_service=None,
        ranking_service: Optional[RankingService] = None,
    ):
        """Initialize RAGPipeline with dependency injection.

        Args:
            db_session: Required database session
            repository: Optional shared repository (created if not provided)
            retriever: Optional retrieval service (created if not provided)
            processor: Optional paper processor (created if not provided)
            llm_service: Optional LLM service (singleton if not provided)
            ranking_service: Optional ranking service (created if not provided)
        """
        self.db_session = db_session
        self.repository = repository or PaperRepository(db_session)
        self.chunk_repository = ChunkRepository(db_session)
        self.retriever = retriever or PaperRetrievalService(
            db=db_session
        )
        self.processor = processor or PaperProcessor(
            repository=self.repository,
            chunk_repository=self.chunk_repository,
            paper_service=self.retriever
        )
        self.llm = llm_service or get_llm_service()
        self.ranking_service = ranking_service or RankingService()

    async def run(
        self,
        query: str,
        max_subtopics: int = 3,
        per_subtopic_limit: int = 5,
        top_chunks: int = 40,
        filters: Optional[Dict[str, Any]] = None,
        enable_reranking: bool = True,
        enable_paper_ranking: bool = True,
        relevance_threshold: float = 0.3,
    ):
        """
        Complete pipeline:
        1. Break down question
        2. Retrieve papers w/ caching + auto-processing
        3. Deduplicate papers
        4. Vector search over chunks
        5. Rerank chunks with cross-encoder (optional)
        6. Rank papers with comprehensive scoring (optional)
        7. Return RAGResult

        Args:
            query: User question
            max_subtopics: Max subtopics to generate
            per_subtopic_limit: Max papers per subtopic
            top_chunks: Max chunks to return
            filters: Optional filters (yearRange, category, openAccessOnly, excludePreprints, topJournalsOnly)
            enable_reranking: Whether to rerank chunks with cross-encoder
            enable_paper_ranking: Whether to apply comprehensive paper ranking
            relevance_threshold: Minimum semantic relevance score for papers (0-1)
        """

        yield RAGPipelineEvent(
            type="step", data="Generating optimized search queries..."
        )
        
        breakdown = await self.llm.breakdown_user_question(user_question=query)
        search_queries = breakdown.search_queries[:max_subtopics]
        logger.info(f"Search queries: {search_queries}")
        yield RAGPipelineEvent(type="search_queries", data={"queries": search_queries, "original": query})

        yield RAGPipelineEvent(type="step", data=f"Searching academic databases...")
        all_papers = []

        for idx, search_query in enumerate(search_queries, 1):
            try:
                papers, metadata = await self.retriever.hybrid_search(
                    query=search_query,
                    semantic_limit=per_subtopic_limit,
                    final_limit=20,
                    filters=filters,
                )
                all_papers.extend(papers)
                logger.info(metadata)
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error retrieving papers for '{search_query}': {e}")

        papers = deduplicate_papers(all_papers)
        logger.info(f"Total unique papers: {len(papers)}")

        if not papers:
            yield RAGPipelineEvent(type="result", data=RAGResult(papers=[], chunks=[]))
            return

        # Process papers (PDF → chunk → embed) with progress streaming
        yield RAGPipelineEvent(type="step", data="Processing papers and extracting chunks...")
        
        processed_results = {}
        async for paper_id, success, current, total in self.processor.process_papers(papers, stream_progress=True): # type: ignore
            processed_results[paper_id] = success
            
            # Yield progress event to keep connection alive
            yield RAGPipelineEvent(
                type="processing", 
                data={
                    "paper_id": paper_id,
                    "success": success,
                    "progress": current,
                    "total": total,
                    "message": f"Processing paper {current}/{total}"
                }
            )
        
        paper_ids = [
            str(p.paper_id)
            for p in papers
            if processed_results.get(str(p.paper_id), False)
        ]

        output = {
            "original": all_papers,
            "deduplicated": papers,
            "processed": paper_ids,
        }
        self._write_log(output)

        # Retrieve chunks using multiple query variants for comprehensive coverage
        yield RAGPipelineEvent(type="step", data="Extracting relevant content from papers...")
        
        all_chunks = []
        queries_for_chunks = [query] + search_queries  # Original + generated queries
        
        for chunk_query in queries_for_chunks:
            query_chunks = await self.retriever.get_relevant_chunks(
                query=chunk_query, 
                paper_ids=paper_ids, 
                limit=top_chunks // len(queries_for_chunks)  # Distribute limit across queries
            )
            all_chunks.extend(query_chunks)
        
        # Deduplicate chunks by chunk_id
        seen_chunk_ids = set()
        chunks = []
        for chunk in all_chunks:
            if chunk.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk.chunk_id)
                chunks.append(chunk)

        if enable_reranking and chunks:
            chunks = self.ranking_service.rerank_chunks(query, chunks)
            logger.info(f"Reranked {len(chunks)} chunks")

        if enable_paper_ranking and papers:
            yield RAGPipelineEvent(type="step", data="Ranking papers by quality, authors, and relevance...")
            # ranked_papers = self.ranking_service.rank_papers(
            #     query=query,
            #     papers=papers,
            #     chunks=chunks,
            #     enable_diversity=True,
            #     relevance_threshold=relevance_threshold
            # )
            # logger.info(f"Ranked to {len(ranked_papers)} papers")
            # papers = ranked_papers

        yield RAGPipelineEvent(
            type="result", data=RAGResult(papers=papers, chunks=chunks)
        )

    def _write_log(self, output: dict):
        """Writing processed datas for debug

        Args:
            output (dict): A .json files contain the data
        """
        base_dir = Path(__file__).parent
        logs_dir = base_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        timestamp = datetime.datetime.now().isoformat().replace(":", "-")
        filename = logs_dir / f"processed_papers-{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4, default=str)

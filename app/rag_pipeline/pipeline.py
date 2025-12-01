# app/rag/pipeline.py

import asyncio
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm import llm_service
from app.processor.paper_processor import PaperProcessor
from app.retriever.paper_service import PaperRetrievalService, RetrievalServiceType
from app.retriever.paper_repository import PaperRepository

from app.rag_pipeline.schemas import RAGPipelineEvent, RAGResult
from app.extensions.logger import create_logger

from app.rag_pipeline.utils import deduplicate_papers
from app.retriever.utils import batch_dbpaper_to_papers

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
    """

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.repository = PaperRepository(db_session)
        self.llm = llm_service
        self.retriever = PaperRetrievalService(db_session)
        self.processor = PaperProcessor(self.repository)

    async def run(
        self,
        query: str,
        max_subtopics: int = 3,
        per_subtopic_limit: int = 3,
        top_chunks: int = 20,
    ):
        """
        Complete pipeline:
        1. Break down question
        2. Retrieve papers w/ caching + auto-processing
        3. Deduplicate papers
        4. Vector search over chunks
        5. Return RAGResult
        """

        yield RAGPipelineEvent(
            type="step", data="Breaking down user question into subtopics..."
        )
        breakdown = await self.llm.breakdown_user_question(user_question=query)
        subtopics = breakdown.subtopics[:max_subtopics]
        logger.info(f"Subtopics: {subtopics}")
        yield RAGPipelineEvent(type="subtopics", data={"subtopics": subtopics})

        yield RAGPipelineEvent(type="step", data=f"Searching for relevant papers...")
        paper_service = PaperRetrievalService(self.db_session)
        all_papers = []

        for idx, subtopic in enumerate(subtopics, 1):
            try:
                papers = await self.retriever.search(
                    query=subtopic,
                    limit=per_subtopic_limit,
                    services=[RetrievalServiceType.SEMANTIC],
                )
                all_papers.extend(papers)
                logger.info(f"Subtopic {idx}: retrieved {len(papers)} papers")
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error retrieving papers for '{subtopic}': {e}")

        # Deduplicate papers
        papers = deduplicate_papers(all_papers)
        logger.info(f"Total unique papers: {len(papers)}")

        if not papers:
            yield RAGPipelineEvent(type="result", data=RAGResult(papers=[], chunks=[]))
            return

        processed_papers = await self.processor.process_papers(papers)
        paper_ids = [
            str(p.paper_id)
            for p in papers
            if processed_papers.get(str(p.paper_id), False)
        ]

        chunks = await paper_service.get_relevant_chunks(
            query=query, paper_ids=paper_ids, limit=top_chunks
        )
        logger.info(f"Found {len(chunks)} relevant chunks")

        yield RAGPipelineEvent(
            type="result", data=RAGResult(papers=papers, chunks=chunks)
        )

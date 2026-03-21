"""
Scoped Paper Pipeline

Optimized pipeline for question answering within a known set of papers.
Skips query decomposition and uses a lightweight query rewrite, then:
1) searches chunks in provided paper IDs
2) reranks chunks
3) ranks papers from the same scope
"""

from datetime import datetime
from typing import AsyncGenerator, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.container import ServiceContainer
from app.domain.chunks.schemas import ChunkRetrieved
from app.domain.papers import LoadOptions
from app.llm import get_llm_service
from app.processor.schemas import RankedPaper
from app.rag_pipeline.schemas import RAGEventType, RAGPipelineEvent, RAGResult
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class ScopedPipeline:
    """Pipeline for scoped QA over provided paper IDs."""

    def __init__(
        self,
        db_session: AsyncSession,
        container: Optional[ServiceContainer] = None,
        llm_service=None,
    ):
        self.db_session = db_session
        self.container = container or ServiceContainer(db_session)
        self.chunk_service = self.container.chunk_service
        self.repository = self.container.paper_repository
        self.ranking_service = self.container.ranking_service
        self.llm = llm_service or get_llm_service()

    async def run_scoped_search_workflow(
        self,
        query: str,
        paper_ids: List[str],
        top_chunks: int = 40,
        top_papers: int = 20,
        enable_reranking: bool = True,
    ) -> AsyncGenerator[RAGPipelineEvent, None]:
        """Run scoped retrieval/ranking workflow on explicit paper IDs."""
        normalized_paper_ids = [str(pid).strip() for pid in paper_ids if str(pid).strip()]

        if not normalized_paper_ids:
            yield RAGPipelineEvent(type=RAGEventType.RESULT, data=RAGResult(papers=[], chunks=[]))
            return

        yield RAGPipelineEvent(
            type=RAGEventType.PROCESSING,
            data={"message": "Rewriting query for scoped paper search"},
        )

        rewritten_query = await self._rewrite_query_lightweight(query)

        yield RAGPipelineEvent(
            type=RAGEventType.SEARCHING,
            data={
                "queries": [rewritten_query],
                "original": query,
                "paper_ids": normalized_paper_ids,
                "scope": "provided_papers",
            },
        )

        yield RAGPipelineEvent(
            type=RAGEventType.PROCESSING,
            data={"message": "Searching chunks in selected papers"},
        )

        chunks = await self.chunk_service.hybrid_search_chunks(
            query=rewritten_query,
            paper_ids=normalized_paper_ids,
            limit=top_chunks,
            bm25_weight=0.4,
            semantic_weight=0.6,
        )

        papers_with_chunks = {str(chunk.paper_id) for chunk in chunks}
        scoped_papers, _ = await self.repository.get_papers(
            skip=0,
            limit=max(len(normalized_paper_ids), 1),
            paper_ids=normalized_paper_ids,
            load_options=LoadOptions(),
        )
        papers_without_chunks = [
            paper
            for paper in scoped_papers
            if str(paper.paper_id) not in papers_with_chunks and paper.abstract
        ]

        for paper in papers_without_chunks:
            virtual_chunk = ChunkRetrieved(
                chunk_id=f"{paper.paper_id}_abstract",
                paper_id=str(paper.paper_id),
                text=paper.abstract,
                token_count=len(paper.abstract.split()),
                chunk_index=0,
                section_title="Abstract",
                page_number=None,
                label="abstract",
                level=0,
                id=int(paper.id),
                char_start=None,
                char_end=None,
                docling_metadata=None,
                embedding=None,
                created_at=datetime.now(),
                relevance_score=0.0,
            )
            chunks.append(virtual_chunk)

        if enable_reranking and chunks:
            try:
                chunks = self.ranking_service.rerank_chunks(rewritten_query, chunks)
            except Exception as e:
                logger.warning(f"Scoped chunk reranking failed: {e}")

        yield RAGPipelineEvent(
            type=RAGEventType.RANKING,
            data={
                "total_papers": len(normalized_paper_ids),
                "total_chunks": len(chunks),
                "scope": "provided_papers",
            },
        )

        ranked_papers = await self._rank_scoped_papers_from_chunks(
            scoped_paper_ids=normalized_paper_ids,
            chunks=chunks,
            top_papers=top_papers,
        )

        yield RAGPipelineEvent(
            type=RAGEventType.PROCESSING,
            data={
                "message": "Scoped answer uses chunk-level citations",
                "citation_format": "(cite:paper_id|chunk_id)",
                "prompt_name": "generate_answer_scoped",
            },
        )

        yield RAGPipelineEvent(
            type=RAGEventType.RESULT,
            data=RAGResult(
                papers=ranked_papers,
                chunks=chunks[:top_chunks],
            ),
        )

    async def _rewrite_query_lightweight(self, query: str) -> str:
        """Use existing LLM decomposition endpoint in lightweight mode for clarified query."""
        try:
            breakdown = await self.llm.decompose_user_query_v2(
                user_question=query,
                num_subtopics=1,
                conversation_history=None,
            )
            clarified = breakdown.clarified_question
            return clarified or query
        except Exception as e:
            logger.warning(f"Scoped query rewrite failed, fallback to original query: {e}")
            return query

    async def _rank_scoped_papers_from_chunks(
        self,
        scoped_paper_ids: List[str],
        chunks,
        top_papers: int,
    ) -> List[RankedPaper]:
        """Build lightweight scoped paper ranking using chunk relevance aggregation."""
        papers, _ = await self.repository.get_papers(
            skip=0,
            limit=max(top_papers, 1),
            paper_ids=scoped_paper_ids,
            load_options=LoadOptions(authors=True, journal=True),
        )

        if not papers:
            return []

        score_map = {str(p.paper_id): 0.0 for p in papers}
        for chunk in chunks:
            pid = str(getattr(chunk, "paper_id", ""))
            if pid in score_map:
                score_map[pid] += float(getattr(chunk, "relevance_score", 0.0) or 0.0)

        ranked: List[RankedPaper] = []
        for p in papers:
            pid = str(p.paper_id)
            score = score_map.get(pid, 0.0)
            ranked.append(
                RankedPaper(
                    id=int(p.id),
                    paper_id=pid,
                    paper=p,
                    relevance_score=score,
                    ranking_scores={"scoped_chunk_relevance": score},
                )
            )

        ranked.sort(key=lambda item: item.relevance_score, reverse=True)
        return ranked[:top_papers]
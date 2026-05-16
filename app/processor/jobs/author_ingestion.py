"""Author-scoped ingestion workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from app.domain.chunks.repository import ChunkRepository
from app.domain.papers.repository import PaperRepository
from app.extensions.logger import create_logger
from app.models.papers import DBPaper
from app.processor.paper_processor import PaperProcessor
from app.rag_pipeline.schemas import PipelineResult
from app.retriever.provider.semantic_scholar_provider import SemanticScholarProvider
from app.retriever.schemas.openalex import OAAuthorResponse
from app.retriever.service import RetrievalService, RetrievalServiceType

if TYPE_CHECKING:
    from app.domain.authors.service import AuthorService

logger = create_logger(__name__)


class AuthorIngestionJobService:
    """Workflow service for fetching and indexing papers for one author."""

    def __init__(
        self,
        author_service: "AuthorService",
        retriever: Optional[RetrievalService] = None,
        processor: Optional[PaperProcessor] = None,
    ):
        self.author_service = author_service
        self.db = author_service.db
        self.repository = author_service.repository
        self.retriever = retriever or RetrievalService(self.db)

        if processor is None:
            paper_repo = PaperRepository(self.db)
            chunk_repo = ChunkRepository(self.db)
            processor = PaperProcessor(
                repository=paper_repo,
                chunk_repository=chunk_repo,
                retrieval_service=self.retriever,
            )
        self.processor = processor

    async def run(
        self,
        author_id: str,
        oa_author_id: Optional[str] = None,
        limit: int = 500,
        compute_relationships: bool = True,
    ) -> Any:
        """Fetch author papers, index their metadata, and update author metrics."""
        logger.info("Starting author ingestion pipeline for %s", author_id)

        db_author = await self.repository.get_author(author_id)
        resolved_oa_author_id = oa_author_id
        if not resolved_oa_author_id and db_author and getattr(db_author, "openalex_id", None):
            resolved_oa_author_id = str(db_author.openalex_id).removeprefix("https://openalex.org/")

        papers = await self.retriever.get_author_papers(author_id=author_id)
        author = await self.retriever.get_author(resolved_oa_author_id) if resolved_oa_author_id else None

        if limit and len(papers) > limit:
            papers = papers[:limit]
        logger.info("Retrieved %s papers from API", len(papers))

        if not papers:
            await self._update_author_without_papers(author_id, author)
            return PipelineResult(papers=[], author=author)

        processed_count = 0
        papers_with_metadata: List[tuple[DBPaper, List[Dict[str, Any]]]] = []
        for paper in papers:
            try:
                db_paper = await self.processor.paper_service.ingest_paper_metadata(
                    paper,
                    defer_enrichment=True,
                )

                if db_paper:
                    processed_count += 1
                    papers_with_metadata.append(
                        (db_paper, self._authors_payload(paper.authors or []))
                    )
            except Exception as exc:
                logger.error(
                    "Error ensuring paper %s: %s",
                    paper.paper_id,
                    exc,
                    exc_info=True,
                )

        logger.info("Processed %s/%s papers successfully", processed_count, len(papers))

        if papers_with_metadata:
            try:
                enrichment_stats = (
                    await self.processor.paper_service.batch_link_paper_relationships(
                        papers_with_metadata
                    )
                )
                logger.info("Batch linked author workflow papers: %s", enrichment_stats)
            except Exception as exc:
                logger.warning(
                    "Batch linking failed for author %s: %s",
                    author_id,
                    exc,
                    exc_info=True,
                )

        await self.author_service.compute_career_metrics(author_id)

        update_payload = self._author_update_payload(author)
        await self._add_conflict_flag(author_id, author, update_payload)
        await self.repository.update_author(author_id, update_payload)

        if compute_relationships:
            try:
                await self.author_service.compute_author_relationships(author_id)
                logger.info("Computed author relationships for %s", author_id)
            except Exception as exc:
                logger.warning(
                    "Failed to compute author relationships: %s",
                    exc,
                    exc_info=True,
                )

        logger.info("Author ingestion pipeline completed for %s", author_id)
        return PipelineResult(
            papers=papers,
            author=author if isinstance(author, OAAuthorResponse) else None,
        )

    async def _update_author_without_papers(
        self,
        author_id: str,
        author: Optional[OAAuthorResponse],
    ) -> None:
        logger.warning("No papers found for author %s", author_id)
        if author is None:
            return
        await self.repository.update_author(author_id, self._author_update_payload(author))

    def _author_update_payload(self, author: Optional[OAAuthorResponse]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "last_paper_indexed_at": datetime.now(timezone.utc),
            "is_processed": True,
        }

        summary_stats = getattr(author, "summary_stats", None) if author else None
        if isinstance(summary_stats, dict):
            i10_index = summary_stats.get("i10_index", summary_stats.get("i10Index"))
            if i10_index is not None:
                try:
                    payload["i10_index"] = int(i10_index)
                except (TypeError, ValueError):
                    pass

        oa_counts_raw = getattr(author, "counts_by_year", None) if author else None
        if isinstance(oa_counts_raw, list):
            openalex_counts_by_year: Dict[str, Dict[str, int]] = {}
            for item in oa_counts_raw:
                if not isinstance(item, dict):
                    continue
                year = item.get("year")
                if year is None:
                    continue
                try:
                    year_key = str(int(year))
                except (TypeError, ValueError):
                    continue
                openalex_counts_by_year[year_key] = {
                    "papers": int(item.get("works_count") or 0),
                    "citations": int(item.get("cited_by_count") or 0),
                }
            if openalex_counts_by_year:
                payload["openalex_counts_by_year"] = openalex_counts_by_year

        return payload

    async def _add_conflict_flag(
        self,
        author_id: str,
        author: Optional[OAAuthorResponse],
        update_payload: Dict[str, Any],
    ) -> None:
        semantic_citations: Optional[int] = None
        openalex_citations: Optional[int] = None

        if author is not None:
            try:
                openalex_citations = int(getattr(author, "cited_by_count", 0) or 0)
            except (TypeError, ValueError):
                openalex_citations = None

        try:
            semantic_provider = self.retriever.get_provider_as(
                RetrievalServiceType.SEMANTIC,
                SemanticScholarProvider,
            )
            semantic_map = await semantic_provider.get_multiple_authors([str(author_id)])
            sem_payload = semantic_map.get(str(author_id)) if semantic_map else None
            if isinstance(sem_payload, dict):
                sem_citations_raw = sem_payload.get("citationCount")
                if sem_citations_raw is not None:
                    semantic_citations = int(sem_citations_raw)
        except Exception as exc:
            logger.warning(
                "Failed semantic citation fetch for conflict check %s: %s",
                author_id,
                exc,
            )

        if semantic_citations is not None and openalex_citations is not None:
            conflict_threshold = 60.0
            baseline = max(int(semantic_citations), int(openalex_citations), 1)
            diff_ratio = abs(int(semantic_citations) - int(openalex_citations)) / baseline
            update_payload["is_conflict"] = diff_ratio >= (conflict_threshold / 100.0)

    @staticmethod
    def _authors_payload(authors: List[Any]) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for author in authors:
            if isinstance(author, dict):
                payload.append(author)
            elif hasattr(author, "model_dump"):
                payload.append(author.model_dump())
        return payload

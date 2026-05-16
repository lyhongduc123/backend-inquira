"""Background job functions for preprocessing workflows."""

from __future__ import annotations

from typing import List, Optional

from app.core.container import ServiceContainer
from app.core.db.database import async_session
from app.extensions.logger import create_logger

logger = create_logger(__name__)


async def run_bulk_search_task(
    job_id: str,
    search_query: str,
    target_count: int,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    fields_of_study: Optional[List[str]] = None,
    resume: bool = True,
):
    """Run bulk search preprocessing in a background-owned database session."""
    db = async_session()
    try:
        container = ServiceContainer(db)
        await container.preprocessing_service.process_bulk_search(
            job_id=job_id,
            search_query=search_query,
            target_count=target_count,
            year_min=year_min,
            year_max=year_max,
            fields_of_study=fields_of_study,
            resume=resume,
        )
    except Exception as exc:
        logger.error("Background bulk search task failed: %s", exc, exc_info=True)
    finally:
        await db.close()


async def run_repository_task(
    job_id: str,
    paper_ids: List[str],
    resume: bool = True,
):
    """Run repository preprocessing for explicit paper IDs."""
    db = async_session()
    try:
        container = ServiceContainer(db)
        service = container.preprocessing_service

        for paper_id in paper_ids:
            try:
                enriched_papers = await service.retriever.get_multiple_papers([paper_id])
                if not enriched_papers:
                    continue

                paper = enriched_papers[0]
                existing = await service.repository.get_paper_by_id(paper_id)
                if not existing:
                    await service.paper_service.ingest_paper_metadata(paper)

                await service.processor.process_single_paper(paper)
                logger.info("[Repository] Processed paper %s", paper_id)
            except Exception as exc:
                logger.error("[Repository] Error processing %s: %s", paper_id, exc)
    except Exception as exc:
        logger.error("Background repository task failed: %s", exc, exc_info=True)
    finally:
        await db.close()


async def run_embedding_backfill_task() -> dict:
    """Queue task wrapper for embedding backfill phase."""
    db = async_session()
    try:
        container = ServiceContainer(db)
        return await container.preprocessing_phase_service.run_embedding_backfill()
    finally:
        await db.close()


async def run_citation_linking_task(
    limit: int,
    references_limit: int,
    citations_limit: int,
) -> dict:
    """Queue task wrapper for citation linking phase."""
    db = async_session()
    try:
        container = ServiceContainer(db)
        return await container.preprocessing_phase_service.run_citation_linking(
            limit=limit,
            references_limit=references_limit,
            citations_limit=citations_limit,
        )
    finally:
        await db.close()


async def run_author_metrics_task(
    only_unprocessed: bool,
    conflict_threshold_percent: float,
    batch_size: int,
) -> dict:
    """Queue task wrapper for author metric computation phase."""
    db = async_session()
    try:
        container = ServiceContainer(db)
        return await container.preprocessing_phase_service.run_author_metrics(
            only_unprocessed=only_unprocessed,
            conflict_threshold_percent=conflict_threshold_percent,
            batch_size=batch_size,
        )
    finally:
        await db.close()


async def run_paper_tagging_task(
    limit: int,
    only_missing_tags: bool,
    candidate_labels: Optional[List[str]],
    category: str,
    min_confidence: float,
    max_tags_per_paper: int,
) -> dict:
    """Queue task wrapper for paper tag computation phase."""
    db = async_session()
    try:
        container = ServiceContainer(db)
        return await container.preprocessing_phase_service.run_paper_tagging(
            limit=limit,
            only_missing_tags=only_missing_tags,
            candidate_labels=candidate_labels,
            category=category,
            min_confidence=min_confidence,
            max_tags_per_paper=max_tags_per_paper,
        )
    finally:
        await db.close()


async def run_content_processing_task(limit: int) -> dict:
    """Queue task wrapper for content processing."""
    db = async_session()
    try:
        container = ServiceContainer(db)
        return await container.preprocessing_service.run_content_processing(limit=limit)
    finally:
        await db.close()


async def run_single_preprocessing_phase_task(
    run_embed: bool,
    run_process_content: bool,
    paper_ids: Optional[List[str]],
    limit: int,
) -> dict:
    """Queue task wrapper for single phase preprocessing."""
    db = async_session()
    try:
        from app.processor.preprocessing_single_phase import PreprocessingSinglePhaseService

        service = PreprocessingSinglePhaseService(db)
        return await service.run_preprocessing(
            run_embed=run_embed,
            run_process_content=run_process_content,
            paper_ids=paper_ids,
            limit=limit,
        )
    finally:
        await db.close()

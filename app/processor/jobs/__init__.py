"""Background job entry points for processor workflows."""

from app.processor.jobs.author_ingestion import AuthorIngestionJobService
from app.processor.jobs.preprocessing_jobs import (
    run_author_metrics_task,
    run_bulk_search_task,
    run_citation_linking_task,
    run_content_processing_task,
    run_embedding_backfill_task,
    run_paper_tagging_task,
    run_repository_task,
    run_single_preprocessing_phase_task,
)

__all__ = [
    "AuthorIngestionJobService",
    "run_author_metrics_task",
    "run_bulk_search_task",
    "run_citation_linking_task",
    "run_content_processing_task",
    "run_embedding_backfill_task",
    "run_paper_tagging_task",
    "run_repository_task",
    "run_single_preprocessing_phase_task",
]

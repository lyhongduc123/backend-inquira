"""
Background worker module for handling async tasks without blocking API responses.

Workers handle:
- Author enrichment (fetching papers from external APIs)
- Paper enrichment (authors, institutions, journals)
- Batch embedding generation
- Citation relationship computation
"""

from .task_queue import TaskQueue, BackgroundTask
from .enrichment_worker import EnrichmentWorker

__all__ = ["TaskQueue", "BackgroundTask", "EnrichmentWorker"]

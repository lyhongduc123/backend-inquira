"""Retriever mapping helpers from provider-normalized contracts."""

from __future__ import annotations

from typing import List

from app.domain.papers.types import PaperEnrichedDTO
from app.retriever.schemas import NormalizedPaperResult
from app.utils.transformers import batch_normalized_to_papers, normalized_to_paper


def paper_from_normalized_result(result: NormalizedPaperResult) -> PaperEnrichedDTO:
    """Map a normalized provider result into the internal enriched paper type."""

    return normalized_to_paper(result)


def papers_from_normalized_results(
    results: List[NormalizedPaperResult],
) -> List[PaperEnrichedDTO]:
    """Map normalized provider results into internal enriched paper types."""

    return batch_normalized_to_papers(results)

"""Helpers for search-time virtual chunks."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional, Set

from app.domain.chunks.types import ChunkRetrieved


def build_abstract_chunk(paper, relevance_score: float = 0.0) -> Optional[ChunkRetrieved]:
    """Create a transient chunk from a paper abstract when no stored chunk exists."""
    abstract = getattr(paper, "abstract", None)
    if not abstract:
        return None

    paper_id = str(getattr(paper, "paper_id"))
    return ChunkRetrieved(
        chunk_id=f"{paper_id}_abstract",
        paper_id=paper_id,
        text=abstract,
        token_count=len(abstract.split()),
        chunk_index=0,
        section_title="Abstract",
        page_number=None,
        label="abstract",
        level=0,
        id=int(getattr(paper, "id")),
        char_start=None,
        char_end=None,
        docling_metadata=None,
        embedding=None,
        created_at=datetime.now(),
        relevance_score=float(relevance_score),
    )


def append_missing_abstract_chunks(
    chunks: List[ChunkRetrieved],
    papers_with_scores: Iterable[tuple],
    *,
    score_multiplier: float = 0.0,
) -> List[ChunkRetrieved]:
    """Append abstract chunks for scored papers that have no retrieved chunks."""
    papers_with_chunks: Set[str] = {str(chunk.paper_id) for chunk in chunks}

    for paper, score in papers_with_scores:
        paper_id = str(getattr(paper, "paper_id"))
        if paper_id in papers_with_chunks:
            continue

        chunk = build_abstract_chunk(
            paper,
            relevance_score=float(score or 0.0) * score_multiplier,
        )
        if chunk:
            chunks.append(chunk)
            papers_with_chunks.add(paper_id)

    return chunks

"""Internal RAG pipeline result and event contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.domain.chunks.types import ChunkRetrieved
from app.domain.papers.types import PaperEnrichedDTO
from app.search.types import RankedPaper
from app.retriever.schemas.openalex import OAAuthorResponse


@dataclass
class RAGResult:
    """RAG pipeline result containing ranked papers and relevant chunks."""

    papers: List[RankedPaper]
    chunks: List[ChunkRetrieved]


@dataclass
class RAGPipelineEvent:
    type: str
    data: dict | str | RAGResult | None


@dataclass
class RAGPipelineContext:
    query: str
    search_queries: List[str] = field(default_factory=list)
    papers: List[PaperEnrichedDTO] = field(default_factory=list)
    filtered_papers: List[PaperEnrichedDTO] = field(default_factory=list)
    papers_with_hybrid_scores: List[tuple] = field(default_factory=list)
    processed_paper_ids: List[str] = field(default_factory=list)
    result_papers: List[RankedPaper] = field(default_factory=list)
    chunks: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    author: Optional[OAAuthorResponse] = None
    papers: List[PaperEnrichedDTO] = field(default_factory=list)

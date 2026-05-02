"""Agentic RAG toolset for chat agent orchestration."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.container import ServiceContainer
from app.domain.chunks.schemas import ChunkRetrieved
from app.domain.papers import LoadOptions, SearchFilterOptions
from app.extensions.logger import create_logger
from app.llm.schemas.chat import QueryIntent
from app.models.papers import DBPaper
from app.processor.schemas import RankedPaper
from app.rag_pipeline.schemas import RAGResult

logger = create_logger(__name__)


class AgentRAGTools:
    """Tool facade for agentic retrieval and ranking operations."""

    def __init__(self, container: ServiceContainer) -> None:
        self.container = container

    @staticmethod
    def infer_intent(query: str, filters: Optional[Dict[str, Any]] = None) -> QueryIntent:
        """Infer coarse query intent from query text and filter presence."""
        q = (query or "").lower()
        f = filters or {}

        if f.get("author") or f.get("author_name") or re.search(r"\bby\s+[a-z]", q):
            return QueryIntent.AUTHOR_PAPERS

        if any(token in q for token in ["compare", "comparison", "vs", "versus", "difference between"]):
            return QueryIntent.COMPARISON

        if any(token in q for token in ["foundational", "seminal", "first paper", "origin"]):
            return QueryIntent.FOUNDATIONAL

        return QueryIntent.COMPREHENSIVE_SEARCH

    @staticmethod
    def infer_filters(query: str, base_filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Infer lightweight filters from query without structured decomposition."""
        merged = dict(base_filters or {})
        q = (query or "").strip()

        if not merged.get("author") and not merged.get("author_name"):
            author_match = re.search(r"\bby\s+([A-Z][\w\-.']+(?:\s+[A-Z][\w\-.']+){0,3})", q)
            if author_match:
                merged["author_name"] = author_match.group(1).strip()

        if merged.get("year_min") is None:
            after_match = re.search(r"\b(?:after|since)\s+(19\d{2}|20\d{2})\b", q, flags=re.IGNORECASE)
            if after_match:
                merged["year_min"] = int(after_match.group(1))

        if merged.get("year_max") is None:
            before_match = re.search(r"\b(?:before|until)\s+(19\d{2}|20\d{2})\b", q, flags=re.IGNORECASE)
            if before_match:
                merged["year_max"] = int(before_match.group(1))

        return merged

    @staticmethod
    def select_tools(query: str, filters: Optional[Dict[str, Any]], intent: QueryIntent) -> List[str]:
        """Select ordered tools based on user request profile."""
        selected: List[str] = []
        q = (query or "").lower()
        has_filters = bool(filters)

        if has_filters or intent == QueryIntent.AUTHOR_PAPERS or any(
            token in q for token in ["exact", "keyword", "title", "author", "journal", "venue"]
        ):
            selected.append("keyword_search")

        selected.append("hybrid_search")

        if any(token in q for token in ["evidence", "grounded", "quote", "citation", "why"]):
            selected.append("rerank_chunks")
        else:
            selected.append("rerank_chunks")

        selected.append("rank_papers")
        return selected

    @staticmethod
    def _get_filter_value(filters: Optional[Dict[str, Any]], snake_key: str) -> Any:
        if not filters:
            return None

        if snake_key in filters:
            return filters.get(snake_key)

        parts = snake_key.split("_")
        camel_key = parts[0] + "".join(part.capitalize() for part in parts[1:])
        return filters.get(camel_key)

    def _build_filter_options(self, filters: Optional[Dict[str, Any]]) -> SearchFilterOptions:
        author_name = self._get_filter_value(filters, "author_name") or self._get_filter_value(filters, "author")
        year_min = self._get_filter_value(filters, "year_min")
        year_max = self._get_filter_value(filters, "year_max")
        venue = self._get_filter_value(filters, "venue")
        min_citations = self._get_filter_value(filters, "min_citation_count") or self._get_filter_value(filters, "min_citations")
        max_citations = self._get_filter_value(filters, "max_citation_count") or self._get_filter_value(filters, "max_citations")
        journal_quartile = self._get_filter_value(filters, "journal_quartile") or self._get_filter_value(filters, "journal_rank")
        field_of_study = self._get_filter_value(filters, "field_of_study")

        fields_of_study: Optional[List[str]] = None
        if isinstance(field_of_study, str) and field_of_study.strip():
            fields_of_study = [field_of_study.strip()]
        else:
            fields = self._get_filter_value(filters, "fields_of_study")
            if isinstance(fields, list):
                fields_of_study = [str(f).strip() for f in fields if str(f).strip()]

        return SearchFilterOptions(
            author_name=author_name,
            year_min=year_min,
            year_max=year_max,
            venue=venue,
            min_citation_count=min_citations,
            max_citation_count=max_citations,
            journal_quartile=journal_quartile,
            field_of_study=fields_of_study,
        )

    async def keyword_search(
        self,
        *,
        query: str,
        filters: Optional[Dict[str, Any]],
        limit: int,
    ) -> List[Tuple[DBPaper, float]]:
        """Run BM25 keyword search with normalized filters."""
        return await self.container.paper_repository.bm25_search(
            query=query,
            limit=limit,
            filter_options=self._build_filter_options(filters),
        )

    async def hybrid_search(
        self,
        *,
        query: str,
        filters: Optional[Dict[str, Any]],
        limit: int,
    ) -> List[Tuple[DBPaper, float]]:
        """Run fused hybrid search with normalized filters."""
        return await self.container.paper_service.hybrid_search(
            query=query,
            limit=limit,
            filter_options=self._build_filter_options(filters),
        )

    async def fetch_chunks(
        self,
        *,
        query: str,
        paper_ids: List[str],
        top_chunks: int,
        intent: QueryIntent,
    ) -> List[ChunkRetrieved]:
        """Retrieve chunks for selected papers."""
        bm25_weight = 0.4
        semantic_weight = 0.6

        if intent == QueryIntent.FOUNDATIONAL:
            bm25_weight = 0.6
            semantic_weight = 0.4

        return await self.container.chunk_service.hybrid_search_chunks(
            query=query,
            paper_ids=paper_ids,
            limit=top_chunks,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
        )

    def rerank_chunks(self, *, query: str, chunks: List[ChunkRetrieved]) -> List[ChunkRetrieved]:
        """Re-rank chunks using ranking service."""
        return self.container.ranking_service.rerank_chunks(query, chunks)

    def rank_papers(
        self,
        *,
        query: str,
        papers: List[DBPaper],
        chunks: List[ChunkRetrieved],
        paper_hybrid_scores: Dict[str, float],
    ) -> List[RankedPaper]:
        """Rank papers by relevance + authority."""
        try:
            return self.container.ranking_service.rank_papers(
                query=query,
                papers=papers,
                chunks=chunks,
                weights={"relevance": 0.7, "authority": 0.3},
            )
        except Exception as exc:
            logger.error("Paper ranking failed, falling back to hybrid scores: %s", exc)
            return [
                RankedPaper(
                    id=p.id,
                    paper_id=p.paper_id,
                    paper=p,
                    relevance_score=paper_hybrid_scores.get(p.paper_id, 0.0),
                    ranking_scores={"hybrid_score": paper_hybrid_scores.get(p.paper_id, 0.0)},
                )
                for p in papers
            ]

    @staticmethod
    def _fuse_rankings_with_rrf(
        rankings: List[List[Tuple[DBPaper, float]]],
        *,
        k: int = 60,
        limit: int = 100,
    ) -> List[Tuple[DBPaper, float]]:
        scores: Dict[str, float] = {}
        paper_map: Dict[str, DBPaper] = {}

        for ranking in rankings:
            for rank, (paper, _) in enumerate(ranking, start=1):
                paper_id = str(paper.paper_id)
                scores[paper_id] = scores.get(paper_id, 0.0) + (1.0 / (k + rank))
                if paper_id not in paper_map:
                    paper_map[paper_id] = paper

        ranked_ids = sorted(scores.keys(), key=lambda pid: scores[pid], reverse=True)[:limit]
        return [(paper_map[pid], scores[pid]) for pid in ranked_ids]

    async def run_agentic_rag(
        self,
        *,
        query: str,
        filters: Optional[Dict[str, Any]],
        tool_plan: List[str],
        intent: QueryIntent,
        top_papers: int = 50,
        top_chunks: int = 40,
    ) -> RAGResult:
        """Execute selected tools and compose RAG result."""
        rankings: List[List[Tuple[DBPaper, float]]] = []

        if "keyword_search" in tool_plan:
            rankings.append(await self.keyword_search(query=query, filters=filters, limit=max(top_papers * 2, 100)))

        if "hybrid_search" in tool_plan:
            rankings.append(await self.hybrid_search(query=query, filters=filters, limit=max(top_papers * 2, 100)))

        fused = self._fuse_rankings_with_rrf(rankings, limit=max(top_papers * 2, 100)) if rankings else []
        if not fused:
            return RAGResult(papers=[], chunks=[])

        paper_hybrid_scores = {paper.paper_id: score for paper, score in fused}
        paper_ids = [paper.paper_id for paper, _ in fused[:top_papers]]

        chunks = await self.fetch_chunks(
            query=query,
            paper_ids=paper_ids,
            top_chunks=top_chunks,
            intent=intent,
        )

        if "rerank_chunks" in tool_plan and chunks:
            chunks = self.rerank_chunks(query=query, chunks=chunks)

        enriched_papers, _ = await self.container.paper_repository.get_papers(
            paper_ids=paper_ids,
            load_options=LoadOptions(authors=True, journal=True, institutions=True),
        )

        if "rank_papers" in tool_plan:
            ranked_papers = self.rank_papers(
                query=query,
                papers=enriched_papers,
                chunks=chunks,
                paper_hybrid_scores=paper_hybrid_scores,
            )
        else:
            ranked_papers = [
                RankedPaper(
                    id=p.id,
                    paper_id=p.paper_id,
                    paper=p,
                    relevance_score=paper_hybrid_scores.get(p.paper_id, 0.0),
                    ranking_scores={"hybrid_score": paper_hybrid_scores.get(p.paper_id, 0.0)},
                )
                for p in enriched_papers
            ]

        result_paper_ids = {str(p.paper_id) for p in ranked_papers}
        result_chunks = [chunk for chunk in chunks if str(chunk.paper_id) in result_paper_ids]

        return RAGResult(
            papers=ranked_papers[:top_papers],
            chunks=result_chunks[:top_chunks],
        )

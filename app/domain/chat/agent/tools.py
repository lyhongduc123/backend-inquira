"""Agent tools for decomposition, retrieval, escalation, and evaluation."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from app.core.container import ServiceContainer
from app.domain.papers.repository import LoadOptions
from app.extensions.context_builder import ContextBuilder
from app.extensions.logger import create_logger
from app.llm.schemas.chat import GeneratedQueryPlanResponse, QueryIntent
from app.models.papers import DBPaper
from app.rag_pipeline.schemas import RAGEventType, RAGResult, SearchWorkflowConfig
from app.search.virtual_chunks import build_abstract_chunk
from app.search.types import RankedPaper
from app.utils.identifier_normalization import normalize_external_ids

logger = create_logger(__name__)


def extract_dois_from_text(text: str) -> List[str]:
    """Extract DOI strings from free text."""
    doi_pattern = r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b"
    return re.findall(doi_pattern, text, flags=re.IGNORECASE)


class AgentTools:
    """Encapsulates the retrieval and evaluation logic used by the agent graph."""

    def __init__(self, container: ServiceContainer) -> None:
        self.container = container

    async def decompose_query(
        self,
        *,
        query: str,
        conversation_history: List[Dict[str, str]],
        filters: Dict[str, Any],
    ) -> GeneratedQueryPlanResponse:
        """Normalize the user question into a retrieval plan."""
        breakdown = await self.container.llm_service.decompose_user_query_v3(
            user_question=query,
            conversation_history=conversation_history,
        )

        raw_intent = breakdown.intent
        intent_value = (
            raw_intent.value
            if isinstance(raw_intent, QueryIntent)
            else str(raw_intent or "").strip().lower()
        )

        merged_filters = dict(filters)
        if breakdown.filters:
            merged_filters.update(breakdown.filters)

        dois: List[str] = []
        seen_dois: set[str] = set()
        for doi in [*extract_dois_from_text(query), *(breakdown.dois or [])]:
            cleaned_doi = str(doi or "").strip()
            if not cleaned_doi or cleaned_doi in seen_dois:
                continue
            seen_dois.add(cleaned_doi)
            dois.append(cleaned_doi)

        if intent_value in {"system", "gibberish"}:
            return GeneratedQueryPlanResponse(
                original_question=query,
                clarified_question=query,
                hybrid_queries=[query],
                specific_papers=[],
                has_doi=bool(dois),
                dois=dois,
                intent=(
                    QueryIntent.SYSTEM
                    if intent_value == "system"
                    else QueryIntent.GIBBERISH
                ),
                skip=[],
                filters=merged_filters,
            )

        hybrid_queries = breakdown.hybrid_queries or [
            breakdown.clarified_question or query
        ]
        return GeneratedQueryPlanResponse(
            original_question=query,
            clarified_question=breakdown.clarified_question or query,
            hybrid_queries=hybrid_queries,
            specific_papers=list(breakdown.specific_papers or []),
            has_doi=bool(breakdown.has_doi or dois),
            dois=dois,
            intent=breakdown.intent or QueryIntent.COMPREHENSIVE_SEARCH,
            skip=list(getattr(breakdown, "skip", []) or []),
            filters=merged_filters,
        )

    async def retrieve_local(
        self,
        *,
        query: str,
        filters: Dict[str, Any],
        plan: GeneratedQueryPlanResponse,
        search_queries: Optional[List[str]] = None,
    ) -> RAGResult:
        """Route local retrieval to the DOI/title or database pipeline."""
        queries = list(search_queries or plan.hybrid_queries or [query])
        merged_filters = {
            **(plan.filters if isinstance(plan.filters, dict) else {}),
            **(filters or {}),
        }

        if plan.specific_papers or plan.dois:
            return await self.container.doi_title_pipeline.run_doi_title_lookup(
                original_query=query,
                specific_papers=list(plan.specific_papers),
                dois=list(plan.dois),
                rerank_query=plan.clarified_question or query,
                top_papers=40,
                top_chunks=40,
                enable_reranking=True,
            ) or RAGResult(papers=[], chunks=[])

        result: Optional[RAGResult] = None
        async for (
            event
        ) in self.container.database_pipeline.run_database_search_workflow(
            SearchWorkflowConfig(
                query=plan.clarified_question or query,
                search_queries=queries,
                intent=plan.intent or QueryIntent.COMPREHENSIVE_SEARCH,
                filters=merged_filters,
                top_papers=30,
                top_chunks=20,
                enable_reranking=True,
                enable_paper_ranking=True,
            )
        ):
            if event.type == RAGEventType.RESULT:
                result = event.data if isinstance(event.data, RAGResult) else None

        return result or RAGResult(papers=[], chunks=[])

    async def expand_queries(
        self,
        *,
        query: str,
        rag_result: Optional[RAGResult],
        existing_queries: Sequence[str],
    ) -> List[str]:
        """Generate up to two short follow-up queries that cover missing evidence."""
        if not query.strip():
            return []

        evidence_lines: List[str] = []
        if rag_result and rag_result.chunks:
            for idx, chunk in enumerate(rag_result.chunks[:6], start=1):
                chunk_text = self._truncate_text(getattr(chunk, "text", ""), 240)
                if chunk_text:
                    evidence_lines.append(f"{idx}. {chunk_text}")

        system_prompt = (
            "You are an academic retrieval gap analyzer. Given the clarified question and current evidence snippets, output concise missing-information queries. "
            'Return ONLY JSON: {"gap_queries": [string], "reason": string}.'
        )
        payload = {
            "clarified_question": query,
            "existing_queries": list(existing_queries),
            "evidence": evidence_lines,
            "constraints": {
                "max_queries": 2,
                "avoid_repeating_existing_queries": True,
                "query_style": "short academic search phrases",
            },
        }

        data = self.container.llm_service.prompt_json(
            system_prompt=system_prompt,
            user_payload=payload,
        )
        raw_gap_queries = data.get("gap_queries", [])
        if not isinstance(raw_gap_queries, list):
            return []

        normalized = self._normalize_queries(
            [str(item).strip() for item in raw_gap_queries if str(item).strip()]
        )
        existing_lower = {item.lower() for item in existing_queries}
        return [item for item in normalized if item.lower() not in existing_lower][:2]
    
    async def expand_external_queries(
        self,
        *,
        query: str,
        rag_result: Optional[RAGResult],
        existing_queries: Sequence[str],
    ) -> List[str]:
        if not query.strip():
            return []

        evidence_lines: List[str] = []
        if rag_result and rag_result.chunks:
            for idx, chunk in enumerate(rag_result.chunks[:6], start=1):
                chunk_text = self._truncate_text(getattr(chunk, "text", ""), 240)
                if chunk_text:
                    evidence_lines.append(f"{idx}. {chunk_text}")

        system_prompt = (
            "You are an academic retrieval professional. Given the clarified question and current evidence snippets, output queries that would be effective for searching external databases like Semantic Scholar, OpenAlex. The queries can be paper titles, author names, or keyword phrases that are relevant or might give answers to the question. "
            'Return ONLY JSON: {"queries": [string], "reason": string}.'
        )
        payload = {
            "clarified_question": query,
            "existing_queries": list(existing_queries),
            "evidence": evidence_lines,
            "constraints": {
                "max_queries": 4,
                "avoid_repeating_existing_queries": True,
            },
        }
        data = self.container.llm_service.prompt_json(
            system_prompt=system_prompt,
            user_payload=payload,
        )
        raw_gap_queries = data.get("queries", [])
        if not isinstance(raw_gap_queries, list):
            return []

        normalized = self._normalize_queries(
            [str(item).strip() for item in raw_gap_queries if str(item).strip()]
        )
        existing_lower = {item.lower() for item in existing_queries}
        return [item for item in normalized if item.lower() not in existing_lower][:2]

    async def retrieve_external(
        self,
        *,
        query: str,
        filters: Dict[str, Any],
        search_queries: Optional[List[str]] = None,
    ) -> RAGResult:
        """Search OpenAlex, enrich with Semantic Scholar, ingest papers, then resolve them locally."""
        external_queries = list(search_queries or [query])
        normalized_results: List[Any] = []
        seen_paper_ids: set[str] = set()

        for external_query in external_queries[:2]:
            query_results, _ = await self.container.retrieval_service.hybrid_search(
                query=external_query,
                s2_limit=20,
                filters=filters,
            )
            for paper in query_results:
                paper_id = str(getattr(paper, "paper_id", "") or "").strip()
                if not paper_id or paper_id in seen_paper_ids:
                    continue
                seen_paper_ids.add(paper_id)
                normalized_results.append(paper)

        if not normalized_results:
            return RAGResult(papers=[], chunks=[])

        ingested_dois: List[str] = []
        for paper in normalized_results:
            try:
                db_paper = await self.container.paper_service.ingest_paper_metadata(
                    paper,
                    defer_enrichment=False,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to ingest external paper %s: %s", paper.paper_id, exc
                )
                continue

            if db_paper:
                external_ids = normalize_external_ids(
                    getattr(paper, "external_ids", None) or {}
                )
                doi = str(external_ids.get("doi") or "").strip()
                if doi:
                    ingested_dois.append(doi)
                    
        if not ingested_dois:
            return RAGResult(papers=[], chunks=[])

        db_papers = await self.container.paper_service.get_papers_by_dois(
            ingested_dois[:20],
            load_options=LoadOptions(authors=True, journal=True, conference=True),
        )

        ranked_papers = [
            RankedPaper(
                id=paper.id,
                paper_id=paper.paper_id,
                paper=paper,
                relevance_score=1.0 / (index + 1),
                ranking_scores={"external_doi_score": 1.0 / (index + 1)},
            )
            for index, paper in enumerate(db_papers)
        ]
        
        virtual_chunks = [
            build_abstract_chunk(paper) for paper in db_papers
        ]
        filtered_chunks = [c for c in virtual_chunks if c is not None]

        return RAGResult(papers=ranked_papers, chunks=filtered_chunks)

    def merge_results_with_rrf(self, results_array: List[RAGResult]) -> RAGResult:
        """Merge multiple results using Reciprocal Rank Fusion based on chunk scores."""
        if not results_array:
            return RAGResult(papers=[], chunks=[])

        if len(results_array) == 1:
            result = results_array[0]
            return RAGResult(
                papers=self._sort_papers_by_chunk_relevance(
                    result.papers,
                    result.chunks,
                ),
                chunks=sorted(
                    result.chunks,
                    key=lambda chunk: float(getattr(chunk, "relevance_score", 0) or 0),
                    reverse=True,
                ),
            )
        
        # Build RRF scores for chunks across all results
        chunk_rrf_scores: Dict[int, float] = {}  # chunk id -> RRF score
        chunk_map: Dict[int, Any] = {}  # chunk id -> chunk object
        
        for result in results_array:
            if result.chunks:
                for rank, chunk in enumerate(result.chunks, start=1):
                    chunk_db_id = getattr(chunk, "id", None)
                    if chunk_db_id is None:
                        continue
                    
                    rrf_score = 1.0 / (rank + 60)
                    chunk_rrf_scores[chunk_db_id] = chunk_rrf_scores.get(chunk_db_id, 0) + rrf_score
                    
                    if chunk_db_id not in chunk_map:
                        chunk_map[chunk_db_id] = chunk
        
        paper_rrf_scores: Dict[int, float] = {}  # paper id -> RRF score
        paper_map: Dict[int, Any] = {}  # paper id -> paper object
        
        for result in results_array:
            if result.papers:
                for rank, paper in enumerate(result.papers, start=1):
                    paper_db_id = getattr(paper, "id", None)
                    if paper_db_id is None:
                        continue
                    
                    rrf_score = 1.0 / (rank + 60)
                    paper_rrf_scores[paper_db_id] = paper_rrf_scores.get(paper_db_id, 0) + rrf_score
                    
                    if paper_db_id not in paper_map:
                        paper_map[paper_db_id] = paper
        
        sorted_chunks = sorted(
            [(cid, chunk_map[cid], score) for cid, score in chunk_rrf_scores.items()],
            key=lambda x: x[2],
            reverse=True,
        )
        sorted_papers = sorted(
            [(pid, paper_map[pid], score) for pid, score in paper_rrf_scores.items()],
            key=lambda x: x[2],
            reverse=True,
        )
        
        max_chunk_score = sorted_chunks[0][2] if sorted_chunks else 1.0
        max_paper_score = sorted_papers[0][2] if sorted_papers else 1.0
        
        merged_chunks = []
        for _, chunk, rrf_score in sorted_chunks:
            if hasattr(chunk, "ranking_scores"):
                chunk.ranking_scores = {
                    **(getattr(chunk, "ranking_scores", None) or {}),
                    "rrf_score": rrf_score / max_chunk_score if max_chunk_score > 0 else 0.0,
                }
            merged_chunks.append(chunk)
        
        merged_papers = []
        for _, paper, rrf_score in sorted_papers:
            if hasattr(paper, "ranking_scores"):
                paper.ranking_scores = {
                    **(paper.ranking_scores or {}),
                    "rrf_score": rrf_score / max_paper_score if max_paper_score > 0 else 0.0,
                }
            merged_papers.append(paper)

        if merged_chunks:
            merged_chunks = sorted(
                merged_chunks,
                key=lambda chunk: float(getattr(chunk, "relevance_score", 0) or 0),
                reverse=True,
            )
            merged_papers = self._sort_papers_by_chunk_relevance(
                merged_papers,
                merged_chunks,
            )
        else:
            for paper in merged_papers:
                rrf_score = (paper.ranking_scores or {}).get("rrf_score", 0.0)
                paper.relevance_score = float(rrf_score or 0.0)
        
        return RAGResult(papers=merged_papers, chunks=merged_chunks)

    async def rerank_external_results(self, result: RAGResult, query: str) -> RAGResult:
        """Apply reranker to external retrieval results to improve ranking."""
        if not result or not result.chunks:
            return result
        
        try:
            ranking_service = self.container.ranking_service
            reranked_chunks = ranking_service.rerank_chunks(
                query=query,
                chunks=result.chunks,
            )
            result.chunks = reranked_chunks
            result.papers = self._sort_papers_by_chunk_relevance(
                result.papers,
                reranked_chunks,
            )
        except Exception as exc:
            logger.warning("Failed to rerank chunks: %s", exc)
        
        return result

    @staticmethod
    def _sort_papers_by_chunk_relevance(
        papers: List[RankedPaper],
        chunks: List[Any],
    ) -> List[RankedPaper]:
        """Sort papers by the best reranked chunk/abstract score for each paper."""
        if not papers:
            return []

        best_chunk_scores: Dict[str, float] = {}
        for chunk in chunks:
            paper_id = str(getattr(chunk, "paper_id", "") or "")
            if not paper_id:
                continue
            score = float(getattr(chunk, "relevance_score", 0) or 0)
            best_chunk_scores[paper_id] = max(
                best_chunk_scores.get(paper_id, 0.0),
                score,
            )

        has_chunk_scores = bool(best_chunk_scores)
        for paper in papers:
            paper_id = str(paper.paper_id)
            chunk_score = best_chunk_scores.get(paper_id)
            relevance_score = chunk_score if chunk_score is not None else 0.0
            if has_chunk_scores:
                paper.relevance_score = relevance_score
            paper.ranking_scores = {
                **(paper.ranking_scores or {}),
                "rerank_score": relevance_score,
            }

        return sorted(
            papers,
            key=lambda paper: float(paper.relevance_score or 0),
            reverse=True,
        )

    def evaluate_sufficiency(self, *, result: Optional[RAGResult], stage: str) -> bool:
        """Check whether the current result is sufficient for answer generation.
        The new workflow uses RRF merging instead of sufficiency gates.
        """
        if not result or (not result.papers and not result.chunks):
            return False

        scores = [float(c.relevance_score or 0) for c in result.chunks]
        if not scores:
            return bool(result.papers)

        top_score = scores[0]
        strong_chunks = sum(s >= 0.55 for s in scores[:5])
        score_margin = scores[0] - scores[1] if len(scores) > 1 else scores[0]

        if stage == "local":
            return (
                top_score >= 0.72
                and strong_chunks >= 3
                and score_margin >= 0.05
            )

        if stage == "external":
            return (
                top_score >= 0.60
                and strong_chunks >= 2
            )
        return False

    async def build_answer_context(self, result: RAGResult) -> Dict[str, Any]:
        """Build the response context payload from a retrieval result."""
        response_builder = ContextBuilder()
        context_str, _ = response_builder.build_context_from_results(result)
        return {
            "context": context_str,
            "context_chunks": response_builder.extract_context_chunks_from_results(
                result
            ),
            "paper_snapshots": response_builder.extract_metadata_from_results(result),
            "retrieved_paper_ids": response_builder.get_retrieved_paper_ids(result),
        }

    async def load_db_papers_by_ids(self, paper_ids: List[str]) -> List[DBPaper]:
        """Load DB papers in the requested order."""
        normalized_ids = [
            str(paper_id).strip() for paper_id in paper_ids if str(paper_id).strip()
        ]
        if not normalized_ids:
            return []

        papers, _ = await self.container.paper_repository.get_papers(
            skip=0,
            limit=len(normalized_ids),
            paper_ids=normalized_ids,
            load_options=LoadOptions.none(),
        )
        paper_map = {str(paper.paper_id): paper for paper in papers}
        return [
            paper_map[paper_id] for paper_id in normalized_ids if paper_id in paper_map
        ]

    @staticmethod
    def _normalize_queries(queries: Optional[List[str]]) -> List[str]:
        if not queries:
            return []

        normalized: List[str] = []
        seen: set[str] = set()
        for query in queries:
            value = (query or "").strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(value)
        return normalized

    @staticmethod
    def _truncate_text(text: Optional[str], max_chars: int) -> str:
        value = (text or "").strip()
        if not value:
            return ""
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 1].rstrip() + "…"


RetrievalTools = AgentTools

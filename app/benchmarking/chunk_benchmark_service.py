"""Chunk-search benchmark service for screened vs flat retrieval comparison."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


from app.benchmarking.schemas import (
    ChunkSearchBenchmarkReport,
    ChunkSearchBenchmarkRequest,
    ChunkSearchBenchmarkSummary,
    ChunkSearchModeResult,
    ChunkSearchQueryComparison,
    SearchStageMetrics,
)
from app.core.singletons import get_ranking_service
from app.db.database import db_session_context
from app.domain.chunks.repository import ChunkRepository
from app.domain.papers.repository import PaperRepository
from app.models.benchmark_corpus import DBBenchmarkPaper
from app.processor.services.embeddings import get_embedding_service
from app.utils.benchmark_utils import (
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
    mask_db_url,
)
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class ChunkSearchBenchmarkService:
    """Execute chunk retrieval benchmark in isolated environment."""

    def __init__(self) -> None:
        self.embedding_service = get_embedding_service()
        self.ranking_service = get_ranking_service()

    async def run(self, request: ChunkSearchBenchmarkRequest) -> ChunkSearchBenchmarkReport:
        """Run screened-vs-flat chunk retrieval benchmark."""
        benchmark_id = str(uuid4())
        benchmark_db_url = request.benchmark_db_url or os.getenv("BEIR_TEST_DATABASE_URL")

        if not benchmark_db_url:
            raise ValueError(
                "benchmark_db_url is required. Set request.benchmark_db_url or BEIR_TEST_DATABASE_URL."
            )

        data_path = util.download_and_unzip(
            f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{request.dataset_name}.zip",
            out_dir="beir_datasets",
        )
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=request.dataset_split)

        engine = create_async_engine(benchmark_db_url, echo=False, pool_pre_ping=True)
        session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        try:
            async with engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.run_sync(DBBenchmarkPaper.metadata.create_all)

            async with session_maker() as benchmark_session:
                await self._ensure_indexed(
                    benchmark_session=benchmark_session,
                    corpus=corpus,
                    force_reindex=request.force_reindex,
                )

            query_ids = list(queries.keys())[: request.max_queries]

            screened_results: List[ChunkSearchModeResult] = []
            flat_results: List[ChunkSearchModeResult] = []
            per_query: List[ChunkSearchQueryComparison] = []

            for query_id in query_ids:
                query_text = queries[query_id]
                relevant_ids = set(qrels.get(query_id, {}).keys())

                screened = await self._evaluate_mode(
                    query=query_text,
                    relevant_paper_ids=relevant_ids,
                    request=request,
                    strategy="screened",
                )
                flat = await self._evaluate_mode(
                    query=query_text,
                    relevant_paper_ids=relevant_ids,
                    request=request,
                    strategy="flat",
                )

                screened_results.append(screened)
                flat_results.append(flat)
                per_query.append(
                    ChunkSearchQueryComparison(
                        query_id=query_id,
                        query=query_text,
                        relevant_paper_count=len(relevant_ids),
                        screened=screened,
                        flat=flat,
                    )
                )

            screened_summary = self._summarize("screened", screened_results)
            flat_summary = self._summarize("flat", flat_results)

            output_dir = Path(request.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = output_dir / f"chunk_benchmark_{request.dataset_name}_{benchmark_id}.json"

            report = ChunkSearchBenchmarkReport(
                benchmark_id=benchmark_id,
                dataset_name=request.dataset_name,
                dataset_split=request.dataset_split,
                benchmark_db_url_used=mask_db_url(benchmark_db_url),
                separate_database_used=True,
                query_count=len(query_ids),
                screened=screened_summary,
                flat=flat_summary,
                per_query=per_query,
                created_at=datetime.utcnow(),
                artifact_path=str(artifact_path),
            )

            artifact_path.write_text(
                json.dumps(report.model_dump(mode="json", by_alias=True), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            return report
        finally:
            await engine.dispose()

    async def _ensure_indexed(
        self,
        benchmark_session: AsyncSession,
        corpus: Dict[str, Dict[str, Any]],
        force_reindex: bool,
    ) -> None:
        """Index benchmark corpus in isolated benchmark table."""
        existing = await benchmark_session.execute(select(DBBenchmarkPaper).limit(1))
        has_data = existing.scalar_one_or_none() is not None

        if has_data and not force_reindex:
            return

        await benchmark_session.execute(delete(DBBenchmarkPaper))
        await benchmark_session.commit()

        doc_ids = list(corpus.keys())
        batch_size = 32

        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i : i + batch_size]
            batch_docs = [corpus[doc_id] for doc_id in batch_ids]

            texts = [f"{doc.get('title', '')} {doc.get('text', '')[:1200]}" for doc in batch_docs]
            embeddings = await self.embedding_service.create_embeddings_batch(texts, batch_size=8, task="search_document")

            rows = []
            for doc_id, doc, embedding in zip(batch_ids, batch_docs, embeddings):
                rows.append(
                    DBBenchmarkPaper(
                        paper_id=str(doc_id),
                        title=(doc.get("title") or "Untitled")[:500],
                        abstract=(doc.get("text") or "")[:4000],
                        embedding=embedding,
                        year=2024,
                        citation_count=0,
                        reference_count=0,
                    )
                )

            benchmark_session.add_all(rows)
            await benchmark_session.commit()

    async def _evaluate_mode(
        self,
        query: str,
        relevant_paper_ids: set[str],
        request: ChunkSearchBenchmarkRequest,
        strategy: str,
    ) -> ChunkSearchModeResult:
        """Evaluate one mode: screened (paper-filtered) or flat chunk search."""
        async with db_session_context() as main_session:
            paper_repo = PaperRepository(main_session)
            chunk_repo = ChunkRepository(main_session)

            query_embedding = await self.embedding_service.create_embedding(query, task="search_query")

            paper_ids_for_chunks: List[str] = []
            screened_latency = 0.0
            retrieved_paper_ids: List[str] = []
            paper_metrics = SearchStageMetrics()

            if strategy == "screened":
                t0 = time.perf_counter()
                papers = await paper_repo.hybrid_search_papers(
                    query=query,
                    query_embedding=query_embedding,
                    limit=request.top_papers,
                    bm25_weight=request.bm25_weight,
                    semantic_weight=request.semantic_weight,
                )
                screened_latency = (time.perf_counter() - t0) * 1000

                retrieved_paper_ids = [str(p.paper_id) for p, _ in papers]
                paper_ids_for_chunks = retrieved_paper_ids
                paper_metrics = self._calc_metrics(retrieved_paper_ids, relevant_paper_ids)

            t1 = time.perf_counter()
            chunks_with_scores = await chunk_repo.hybrid_search_chunks(
                query=query,
                query_embedding=query_embedding,
                limit=max(request.top_chunks * 3, request.top_chunks),
                paper_ids=paper_ids_for_chunks if strategy == "screened" else None,
                bm25_weight=request.bm25_weight,
                semantic_weight=request.semantic_weight,
            )
            chunk_search_latency = (time.perf_counter() - t1) * 1000

            chunk_items = [
                {
                    "chunk_id": str(chunk.chunk_id),
                    "paper_id": str(chunk.paper_id),
                    "score": float(score),
                    "text": chunk.text,
                }
                for chunk, score in chunks_with_scores
            ]

            before_ids = [item["paper_id"] for item in chunk_items[: request.top_chunks]]
            chunk_metrics_before = self._calc_metrics(before_ids, relevant_paper_ids, [item["score"] for item in chunk_items[: request.top_chunks]])

            rerank_latency = None
            mmr_latency = None
            chunk_metrics_after_rerank = None
            chunk_metrics_after_mmr = None

            reranked_items = chunk_items
            if request.rerank and chunk_items:
                from app.domain.chunks.schemas import ChunkRetrieved

                rerank_input = [
                    ChunkRetrieved(
                        id=idx + 1,
                        chunk_id=item["chunk_id"],
                        paper_id=item["paper_id"],
                        text=item["text"],
                        token_count=max(1, len(item["text"].split())),
                        chunk_index=idx,
                        section_title=None,
                        page_number=None,
                        label=None,
                        level=None,
                        char_start=None,
                        char_end=None,
                        docling_metadata=None,
                        embedding=None,
                        created_at=datetime.utcnow(),
                        relevance_score=float(item["score"]),
                    )
                    for idx, item in enumerate(chunk_items)
                ]

                t2 = time.perf_counter()
                reranked = self.ranking_service.rerank_chunks(query, rerank_input)
                rerank_latency = (time.perf_counter() - t2) * 1000

                reranked_items = [
                    {
                        "chunk_id": str(chunk.chunk_id),
                        "paper_id": str(chunk.paper_id),
                        "score": float(chunk.relevance_score),
                        "text": chunk.text,
                    }
                    for chunk in reranked
                ]

                rerank_ids = [item["paper_id"] for item in reranked_items[: request.top_chunks]]
                chunk_metrics_after_rerank = self._calc_metrics(
                    rerank_ids,
                    relevant_paper_ids,
                    [item["score"] for item in reranked_items[: request.top_chunks]],
                )

            if request.mmr and reranked_items:
                t3 = time.perf_counter()
                mmr_items = self._apply_mmr(
                    reranked_items,
                    top_k=request.top_chunks,
                    lambda_param=request.mmr_lambda,
                )
                mmr_latency = (time.perf_counter() - t3) * 1000

                mmr_ids = [item["paper_id"] for item in mmr_items]
                chunk_metrics_after_mmr = self._calc_metrics(
                    mmr_ids,
                    relevant_paper_ids,
                    [item["score"] for item in mmr_items],
                )

            final_items = reranked_items[: request.top_chunks]

            if strategy == "flat":
                retrieved_paper_ids = list(dict.fromkeys([item["paper_id"] for item in final_items]))
                paper_metrics = self._calc_metrics(retrieved_paper_ids, relevant_paper_ids)

            return ChunkSearchModeResult(
                strategy=strategy,
                paper_screening_latency_ms=screened_latency,
                chunk_search_latency_ms=chunk_search_latency,
                rerank_latency_ms=rerank_latency,
                mmr_latency_ms=mmr_latency,
                candidate_count=len(chunk_items),
                retrieved_paper_ids=retrieved_paper_ids,
                retrieved_chunk_ids=[item["chunk_id"] for item in final_items],
                paper_metrics=paper_metrics,
                chunk_metrics_before_rerank=chunk_metrics_before,
                chunk_metrics_after_rerank=chunk_metrics_after_rerank,
                chunk_metrics_after_mmr=chunk_metrics_after_mmr,
            )

    def _calc_metrics(
        self,
        retrieved_ids: List[str],
        relevant_ids: set[str],
        scores: List[float] | None = None,
    ) -> SearchStageMetrics:
        """Compute ranking metrics for one stage."""
        if not retrieved_ids or not relevant_ids:
            return SearchStageMetrics()

        ndcg10 = calculate_ndcg_at_k(retrieved_ids, relevant_ids, k=10)
        ndcg20 = calculate_ndcg_at_k(retrieved_ids, relevant_ids, k=20)
        recall10 = calculate_recall_at_k(retrieved_ids[:10], relevant_ids)
        recall20 = calculate_recall_at_k(retrieved_ids[:20], relevant_ids)
        mrr = calculate_mrr(retrieved_ids, relevant_ids)

        return SearchStageMetrics(
            ndcg_at_10=ndcg10,
            ndcg_at_20=ndcg20,
            recall_at_10=recall10,
            recall_at_20=recall20,
            mrr=mrr,
            average_score=float(np.mean(scores)) if scores else 0.0,
        )

    def _summarize(self, strategy: str, rows: List[ChunkSearchModeResult]) -> ChunkSearchBenchmarkSummary:
        """Aggregate per-query rows into summary."""
        if not rows:
            empty = SearchStageMetrics()
            return ChunkSearchBenchmarkSummary(
                strategy=strategy,
                query_count=0,
                paper_metrics=empty,
                chunk_metrics_before_rerank=empty,
            )

        def avg(values: List[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        paper_metrics = self._avg_metrics([r.paper_metrics for r in rows])
        chunk_before = self._avg_metrics([r.chunk_metrics_before_rerank for r in rows])

        rerank_rows = [r.chunk_metrics_after_rerank for r in rows if r.chunk_metrics_after_rerank is not None]
        mmr_rows = [r.chunk_metrics_after_mmr for r in rows if r.chunk_metrics_after_mmr is not None]

        return ChunkSearchBenchmarkSummary(
            strategy=strategy,
            query_count=len(rows),
            average_paper_screening_latency_ms=avg([r.paper_screening_latency_ms for r in rows]),
            average_chunk_search_latency_ms=avg([r.chunk_search_latency_ms for r in rows]),
            average_rerank_latency_ms=avg([r.rerank_latency_ms for r in rows if r.rerank_latency_ms is not None])
            if any(r.rerank_latency_ms is not None for r in rows)
            else None,
            average_mmr_latency_ms=avg([r.mmr_latency_ms for r in rows if r.mmr_latency_ms is not None])
            if any(r.mmr_latency_ms is not None for r in rows)
            else None,
            paper_metrics=paper_metrics,
            chunk_metrics_before_rerank=chunk_before,
            chunk_metrics_after_rerank=self._avg_metrics([m for m in rerank_rows if m is not None])
            if rerank_rows
            else None,
            chunk_metrics_after_mmr=self._avg_metrics([m for m in mmr_rows if m is not None])
            if mmr_rows
            else None,
        )

    @staticmethod
    def _avg_metrics(metrics: List[SearchStageMetrics]) -> SearchStageMetrics:
        """Average metric objects."""
        if not metrics:
            return SearchStageMetrics()

        return SearchStageMetrics(
            ndcg_at_10=float(np.mean([m.ndcg_at_10 for m in metrics])),
            ndcg_at_20=float(np.mean([m.ndcg_at_20 for m in metrics])),
            recall_at_10=float(np.mean([m.recall_at_10 for m in metrics])),
            recall_at_20=float(np.mean([m.recall_at_20 for m in metrics])),
            mrr=float(np.mean([m.mrr for m in metrics])),
            average_score=float(np.mean([m.average_score for m in metrics])),
        )

    @staticmethod
    def _apply_mmr(
        items: List[Dict[str, Any]],
        top_k: int,
        lambda_param: float,
    ) -> List[Dict[str, Any]]:
        """Lightweight MMR on retrieved items using paper diversity as novelty proxy."""
        if not items:
            return []

        selected: List[Dict[str, Any]] = []
        remaining = items.copy()
        seen_papers: set[str] = set()

        while remaining and len(selected) < top_k:
            best_idx = 0
            best_score = -1e9
            for idx, item in enumerate(remaining):
                relevance = float(item.get("score", 0.0))
                novelty = 0.0 if item.get("paper_id") in seen_papers else 1.0
                mmr_score = (lambda_param * relevance) + ((1 - lambda_param) * novelty)
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            seen_papers.add(str(chosen.get("paper_id")))

        return selected

    

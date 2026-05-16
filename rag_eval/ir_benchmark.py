"""
End-to-End Pipeline Benchmark for RAG System

This module provides comprehensive evaluation of the COMPLETE RAG pipeline,
testing all components: query decomposition, multi-source retrieval, RRF fusion,
ranking, chunk search, and reranking.

Tests what REALLY matters:
- Does query decomposition improve results? (vs single query)
- Does RRF fusion help? (vs single ranking)
- Does the reranker add value? (vs base ranking)
- Does multi-source retrieval (S2 + OA) help? (vs database only)
- How does the full pipeline compare to baselines?

Evaluation Approach:
1. Use BEIR datasets with realistic academic queries
2. Run full pipeline vs ablations (removed components)
3. Measure end-to-end performance (query → final answer)
4. Report component contributions

Key Metrics:
- Paper Retrieval: NDCG@10, Recall@20, MRR
- Chunk Retrieval: NDCG@10, Recall@10
- Component Impact: Delta scores when removing each component
- Latency: Time per pipeline stage
"""

import asyncio
import json
import time
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, cast
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy import ColumnElement, Index, desc, select, delete, text, func
from sqlalchemy.schema import CreateIndex
from sqlalchemy.dialects import postgresql

from app.core.db.database import db_session_context
from app.models import *
from app.models.benchmark_corpus import DBBenchmarkPaper
from app.domain.chunks.types import ChunkRetrieved
from app.llm import get_llm_service
from app.core.singletons import get_ranking_service
from app.processor.services.embeddings import EmbeddingService
from app.utils.benchmark_utils import (
    calculate_f1,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    mask_db_url,
)
from app.extensions.logger import create_logger
from app.search.query_builder import build_paradedb_query

logger = create_logger(__name__)


PAPER_RETRIEVAL_LIMIT_PER_RANKING = 100
PAPER_LIMIT_AFTER_RRF = 100
CHUNK_RETRIEVAL_LIMIT = 100
CHUNK_LIMIT_AFTER_RERANK = 20
BM25_RRF_WEIGHT = 0.15
SEMANTIC_RRF_WEIGHT = 0.85

DECOMPOSED_QUERY_CANDIDATE_FILES = [
    "queries_decomposed.json",
    "queries_decomposed.jsonl",
    "decomposed_queries.json",
    "decomposed_queries.jsonl",
]


class PipelineVariant(Enum):
    """Pipeline configuration variants for ablation study."""

    DECOMPOSED_BEFORE_RERANK = "decomposed_before_rerank"
    DECOMPOSED_AFTER_RERANK = "decomposed_after_rerank"
    ORIGINAL_BEFORE_RERANK = "original_before_rerank"
    ORIGINAL_AFTER_RERANK = "original_after_rerank"


class BM25QuerySource(Enum):
    """Which decomposition query source to use for BM25 retrieval."""

    BOOLEAN = "boolean"
    SEMANTIC = "semantic"
    BOTH = "both"


@dataclass
class ComponentMetrics:
    """Metrics for individual pipeline components."""

    query_decomposition_time: float = 0.0
    retrieval_time: float = 0.0
    rrf_time: float = 0.0
    paper_ranking_time: float = 0.0
    chunk_retrieval_time: float = 0.0
    chunk_reranking_time: float = 0.0
    total_time: float = 0.0

    num_subqueries: int = 0
    num_papers_retrieved: int = 0
    num_papers_after_rrf: int = 0
    num_chunks_retrieved: int = 0
    num_chunks_after_reranking: int = 0


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics."""

    ndcg_at_10: float = 0.0
    ndcg_at_20: float = 0.0
    precision_at_10: float = 0.0
    precision_at_20: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    f1_at_10: float = 0.0
    f1_at_20: float = 0.0
    mrr: float = 0.0

    chunk_ndcg_at_5: float = 0.0
    chunk_ndcg_at_10: float = 0.0
    chunk_recall_at_5: float = 0.0
    chunk_recall_at_10: float = 0.0

    latency_ms: float = 0.0


@dataclass
class PipelineEvaluationResult:
    """Complete evaluation result for a pipeline variant."""

    variant: str
    dataset: str
    query_count: int

    avg_metrics: RetrievalMetrics
    component_metrics: ComponentMetrics

    query_results: List[Dict[str, Any]]

    timestamp: str
    config: Dict[str, Any]


class ExegentPipelineBenchmark:
    """
    End-to-end benchmark for RAG pipeline.

    Evaluates full pipeline with ablations to measure component contributions.

    Database Configuration:
        Use separate test database for complete isolation from production.
        Set BEIR_TEST_DATABASE_URL environment variable:

        export BEIR_TEST_DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5433/beir_test"

        If not set, uses main database with separate beir_test_papers table.
    """

    def __init__(
        self,
        db_session: Optional[AsyncSession] = None,
        dataset_name: str = "scifact",
        output_dir: Optional[str] = None,
        test_db_url: Optional[str] = None,
        bm25_query_source: str = BM25QuerySource.BOTH.value,
    ):
        """Initialize benchmark.

        Args:
            db_session: Main database session (optional if test_db_url provided)
            dataset_name: BEIR dataset name
            output_dir: Output directory for results
            test_db_url: Test database URL (overrides BEIR_TEST_DATABASE_URL env var)
        """
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir or f"rag_eval/evals/experiments/{dataset_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.bm25_query_source = BM25QuerySource(bm25_query_source)
        except ValueError as exc:
            raise ValueError(
                f"Invalid bm25_query_source: {bm25_query_source}. "
                f"Expected one of {[mode.value for mode in BM25QuerySource]}"
            ) from exc

        self.test_db_url = test_db_url or os.getenv("BEIR_TEST_DATABASE_URL")
        self.use_separate_db = self.test_db_url is not None

        self.test_engine: Optional[AsyncEngine] = None
        self.test_session_maker = None
        self._db_session = db_session

        if self.use_separate_db:
            logger.info(
                f"Using separate test database: {mask_db_url(self.test_db_url or '')}"
            )
        else:
            logger.info("Using separate beir_test_papers table in main database")
            if not db_session:
                raise ValueError(
                    "db_session required when not using separate test database"
                )

        data_dir = Path("data/beir_datasets")
        data_path = data_dir / dataset_name

        if not data_path.exists():
            logger.info(
                f"BEIR dataset '{dataset_name}' not found locally, downloading..."
            )
            data_path = util.download_and_unzip(
                f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip",
                out_dir="data/beir_datasets",
            )

        logger.info(f"Loading BEIR dataset: {dataset_name}")
        self.corpus, self.queries, self.qrels = GenericDataLoader(
            data_folder=str(data_path)
        ).load(split="test")

        self.decomposed_queries = self._load_decomposed_queries(Path(data_path))

        logger.info(f"Loaded {len(self.corpus)} documents, {len(self.queries)} queries")
        logger.info(f"BM25 query source mode: {self.bm25_query_source.value}")
        if self.decomposed_queries:
            logger.info(f"Loaded {len(self.decomposed_queries)} pre-decomposed queries")
        else:
            logger.info(
                "No pre-decomposed query file found; falling back to live decomposition"
            )

        self.llm_service = get_llm_service()
        self.ranking_service = get_ranking_service()
        self.embedding_service = EmbeddingService()

    @property
    def db_session(self) -> AsyncSession:
        """Get current database session."""
        if not self._db_session:
            raise RuntimeError("Database session not initialized. Call setup() first.")
        return self._db_session

    async def setup(self):
        """Setup database connections and initialize pipelines."""
        if self.use_separate_db:
            if not self.test_db_url:
                raise ValueError("test_db_url is required when use_separate_db=True")
            self.test_engine = create_async_engine(
                self.test_db_url,
                echo=False,
                pool_pre_ping=True,
            )

            from sqlalchemy.ext.asyncio import async_sessionmaker

            self.test_session_maker = async_sessionmaker(
                self.test_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            self._db_session = self.test_session_maker()
            async with self.test_engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.run_sync(DBBenchmarkPaper.metadata.create_all)

            logger.info("Test database initialized")

    async def cleanup(self):
        """Cleanup database connections."""
        if self.use_separate_db and self._db_session:
            await self._db_session.close()
            if self.test_engine:
                await self.test_engine.dispose()
            logger.info("Test database connections closed")

    async def run_full_evaluation(
        self,
        variants: Optional[List[PipelineVariant]] = None,
        max_queries: Optional[int] = None,
    ) -> Dict[str, PipelineEvaluationResult]:
        """
        Run full evaluation across all pipeline variants.

        Args:
            variants: List of variants to test (default: all)
            max_queries: Limit number of queries for faster testing

        Returns:
            Dict mapping variant name to evaluation results
        """
        if variants is None:
            variants = list(PipelineVariant)

        await self.setup()
        try:
            await self._ensure_corpus_indexed()
            query_ids = list(self.queries.keys())
            if max_queries:
                query_ids = query_ids[:max_queries]

            logger.info(
                f"Evaluating {len(variants)} variants on {len(query_ids)} queries"
            )

            results: Dict[str, PipelineEvaluationResult] = {}
            pending_variants = set(variants)

            paired_modes = [
                (
                    PipelineVariant.DECOMPOSED_BEFORE_RERANK,
                    PipelineVariant.DECOMPOSED_AFTER_RERANK,
                    True,
                ),
                (
                    PipelineVariant.ORIGINAL_BEFORE_RERANK,
                    PipelineVariant.ORIGINAL_AFTER_RERANK,
                    False,
                ),
            ]

            for before_variant, after_variant, is_decomposed in paired_modes:
                if (
                    before_variant in pending_variants
                    and after_variant in pending_variants
                ):
                    logger.info(f"\n{'='*60}")
                    logger.info(
                        f"Evaluating pair in one pass: {before_variant.value} + {after_variant.value}"
                    )
                    logger.info(f"{'='*60}\n")

                    pair_results = await self._evaluate_variant_pair(
                        query_ids=query_ids,
                        is_decomposed=is_decomposed,
                    )
                    results.update(pair_results)
                    pending_variants.remove(before_variant)
                    pending_variants.remove(after_variant)

            for variant in variants:
                if variant not in pending_variants:
                    continue

                logger.info(f"\n{'='*60}")
                logger.info(f"Evaluating: {variant.value}")
                logger.info(f"{'='*60}\n")

                result = await self._evaluate_variant(variant, query_ids)
                results[variant.value] = result

                self._print_variant_summary(result)

            self._print_comparison(results)
            self._save_results(results)
            return results
        finally:
            await self.cleanup()

    async def _evaluate_variant_pair(
        self,
        query_ids: List[str],
        is_decomposed: bool,
    ) -> Dict[str, PipelineEvaluationResult]:
        """Evaluate before/after rerank variants with one retrieval pass per query."""
        before_variant = (
            PipelineVariant.DECOMPOSED_BEFORE_RERANK
            if is_decomposed
            else PipelineVariant.ORIGINAL_BEFORE_RERANK
        )
        after_variant = (
            PipelineVariant.DECOMPOSED_AFTER_RERANK
            if is_decomposed
            else PipelineVariant.ORIGINAL_AFTER_RERANK
        )

        before_query_results: List[Dict[str, Any]] = []
        before_component_metrics: List[ComponentMetrics] = []
        before_retrieval_metrics: List[RetrievalMetrics] = []

        after_query_results: List[Dict[str, Any]] = []
        after_component_metrics: List[ComponentMetrics] = []
        after_retrieval_metrics: List[RetrievalMetrics] = []

        for query_id in query_ids:
            query_text = self.queries[query_id]
            relevance_judgments = self.qrels.get(query_id, {})

            logger.info(f"Query {query_id}: {query_text[:80]}...")

            start_time = time.time()
            (
                retrieved_papers,
                retrieved_chunks_before,
                retrieved_chunks_after,
                metrics_before,
                metrics_after,
            ) = await self._run_variant_pair(
                query=query_text,
                query_id=query_id,
                is_decomposed=is_decomposed,
            )
            latency = (time.time() - start_time) * 1000

            retrieval_before = self._calculate_metrics(
                retrieved_papers,
                retrieved_chunks_before,
                relevance_judgments,
                latency,
            )
            before_query_results.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "metrics": asdict(retrieval_before),
                    "component_metrics": asdict(metrics_before),
                    "num_papers": len(retrieved_papers),
                    "num_chunks": len(retrieved_chunks_before),
                }
            )
            before_component_metrics.append(metrics_before)
            before_retrieval_metrics.append(retrieval_before)

            retrieval_after = self._calculate_metrics(
                retrieved_papers,
                retrieved_chunks_after,
                relevance_judgments,
                latency,
            )
            after_query_results.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "metrics": asdict(retrieval_after),
                    "component_metrics": asdict(metrics_after),
                    "num_papers": len(retrieved_papers),
                    "num_chunks": len(retrieved_chunks_after),
                }
            )
            after_component_metrics.append(metrics_after)
            after_retrieval_metrics.append(retrieval_after)

        before_result = PipelineEvaluationResult(
            variant=before_variant.value,
            dataset=self.dataset_name,
            query_count=len(query_ids),
            avg_metrics=self._average_retrieval_metrics(before_retrieval_metrics),
            component_metrics=self._average_component_metrics(before_component_metrics),
            query_results=before_query_results,
            timestamp=datetime.now().isoformat(),
            config=self._get_variant_config(before_variant),
        )

        after_result = PipelineEvaluationResult(
            variant=after_variant.value,
            dataset=self.dataset_name,
            query_count=len(query_ids),
            avg_metrics=self._average_retrieval_metrics(after_retrieval_metrics),
            component_metrics=self._average_component_metrics(after_component_metrics),
            query_results=after_query_results,
            timestamp=datetime.now().isoformat(),
            config=self._get_variant_config(after_variant),
        )

        self._print_variant_summary(before_result)
        self._print_variant_summary(after_result)

        return {
            before_variant.value: before_result,
            after_variant.value: after_result,
        }

    async def _evaluate_variant(
        self,
        variant: PipelineVariant,
        query_ids: List[str],
    ) -> PipelineEvaluationResult:
        """Evaluate a single pipeline variant."""
        query_results = []
        all_component_metrics = []
        all_retrieval_metrics = []

        for query_id in query_ids:
            query_text = self.queries[query_id]
            relevance_judgments = self.qrels.get(query_id, {})

            logger.info(f"Query {query_id}: {query_text[:80]}...")

            start_time = time.time()
            retrieved_papers, retrieved_chunks, component_metrics = (
                await self._run_variant(variant, query_text, query_id)
            )
            latency = (time.time() - start_time) * 1000

            retrieval_metrics = self._calculate_metrics(
                retrieved_papers,
                retrieved_chunks,
                relevance_judgments,
                latency,
            )
            query_results.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "metrics": asdict(retrieval_metrics),
                    "component_metrics": asdict(component_metrics),
                    "num_papers": len(retrieved_papers),
                    "num_chunks": len(retrieved_chunks),
                }
            )

            all_component_metrics.append(component_metrics)
            all_retrieval_metrics.append(retrieval_metrics)

        avg_retrieval = self._average_retrieval_metrics(all_retrieval_metrics)
        avg_component = self._average_component_metrics(all_component_metrics)

        return PipelineEvaluationResult(
            variant=variant.value,
            dataset=self.dataset_name,
            query_count=len(query_ids),
            avg_metrics=avg_retrieval,
            component_metrics=avg_component,
            query_results=query_results,
            timestamp=datetime.now().isoformat(),
            config=self._get_variant_config(variant),
        )

    async def _run_variant(
        self,
        variant: PipelineVariant,
        query: str,
        query_id: str,
    ) -> Tuple[List[Dict], List[Dict], ComponentMetrics]:
        """
        Run specific pipeline variant and collect results.

        Returns:
            (papers, chunks, component_metrics)
        """
        metrics = ComponentMetrics()
        papers = []
        chunks = []

        config = {
            PipelineVariant.DECOMPOSED_BEFORE_RERANK: {
                "decompose": True,
                "rerank": False,
            },
            PipelineVariant.DECOMPOSED_AFTER_RERANK: {
                "decompose": True,
                "rerank": True,
            },
            PipelineVariant.ORIGINAL_BEFORE_RERANK: {
                "decompose": False,
                "rerank": False,
            },
            PipelineVariant.ORIGINAL_AFTER_RERANK: {
                "decompose": False,
                "rerank": True,
            },
        }[variant]

        hybrid_queries, intent, decomposition_time_ms = (
            await self._resolve_query_decomposition(
                query=query,
                query_id=query_id,
                should_decompose=config["decompose"],
            )
        )
        metrics.query_decomposition_time = decomposition_time_ms
        search_queries = hybrid_queries or [query]
        metrics.num_subqueries = len(search_queries)

        retrieval_start = time.time()
        papers, rrf_time_ms = await self._search_benchmark_papers(
            bm25_query=None,
            semantic_queries=search_queries,
            per_ranking_limit=PAPER_RETRIEVAL_LIMIT_PER_RANKING,
            rrf_limit=PAPER_LIMIT_AFTER_RRF,
        )
        metrics.retrieval_time = (time.time() - retrieval_start) * 1000
        metrics.rrf_time = rrf_time_ms
        metrics.num_papers_retrieved = len(papers)
        metrics.num_papers_after_rrf = len(papers)

        chunk_start = time.time()
        chunks = self._build_virtual_chunks(papers, top_k=CHUNK_RETRIEVAL_LIMIT)
        metrics.chunk_retrieval_time = (time.time() - chunk_start) * 1000
        metrics.num_chunks_retrieved = len(chunks)

        if config["rerank"] and chunks:
            rerank_start = time.time()
            chunks = self._rerank_virtual_chunks(query=query, chunks=chunks)
            chunks = chunks[:CHUNK_LIMIT_AFTER_RERANK]
            metrics.chunk_reranking_time = (time.time() - rerank_start) * 1000
            metrics.num_chunks_after_reranking = len(chunks)
        else:
            chunks = chunks[:CHUNK_LIMIT_AFTER_RERANK]
            metrics.num_chunks_after_reranking = len(chunks)

        metrics.total_time = (
            metrics.query_decomposition_time
            + metrics.retrieval_time
            + metrics.rrf_time
            + metrics.paper_ranking_time
            + metrics.chunk_retrieval_time
            + metrics.chunk_reranking_time
        )

        return papers, chunks, metrics

    async def _run_variant_pair(
        self,
        query: str,
        query_id: str,
        is_decomposed: bool,
    ) -> Tuple[List[Dict], List[Dict], List[Dict], ComponentMetrics, ComponentMetrics]:
        """Run one retrieval pass and produce both before/after-rerank outputs."""
        (
            hybrid_queries,
            intent,
            decomposition_time_ms,
        ) = await self._resolve_query_decomposition(
            query=query,
            query_id=query_id,
            should_decompose=is_decomposed,
        )
        search_queries = hybrid_queries or [query]

        retrieval_start = time.time()
        papers, rrf_time_ms = await self._search_benchmark_papers(
            bm25_query=None,
            semantic_queries=search_queries,
            per_ranking_limit=PAPER_RETRIEVAL_LIMIT_PER_RANKING,
            rrf_limit=PAPER_LIMIT_AFTER_RRF,
        )
        retrieval_time_ms = (time.time() - retrieval_start) * 1000

        chunk_start = time.time()
        chunks_base = self._build_virtual_chunks(papers, top_k=CHUNK_RETRIEVAL_LIMIT)
        chunk_retrieval_time_ms = (time.time() - chunk_start) * 1000

        chunks_before = chunks_base[:CHUNK_LIMIT_AFTER_RERANK]

        rerank_start = time.time()
        chunks_after = self._rerank_virtual_chunks(query=query, chunks=chunks_base)
        chunks_after = chunks_after[:CHUNK_LIMIT_AFTER_RERANK]
        rerank_time_ms = (time.time() - rerank_start) * 1000

        common_fields = {
            "query_decomposition_time": decomposition_time_ms,
            "retrieval_time": retrieval_time_ms,
            "rrf_time": rrf_time_ms,
            "paper_ranking_time": 0.0,
            "chunk_retrieval_time": chunk_retrieval_time_ms,
            "num_subqueries": len(search_queries),
            "num_papers_retrieved": len(papers),
            "num_papers_after_rrf": len(papers),
            "num_chunks_retrieved": len(chunks_base),
        }

        metrics_before = ComponentMetrics(
            **common_fields,
            chunk_reranking_time=0.0,
            num_chunks_after_reranking=len(chunks_before),
            total_time=(
                decomposition_time_ms
                + retrieval_time_ms
                + rrf_time_ms
                + chunk_retrieval_time_ms
            ),
        )

        metrics_after = ComponentMetrics(
            **common_fields,
            chunk_reranking_time=rerank_time_ms,
            num_chunks_after_reranking=len(chunks_after),
            total_time=(
                decomposition_time_ms
                + retrieval_time_ms
                + rrf_time_ms
                + chunk_retrieval_time_ms
                + rerank_time_ms
            ),
        )

        return papers, chunks_before, chunks_after, metrics_before, metrics_after

    async def _resolve_query_decomposition(
        self,
        query: str,
        query_id: str,
        should_decompose: bool,
    ) -> Tuple[Optional[List[str]], Optional[Any], float]:
        """Resolve decomposition payload or fall back to original query."""
        hybrid_queries = None
        intent = None

        if not should_decompose:
            return hybrid_queries, intent, .00

        decomposition_start = time.time()
        try:
            breakdown = self._get_predecomposed_breakdown(query_id=query_id)
            if not breakdown:
                logger.warning(
                    f"No predecomposed breakdown found for query ID: {query_id}"
                )
                return hybrid_queries, intent, 0.0

            intent = breakdown.intent
            hybrid_queries = breakdown.search_queries
            decomposition_time_ms = (time.time() - decomposition_start) * 1000
            return hybrid_queries, intent, decomposition_time_ms
        except Exception as exc:
            logger.warning(
                f"Query decomposition failed, fallback to original query: {exc}"
            )
            return hybrid_queries, intent, 0.0

    async def _search_benchmark_papers(
        self,
        bm25_query: Optional[str],
        semantic_queries: Optional[List[str]],
        per_ranking_limit: int,
        rrf_limit: int,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Search benchmark corpus with configurable BM25 query source + semantic retrieval."""

        def _dedupe_keep_order(items: List[str]) -> List[str]:
            seen = set()
            out: List[str] = []
            for item in items:
                if item in seen:
                    continue
                seen.add(item)
                out.append(item)
            return out

        semantic_query_list = [q for q in (semantic_queries or []) if q]

        bm25_queries: List[str] = []
        if self.bm25_query_source in (BM25QuerySource.BOOLEAN, BM25QuerySource.BOTH):
            if bm25_query:
                bm25_queries.append(bm25_query)
        if self.bm25_query_source in (BM25QuerySource.SEMANTIC, BM25QuerySource.BOTH):
            bm25_queries.extend(semantic_query_list)

        bm25_queries = _dedupe_keep_order(bm25_queries)
        semantic_query_list = _dedupe_keep_order(semantic_query_list)

        if not bm25_queries and not semantic_query_list:
            return [], 0.0

        rankings: List[List[Dict[str, Any]]] = []
        ranking_weights: List[float] = []
        rrf_start = time.time()
        
        for q in semantic_query_list:
            ranking = await self._run_hybrid_search_for_benchmark(
                query=q,
                limit=per_ranking_limit,
            )
            if ranking:
                rankings.append(ranking)
        fused = self._fuse_rankings_with_rrf(
            rankings=rankings,
            ranking_weights=ranking_weights,
            limit=rrf_limit,
        )
        rrf_time_ms = (time.time() - rrf_start) * 1000
        return fused, rrf_time_ms

    async def _run_hybrid_search_for_benchmark(
        self,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        import paradedb

        paradedb_query = query.strip()
        safe_query = build_paradedb_query(paradedb_query, ["title", "abstract"])

        embedding = await self.embedding_service.create_embedding(
            query, task="search_query"
        )
        if not embedding:
            logger.warning("Embedding generation failed for benchmark semantic search")
            return []

        benchmark_id_col = DBBenchmarkPaper.__table__.c.id
        semantic_score = (
            1 - DBBenchmarkPaper.embedding.cosine_distance(embedding)
        ).label("semantic_score")
        bm25_score = cast(ColumnElement[float], paradedb.score(benchmark_id_col)).label(
            "bm25_score"
        )

        combined_score = (bm25_score * 0.7 + semantic_score * 0.3).label(
            "combined_score"
        )
        try:

            result = await self.db_session.execute(
                select(
                    DBBenchmarkPaper.paper_id,
                    DBBenchmarkPaper.title,
                    func.coalesce(DBBenchmarkPaper.abstract, "").label("abstract"),
                    combined_score,
                )
                .where(benchmark_id_col.op("@@@")(safe_query))
                .order_by(desc(combined_score))
                .limit(limit)
            )
            rows = result.all()
            return [
                {
                    "paper_id": row.paper_id,
                    "title": row.title,
                    "abstract": row.abstract,
                    "score": float(row.combined_score),
                }
                for row in rows
            ]
        except Exception as exc:
            logger.warning(
                f"ParadeDB hybrid benchmark query failed, fallback to semantic-only: {exc}"
            )
            return await self._run_semantic_search_for_benchmark(
                query=query, limit=limit
            )

    async def _run_bm25_search_for_benchmark(
        self,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Run BM25-only ranking against benchmark corpus table."""
        import paradedb

        paradedb_query = query.strip()
        safe_query = build_paradedb_query(paradedb_query, ["title", "abstract"])

        try:
            benchmark_id_col = DBBenchmarkPaper.__table__.c.id
            result = await self.db_session.execute(
                select(
                    DBBenchmarkPaper.paper_id,
                    DBBenchmarkPaper.title,
                    func.coalesce(DBBenchmarkPaper.abstract, "").label("abstract"),
                    paradedb.score(benchmark_id_col).label("score"),  # type: ignore[attr-defined]
                )
                .where(benchmark_id_col.op("@@@")(safe_query))
                .order_by(text("score DESC"))
                .limit(limit)
            )

            rows = result.fetchall()
            if rows:
                return [
                    {
                        "paper_id": str(row.paper_id),
                        "title": row.title,
                        "abstract": row.abstract or "",
                        "score": float(row.score),
                    }
                    for row in rows
                ]
        except Exception as exc:
            logger.warning(
                f"ParadeDB BM25 benchmark query failed, fallback to ts_rank_cd: {exc}"
            )

        ts_query = func.websearch_to_tsquery("english", query)
        ts_vector = func.to_tsvector(
            "english",
            func.coalesce(DBBenchmarkPaper.title, "")
            + " "
            + func.coalesce(DBBenchmarkPaper.abstract, ""),
        )
        bm25_score = func.ts_rank_cd(ts_vector, ts_query)

        fallback_result = await self.db_session.execute(
            select(DBBenchmarkPaper, bm25_score.label("score"))
            .order_by(text("score DESC"))
            .limit(limit)
        )

        return [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "abstract": paper.abstract or "",
                "score": float(score),
            }
            for paper, score in fallback_result.all()
        ]

    async def _run_semantic_search_for_benchmark(
        self,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Run semantic-only ranking against benchmark corpus table."""
        embedding = await self.embedding_service.create_embedding(
            query, task="search_query"
        )
        if not embedding:
            logger.warning("Embedding generation failed for benchmark semantic search")
            return []

        semantic_score = 1 - DBBenchmarkPaper.embedding.cosine_distance(embedding)

        result = await self.db_session.execute(
            select(DBBenchmarkPaper, semantic_score.label("score"))
            .where(DBBenchmarkPaper.embedding.isnot(None))
            .order_by(text("score DESC"))
            .limit(limit)
        )

        return [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "abstract": paper.abstract or "",
                "score": float(score),
            }
            for paper, score in result.all()
        ]

    def _fuse_rankings_with_rrf(
        self,
        rankings: List[List[Dict[str, Any]]],
        ranking_weights: Optional[List[float]] = None,
        k: int = 60,
        limit: int = PAPER_LIMIT_AFTER_RRF,
    ) -> List[Dict[str, Any]]:
        """Fuse ranked lists with Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[str, float] = {}
        paper_map: Dict[str, Dict[str, Any]] = {}

        for idx, ranking in enumerate(rankings):
            weight = 1.0
            if ranking_weights and idx < len(ranking_weights):
                weight = ranking_weights[idx]

            for rank, item in enumerate(ranking, start=1):
                pid = item["paper_id"]
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (weight / (k + rank))
                if pid not in paper_map:
                    paper_map[pid] = item

        ranked_ids = sorted(
            rrf_scores.keys(),
            key=lambda pid: rrf_scores[pid],
            reverse=True,
        )[:limit]

        return [
            {
                **paper_map[pid],
                "score": float(rrf_scores[pid]),
            }
            for pid in ranked_ids
        ]

    def _load_decomposed_queries(self, data_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load pre-decomposed query payloads from dataset folder when available."""
        for filename in DECOMPOSED_QUERY_CANDIDATE_FILES:
            candidate = data_path / filename
            if not candidate.exists():
                logger.warning(
                    f"Decomposed query candidate file not found: {candidate}"
                )
                continue

            try:
                if candidate.suffix == ".jsonl":
                    loaded: Dict[str, Dict[str, Any]] = {}
                    with open(candidate, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            row = json.loads(line)
                            qid = str(row.get("_id") or row.get("id") or "").strip()
                            if not qid:
                                continue
                            loaded[qid] = row

                    logger.debug(
                        f"Loaded {len(loaded)} decomposed queries from {candidate}"
                    )
                    return loaded

                with open(candidate, "r", encoding="utf-8") as f:
                    payload = json.load(f)

                if isinstance(payload, dict):
                    return {
                        str(k): v for k, v in payload.items() if isinstance(v, dict)
                    }

                if isinstance(payload, list):
                    loaded = {}
                    for row in payload:
                        if not isinstance(row, dict):
                            continue
                        qid = str(row.get("_id") or row.get("id") or "").strip()
                        if not qid:
                            continue
                        loaded[qid] = row
                    return loaded
            except Exception as exc:
                logger.warning(
                    f"Failed to load decomposed queries from {candidate}: {exc}"
                )

        return {}

    def _get_predecomposed_breakdown(self, query_id: str):
        """Build lightweight breakdown object from pre-decomposed query payload."""
        payload = self.decomposed_queries.get(str(query_id), {})
        if not payload:
            return None

        class _Breakdown:
            def __init__(self, raw: Dict[str, Any]):
                self.clarified_question = raw.get("clarified_question")
                search_queries = raw.get("hybrid_queries") or []
                self.search_queries = [
                    q for q in search_queries if isinstance(q, str) and q.strip()
                ]
                self.intent = raw.get("intent")

        return _Breakdown(payload.get("decomposed", {}))

    def _build_virtual_chunks(
        self, papers: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Convert abstract-only papers into virtual chunks for rerank comparison."""
        chunks: List[Dict[str, Any]] = []
        for paper in papers[:top_k]:
            chunks.append(
                {
                    "chunk_id": f"{paper['paper_id']}_abstract",
                    "paper_id": paper["paper_id"],
                    "text": paper.get("abstract", "") or "",
                    "score": float(paper.get("score", 0.0)),
                }
            )
        return chunks

    def _rerank_virtual_chunks(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply existing reranker on virtual chunks built from abstracts."""
        chunk_models: List[ChunkRetrieved] = []
        for idx, item in enumerate(chunks):
            chunk_models.append(
                ChunkRetrieved(
                    id=idx + 1,
                    chunk_id=item["chunk_id"],
                    paper_id=item["paper_id"],
                    text=item["text"],
                    token_count=max(1, len((item.get("text") or "").split())),
                    chunk_index=idx,
                    section_title="Abstract",
                    page_number=None,
                    label="abstract",
                    level=0,
                    char_start=None,
                    char_end=None,
                    docling_metadata=None,
                    embedding=None,
                    created_at=datetime.now(),
                    relevance_score=float(item.get("score", 0.0)),
                )
            )

        reranked = self.ranking_service.rerank_chunks(query, chunk_models)
        return [
            {
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "text": chunk.text,
                "score": float(chunk.relevance_score),
            }
            for chunk in reranked
        ]

    def _calculate_metrics(
        self,
        papers: List[Dict],
        chunks: List[Dict],
        qrels: Dict[str, int],
        latency: float,
    ) -> RetrievalMetrics:
        """Calculate retrieval metrics for a single query."""
        metrics = RetrievalMetrics(latency_ms=latency)

        if not papers or not qrels:
            return metrics

        retrieved_ids = []
        seen = set()
        source_list = chunks if chunks else papers
        for item in source_list:
            pid = item.get("paper_id", "").replace("beir_", "")
            if pid not in seen:
                seen.add(pid)
                retrieved_ids.append(pid)
            if len(retrieved_ids) == 20:
                break

        relevant_ids = set(qrels.keys())

        metrics.ndcg_at_10 = calculate_ndcg_at_k(retrieved_ids, qrels, k=10)
        metrics.ndcg_at_20 = calculate_ndcg_at_k(retrieved_ids, qrels, k=20)

        metrics.precision_at_10 = calculate_precision_at_k(
            retrieved_ids[:10], relevant_ids
        )
        metrics.precision_at_20 = calculate_precision_at_k(
            retrieved_ids[:20], relevant_ids
        )
        metrics.recall_at_10 = calculate_recall_at_k(retrieved_ids[:10], relevant_ids)
        metrics.recall_at_20 = calculate_recall_at_k(retrieved_ids[:20], relevant_ids)
        metrics.f1_at_10 = calculate_f1(metrics.precision_at_10, metrics.recall_at_10)
        metrics.f1_at_20 = calculate_f1(metrics.precision_at_20, metrics.recall_at_20)

        metrics.mrr = calculate_mrr(retrieved_ids, relevant_ids)

        if chunks:
            metrics.chunk_ndcg_at_5 = 0.0
            metrics.chunk_ndcg_at_10 = 0.0
            metrics.chunk_recall_at_5 = len(chunks[:5]) / max(len(chunks), 1)
            metrics.chunk_recall_at_10 = len(chunks[:10]) / max(len(chunks), 1)

        assert 0 <= metrics.ndcg_at_10 <= 1, f"NDCG@10 out of range: {metrics.ndcg_at_10}"
        assert 0 <= metrics.ndcg_at_20 <= 1, f"NDCG@20 out of range: {metrics.ndcg_at_20}"
        assert 0 <= metrics.precision_at_10 <= 1, f"Precision@10 out of range: {metrics.precision_at_10}"
        assert 0 <= metrics.precision_at_20 <= 1, f"Precision@20 out of range: {metrics.precision_at_20}"
        assert 0 <= metrics.recall_at_10 <= 1, f"Recall@10 out of range: {metrics.recall_at_10}"
        assert 0 <= metrics.recall_at_20 <= 1, f"Recall@20 out of range: {metrics.recall_at_20}"
        assert 0 <= metrics.f1_at_10 <= 1, f"F1@10 out of range: {metrics.f1_at_10}"
        assert 0 <= metrics.f1_at_20 <= 1, f"F1@20 out of range: {metrics.f1_at_20}"
        assert 0 <= metrics.mrr <= 1, f"MRR out of range: {metrics.mrr}"
        assert metrics.latency_ms >= 0, f"Latency negative: {metrics.latency_ms}"

        return metrics

    def _average_retrieval_metrics(
        self, metrics_list: List[RetrievalMetrics]
    ) -> RetrievalMetrics:
        """Average retrieval metrics across queries."""
        if not metrics_list:
            return RetrievalMetrics()

        return RetrievalMetrics(
            ndcg_at_10=float(np.mean([m.ndcg_at_10 for m in metrics_list])),
            ndcg_at_20=float(np.mean([m.ndcg_at_20 for m in metrics_list])),
            precision_at_10=float(np.mean([m.precision_at_10 for m in metrics_list])),
            precision_at_20=float(np.mean([m.precision_at_20 for m in metrics_list])),
            recall_at_10=float(np.mean([m.recall_at_10 for m in metrics_list])),
            recall_at_20=float(np.mean([m.recall_at_20 for m in metrics_list])),
            f1_at_10=float(np.mean([m.f1_at_10 for m in metrics_list])),
            f1_at_20=float(np.mean([m.f1_at_20 for m in metrics_list])),
            mrr=float(np.mean([m.mrr for m in metrics_list])),
            chunk_ndcg_at_5=float(np.mean([m.chunk_ndcg_at_5 for m in metrics_list])),
            chunk_ndcg_at_10=float(np.mean([m.chunk_ndcg_at_10 for m in metrics_list])),
            chunk_recall_at_5=float(
                np.mean([m.chunk_recall_at_5 for m in metrics_list])
            ),
            chunk_recall_at_10=float(
                np.mean([m.chunk_recall_at_10 for m in metrics_list])
            ),
            latency_ms=float(np.mean([m.latency_ms for m in metrics_list])),
        )

    def _average_component_metrics(
        self, metrics_list: List[ComponentMetrics]
    ) -> ComponentMetrics:
        """Average component metrics across queries."""
        if not metrics_list:
            return ComponentMetrics()

        return ComponentMetrics(
            query_decomposition_time=float(
                np.mean([m.query_decomposition_time for m in metrics_list])
            ),
            retrieval_time=float(np.mean([m.retrieval_time for m in metrics_list])),
            rrf_time=float(np.mean([m.rrf_time for m in metrics_list])),
            paper_ranking_time=float(
                np.mean([m.paper_ranking_time for m in metrics_list])
            ),
            chunk_retrieval_time=float(
                np.mean([m.chunk_retrieval_time for m in metrics_list])
            ),
            chunk_reranking_time=float(
                np.mean([m.chunk_reranking_time for m in metrics_list])
            ),
            total_time=float(np.mean([m.total_time for m in metrics_list])),
            num_subqueries=int(np.mean([m.num_subqueries for m in metrics_list])),
            num_papers_retrieved=int(
                np.mean([m.num_papers_retrieved for m in metrics_list])
            ),
            num_papers_after_rrf=int(
                np.mean([m.num_papers_after_rrf for m in metrics_list])
            ),
            num_chunks_retrieved=int(
                np.mean([m.num_chunks_retrieved for m in metrics_list])
            ),
            num_chunks_after_reranking=int(
                np.mean([m.num_chunks_after_reranking for m in metrics_list])
            ),
        )

    async def _ensure_corpus_indexed(self):
        """Ensure BEIR corpus is indexed in separate test table."""
        if self.use_separate_db and self.test_engine is not None:
            async with self.test_engine.begin() as conn:
                await conn.run_sync(DBBenchmarkPaper.metadata.create_all)
        else:
            from app.core.db.database import engine

            async with engine.begin() as conn:
                await conn.run_sync(DBBenchmarkPaper.metadata.create_all)

        result = await self.db_session.execute(select(DBBenchmarkPaper).limit(1))
        existing = result.scalar_one_or_none()

        await self._ensure_benchmark_paradedb_index()

        if existing and len(self.corpus) > 0:
            result = await self.db_session.execute(select(DBBenchmarkPaper))
            count = len(result.scalars().all())
            if count >= len(self.corpus) * 0.9:
                logger.info(f"Corpus already indexed ({count} papers in test table)")
                return

        logger.info(f"Indexing {len(self.corpus)} documents into test table...")

        await self.db_session.execute(delete(DBBenchmarkPaper))
        await self.db_session.commit()

        batch_size = 50
        doc_ids = list(self.corpus.keys())

        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i : i + batch_size]
            batch_docs = [self.corpus[doc_id] for doc_id in batch_ids]

            batch_texts = [
                f"{doc.get('title', '')} {doc.get('text', '')[:500]}"
                for doc in batch_docs
            ]

            try:
                embeddings = await self._get_embeddings_batch(batch_texts)
            except Exception as e:
                logger.warning(f"Error generating embeddings for batch {i}: {e}")
                embeddings = [[0.0] * 768] * len(batch_texts)

            test_papers = []
            for doc_id, doc, embedding in zip(batch_ids, batch_docs, embeddings):
                test_paper = DBBenchmarkPaper(
                    paper_id=doc_id,
                    title=doc.get("title", "Untitled")[:500],
                    abstract=doc.get("text", "")[:2000],
                    embedding=embedding,
                    year=2024,
                    citation_count=0,
                    reference_count=0,
                    created_at=datetime.now(),
                )
                test_papers.append(test_paper)

            self.db_session.add_all(test_papers)
            await self.db_session.commit()

            logger.info(
                f"Indexed batch {i//batch_size + 1}/{(len(doc_ids)-1)//batch_size + 1}"
            )

        logger.info("Corpus indexing complete")

        await self._ensure_benchmark_paradedb_index()

    async def _ensure_benchmark_paradedb_index(self) -> None:
        """Ensure ParadeDB BM25 index exists on benchmark table."""
        try:
            await self.db_session.execute(
                text("CREATE EXTENSION IF NOT EXISTS pg_search")
            )

            idx = Index(
                "idx_beir_test_papers_paradedb_bm25",
                DBBenchmarkPaper.id,
                DBBenchmarkPaper.title,
                DBBenchmarkPaper.abstract,
                postgresql_using="bm25",
                postgresql_with={"key_field": "id"},
            )

            ddl = str(CreateIndex(idx).compile(dialect=postgresql.dialect()))
            ddl_if_not_exists = ddl.replace(
                "CREATE INDEX ",
                "CREATE INDEX IF NOT EXISTS ",
                1,
            )

            await self.db_session.execute(text(ddl_if_not_exists))
            await self.db_session.commit()
        except Exception as exc:
            logger.warning(f"Unable to ensure benchmark ParadeDB index: {exc}")
            await self.db_session.rollback()

    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        embeddings = []
        for text in texts:
            try:
                embedding = await self.embedding_service.create_embedding(
                    text, task="search_document"
                )
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Error generating embedding: {e}")
                embeddings.append([0.0] * 768)
        return embeddings

    def _get_variant_config(self, variant: PipelineVariant) -> Dict[str, Any]:
        """Get configuration for variant."""
        is_decomposed = "decomposed" in variant.value
        is_after_rerank = "after_rerank" in variant.value
        return {
            "variant": variant.value,
            "dataset": self.dataset_name,
            "decomposition": is_decomposed,
            "rerank": is_after_rerank,
            "bm25_query_source": self.bm25_query_source.value,
        }

    def _print_variant_summary(self, result: PipelineEvaluationResult):
        """Print summary for a variant with validation."""
        print(f"\nVariant: {result.variant}")
        print(f"Dataset: {result.dataset} ({result.query_count} queries)")
        print("\nPaper Retrieval:")
        print(f"  NDCG@10:     {result.avg_metrics.ndcg_at_10:.6f}")
        print(f"  NDCG@20:     {result.avg_metrics.ndcg_at_20:.6f}")
        print(f"  Precision@10:{result.avg_metrics.precision_at_10:.6f}")
        print(f"  Precision@20:{result.avg_metrics.precision_at_20:.6f}")
        print(f"  Recall@10:   {result.avg_metrics.recall_at_10:.6f}")
        print(f"  Recall@20:   {result.avg_metrics.recall_at_20:.6f}")
        print(f"  F1@10:       {result.avg_metrics.f1_at_10:.6f}")
        print(f"  F1@20:       {result.avg_metrics.f1_at_20:.6f}")
        print(f"  MRR:         {result.avg_metrics.mrr:.6f}")
        print(f"\nLatency:       {result.avg_metrics.latency_ms:.2f} ms")
        print(f"Chunk Metrics: NDCG@5={result.avg_metrics.chunk_ndcg_at_5:.6f}, Recall@10={result.avg_metrics.chunk_recall_at_10:.6f}")
        print(f"Component Breakdown:")
        print(f"  Decomposition: {result.component_metrics.query_decomposition_time:.2f}ms")
        print(f"  Retrieval:     {result.component_metrics.retrieval_time:.2f}ms")
        print(f"  RRF:           {result.component_metrics.rrf_time:.2f}ms")
        print(f"  Chunk Rerank:  {result.component_metrics.chunk_reranking_time:.2f}ms")
        print(f"  Total:         {result.component_metrics.total_time:.2f}ms")

    def _print_comparison(self, results: Dict[str, PipelineEvaluationResult]):
        """Print comparison table."""
        print(f"\n{'='*100}")
        print("PIPELINE COMPARISON")
        print(f"{'='*100}\n")

        header = f"{'Variant':<20} {'NDCG@10':<10} {'Recall@20':<12} {'F1@20':<10} {'MRR':<10} {'Latency(ms)':<12} {'Papers':<8}"
        print(header)
        print("-" * 100)

        baseline_ndcg = None
        if "baseline" in results:
            baseline_ndcg = results["baseline"].avg_metrics.ndcg_at_10

        for variant_name, result in results.items():
            m = result.avg_metrics
            c = result.component_metrics

            ndcg_str = f"{m.ndcg_at_10:.4f}"
            if baseline_ndcg and variant_name != "baseline":
                delta = m.ndcg_at_10 - baseline_ndcg
                ndcg_str += f" ({delta:+.4f})"

            row = (
                f"{variant_name:<20} {ndcg_str:<10} {m.recall_at_20:<12.4f} "
                f"{m.f1_at_20:<10.4f} {m.mrr:<10.4f} {m.latency_ms:<12.0f} {c.num_papers_retrieved:<8}"
            )
            print(row)

        print(f"\n{'='*100}\n")

    def _save_results(self, results: Dict[str, PipelineEvaluationResult]):
        """Save results to CSV files in a results folder."""
        import csv
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_folder = self.output_dir / f"ir_benchmark_{self.dataset_name}_{timestamp}"
        results_folder.mkdir(parents=True, exist_ok=True)

        raw_file = results_folder / "results.json"
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(
                {variant_name: asdict(result) for variant_name, result in results.items()},
                f,
                indent=2,
            )
        
        summary_file = results_folder / "summary.csv"
        with open(summary_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Variant",
                "Dataset",
                "Query Count",
                "NDCG@10",
                "NDCG@20",
                "Precision@10",
                "Precision@20",
                "Recall@10",
                "Recall@20",
                "F1@10",
                "F1@20",
                "MRR",
                "Chunk NDCG@5",
                "Chunk NDCG@10",
                "Chunk Recall@5",
                "Chunk Recall@10",
                "Avg Latency (ms)",
                "Avg Papers Retrieved",
                "Avg Papers After RRF",
                "Avg Chunks Retrieved",
                "Avg Chunks After Rerank",
                "Avg Subqueries",
            ])
            for variant_name, result in results.items():
                writer.writerow([
                    variant_name,
                    result.dataset,
                    result.query_count,
                    f"{result.avg_metrics.ndcg_at_10:.6f}",
                    f"{result.avg_metrics.ndcg_at_20:.6f}",
                    f"{result.avg_metrics.precision_at_10:.6f}",
                    f"{result.avg_metrics.precision_at_20:.6f}",
                    f"{result.avg_metrics.recall_at_10:.6f}",
                    f"{result.avg_metrics.recall_at_20:.6f}",
                    f"{result.avg_metrics.f1_at_10:.6f}",
                    f"{result.avg_metrics.f1_at_20:.6f}",
                    f"{result.avg_metrics.mrr:.6f}",
                    f"{result.avg_metrics.chunk_ndcg_at_5:.6f}",
                    f"{result.avg_metrics.chunk_ndcg_at_10:.6f}",
                    f"{result.avg_metrics.chunk_recall_at_5:.6f}",
                    f"{result.avg_metrics.chunk_recall_at_10:.6f}",
                    f"{result.avg_metrics.latency_ms:.2f}",
                    f"{result.component_metrics.num_papers_retrieved:.0f}",
                    f"{result.component_metrics.num_papers_after_rrf:.0f}",
                    f"{result.component_metrics.num_chunks_retrieved:.0f}",
                    f"{result.component_metrics.num_chunks_after_reranking:.0f}",
                    f"{result.component_metrics.num_subqueries:.0f}",
                ])
        
        components_file = results_folder / "components.csv"
        with open(components_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Variant",
                "Query Decomposition (ms)",
                "Retrieval (ms)",
                "RRF (ms)",
                "Paper Ranking (ms)",
                "Chunk Retrieval (ms)",
                "Chunk Reranking (ms)",
                "Total Time (ms)",
            ])
            for variant_name, result in results.items():
                m = result.component_metrics
                writer.writerow([
                    variant_name,
                    f"{m.query_decomposition_time:.2f}",
                    f"{m.retrieval_time:.2f}",
                    f"{m.rrf_time:.2f}",
                    f"{m.paper_ranking_time:.2f}",
                    f"{m.chunk_retrieval_time:.2f}",
                    f"{m.chunk_reranking_time:.2f}",
                    f"{m.total_time:.2f}",
                ])
        
        for variant_name, result in results.items():
            query_file = results_folder / f"{variant_name}_queries.csv"
            with open(query_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Query ID",
                    "NDCG@10",
                    "NDCG@20",
                    "Precision@10",
                    "Precision@20",
                    "Recall@10",
                    "Recall@20",
                    "F1@10",
                    "F1@20",
                    "MRR",
                    "Latency (ms)",
                    "Papers Retrieved",
                    "Chunks Retrieved",
                ])
                for query_result in result.query_results:
                    metrics = query_result.get("metrics", {})
                    writer.writerow([
                        query_result.get("query_id", ""),
                        f"{metrics.get('ndcg_at_10', 0):.6f}",
                        f"{metrics.get('ndcg_at_20', 0):.6f}",
                        f"{metrics.get('precision_at_10', 0):.6f}",
                        f"{metrics.get('precision_at_20', 0):.6f}",
                        f"{metrics.get('recall_at_10', 0):.6f}",
                        f"{metrics.get('recall_at_20', 0):.6f}",
                        f"{metrics.get('f1_at_10', 0):.6f}",
                        f"{metrics.get('f1_at_20', 0):.6f}",
                        f"{metrics.get('mrr', 0):.6f}",
                        f"{metrics.get('latency_ms', 0):.2f}",
                        query_result.get("num_papers", 0),
                        query_result.get("num_chunks", 0),
                    ])
        
        comparison_file = results_folder / "comparison.txt"
        with open(comparison_file, "w") as f:
            f.write("=" * 120 + "\n")
            f.write(f"Benchmark Results: {self.dataset_name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 120 + "\n\n")
            
            f.write("AGGREGATE METRICS SUMMARY\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'Variant':<25} {'NDCG@10':<12} {'Recall@20':<12} {'F1@20':<12} {'MRR':<12} {'Latency':<12} {'Queries':<10}\n")
            f.write("-" * 120 + "\n")
            
            baseline_ndcg = None
            first_variant = list(results.values())[0]
            if "original_before_rerank" in results:
                baseline_ndcg = results["original_before_rerank"].avg_metrics.ndcg_at_10
            
            for variant_name in sorted(results.keys()):
                result = results[variant_name]
                m = result.avg_metrics
                
                ndcg_str = f"{m.ndcg_at_10:.4f}"
                if baseline_ndcg and variant_name != "original_before_rerank" and baseline_ndcg > 0:
                    delta = m.ndcg_at_10 - baseline_ndcg
                    pct_change = (delta / baseline_ndcg) * 100 if baseline_ndcg > 0 else 0
                    ndcg_str += f" ({delta:+.4f}, {pct_change:+.1f}%)"
                
                f.write(f"{variant_name:<25} {ndcg_str:<12} {m.recall_at_20:<12.4f} {m.f1_at_20:<12.4f} {m.mrr:<12.4f} {m.latency_ms:<12.0f} {result.query_count:<10}\n")
            
            f.write("\n" + "=" * 120 + "\n\n")
            
            f.write("DETAILED COMPONENT BREAKDOWN\n")
            for variant_name in sorted(results.keys()):
                result = results[variant_name]
                m = result.avg_metrics
                c = result.component_metrics
                
                f.write(f"\n{variant_name}\n")
                f.write("-" * 120 + "\n")
                f.write(f"  Paper Retrieval Metrics:\n")
                f.write(f"    NDCG@10:  {m.ndcg_at_10:.6f}\n")
                f.write(f"    NDCG@20:  {m.ndcg_at_20:.6f}\n")
                f.write(f"    Precision@10: {m.precision_at_10:.6f}\n")
                f.write(f"    Precision@20: {m.precision_at_20:.6f}\n")
                f.write(f"    Recall@10: {m.recall_at_10:.6f}\n")
                f.write(f"    Recall@20: {m.recall_at_20:.6f}\n")
                f.write(f"    F1@10:    {m.f1_at_10:.6f}\n")
                f.write(f"    F1@20:    {m.f1_at_20:.6f}\n")
                f.write(f"    MRR:      {m.mrr:.6f}\n")
                f.write(f"  Chunk Retrieval Metrics:\n")
                f.write(f"    NDCG@5:   {m.chunk_ndcg_at_5:.6f}\n")
                f.write(f"    NDCG@10:  {m.chunk_ndcg_at_10:.6f}\n")
                f.write(f"    Recall@5: {m.chunk_recall_at_5:.6f}\n")
                f.write(f"    Recall@10: {m.chunk_recall_at_10:.6f}\n")
                f.write(f"  Component Timings (ms):\n")
                f.write(f"    Query Decomposition:  {c.query_decomposition_time:>8.2f}\n")
                f.write(f"    Retrieval:            {c.retrieval_time:>8.2f}\n")
                f.write(f"    RRF Fusion:           {c.rrf_time:>8.2f}\n")
                f.write(f"    Paper Ranking:        {c.paper_ranking_time:>8.2f}\n")
                f.write(f"    Chunk Retrieval:      {c.chunk_retrieval_time:>8.2f}\n")
                f.write(f"    Chunk Reranking:      {c.chunk_reranking_time:>8.2f}\n")
                f.write(f"    Total:                {c.total_time:>8.2f}\n")
                f.write(f"  Query Aggregates:\n")
                f.write(f"    Avg Latency:          {m.latency_ms:>8.2f} ms\n")
                f.write(f"    Papers Retrieved:     {c.num_papers_retrieved:>8.0f}\n")
                f.write(f"    Papers After RRF:     {c.num_papers_after_rrf:>8.0f}\n")
                f.write(f"    Chunks Retrieved:     {c.num_chunks_retrieved:>8.0f}\n")
                f.write(f"    Chunks After Rerank:  {c.num_chunks_after_reranking:>8.0f}\n")
                f.write(f"    Avg Subqueries:       {c.num_subqueries:>8.0f}\n")
        
        logger.info(f"Results saved to folder: {results_folder}")
        logger.info(f"  - results.json: Complete raw results for reproducibility")
        logger.info(f"  - summary.csv: Aggregate metrics per variant")
        logger.info(f"  - components.csv: Component timing breakdown")
        logger.info(f"  - *_queries.csv: Per-query detailed results")
        logger.info(f"  - comparison.txt: Human-readable comparison report")


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="End-to-end pipeline benchmark"
    )
    parser.add_argument(
        "--dataset",
        default="scifact",
        choices=["scifact", "nfcorpus", "scidocs", "trec-covid"],
        help="BEIR dataset",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=50,
        help="Max queries to test (for faster evaluation)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[v.value for v in PipelineVariant],
        help="Variants to test (default: all)",
    )
    parser.add_argument(
        "--decomposition",
        choices=["both", "on", "off"],
        default="off",
        help="Filter variants by decomposition mode when --variants is not provided",
    )
    parser.add_argument(
        "--compare-rerank",
        action="store_true",
        help="When set and --variants is not provided, run both before/after rerank variants",
    )
    parser.add_argument(
        "--test-db-url",
        help="Test database URL (overrides BEIR_TEST_DATABASE_URL env var)",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for benchmark exports (default: rag_eval/evals/experiments/<dataset>)",
    )
    parser.add_argument(
        "--bm25-source",
        choices=[mode.value for mode in BM25QuerySource],
        default=BM25QuerySource.BOTH.value,
        help="BM25 query source: boolean decomposition query, semantic subqueries, or both",
    )

    args = parser.parse_args()

    variants = None
    if args.variants:
        variants = [PipelineVariant(v) for v in args.variants]
    else:
        base_variants: List[PipelineVariant] = []
        if args.decomposition in ("both", "on"):
            base_variants.extend(
                [
                    PipelineVariant.DECOMPOSED_BEFORE_RERANK,
                    PipelineVariant.DECOMPOSED_AFTER_RERANK,
                ]
            )
        if args.decomposition in ("both", "off"):
            base_variants.extend(
                [
                    PipelineVariant.ORIGINAL_BEFORE_RERANK,
                    PipelineVariant.ORIGINAL_AFTER_RERANK,
                ]
            )

        if not args.compare_rerank:
            base_variants = [
                variant for variant in base_variants if "after_rerank" in variant.value
            ]

        variants = base_variants

    test_db_url = args.test_db_url or os.getenv("BEIR_TEST_DATABASE_URL")

    if test_db_url:
        benchmark = ExegentPipelineBenchmark(
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            test_db_url=test_db_url,
            bm25_query_source=args.bm25_source,
        )

        await benchmark.run_full_evaluation(
            variants=variants,
            max_queries=args.max_queries,
        )
    else:
        async with db_session_context() as session:
            benchmark = ExegentPipelineBenchmark(
                db_session=session,
                dataset_name=args.dataset,
                output_dir=args.output_dir,
                bm25_query_source=args.bm25_source,
            )

            await benchmark.run_full_evaluation(
                variants=variants,
                max_queries=args.max_queries,
            )


if __name__ == "__main__":
    asyncio.run(main())

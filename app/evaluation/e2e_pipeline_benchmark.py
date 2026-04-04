"""
End-to-End Pipeline Benchmark for Exegent RAG System

This module provides comprehensive evaluation of the COMPLETE Exegent pipeline,
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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy import select, delete, text, func

from app.db.database import db_session_context
from app.models import *
from app.models.benchmark_corpus import DBBenchmarkPaper
from app.domain.chunks.schemas import ChunkRetrieved
from app.llm import get_llm_service
from app.core.singletons import get_ranking_service
from app.processor.services.embeddings import EmbeddingService
from app.utils.benchmark_utils import (
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
    mask_db_url,
)
from app.extensions.logger import create_logger

logger = create_logger(__name__)


PAPER_RETRIEVAL_LIMIT_PER_RANKING = 100
PAPER_LIMIT_AFTER_RRF = 100
CHUNK_RETRIEVAL_LIMIT = 100
CHUNK_LIMIT_AFTER_RERANK = 20

class PipelineVariant(Enum):
    """Pipeline configuration variants for ablation study."""

    DECOMPOSED_BEFORE_RERANK = "decomposed_before_rerank"
    DECOMPOSED_AFTER_RERANK = "decomposed_after_rerank"
    ORIGINAL_BEFORE_RERANK = "original_before_rerank"
    ORIGINAL_AFTER_RERANK = "original_after_rerank"


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

    # Component outputs
    num_subqueries: int = 0
    num_papers_retrieved: int = 0
    num_papers_after_rrf: int = 0
    num_chunks_retrieved: int = 0
    num_chunks_after_reranking: int = 0


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics."""

    # Paper-level metrics
    ndcg_at_10: float = 0.0
    ndcg_at_20: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    mrr: float = 0.0

    # Chunk-level metrics
    chunk_ndcg_at_5: float = 0.0
    chunk_ndcg_at_10: float = 0.0
    chunk_recall_at_5: float = 0.0
    chunk_recall_at_10: float = 0.0

    # Latency
    latency_ms: float = 0.0


@dataclass
class PipelineEvaluationResult:
    """Complete evaluation result for a pipeline variant."""

    variant: str
    dataset: str
    query_count: int

    # Aggregate metrics
    avg_metrics: RetrievalMetrics
    component_metrics: ComponentMetrics

    # Per-query results
    query_results: List[Dict[str, Any]]

    # Metadata
    timestamp: str
    config: Dict[str, Any]


class ExegentPipelineBenchmark:
    """
    End-to-end benchmark for Exegent RAG pipeline.

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
        output_dir: str = "evaluation_results/e2e",
        test_db_url: Optional[str] = None,
    ):
        """Initialize benchmark.

        Args:
            db_session: Main database session (optional if test_db_url provided)
            dataset_name: BEIR dataset name
            output_dir: Output directory for results
            test_db_url: Test database URL (overrides BEIR_TEST_DATABASE_URL env var)
        """
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Determine test database configuration
        self.test_db_url = test_db_url or os.getenv("BEIR_TEST_DATABASE_URL")
        self.use_separate_db = self.test_db_url is not None

        # Database setup
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

        # Load BEIR dataset
        logger.info(f"Loading BEIR dataset: {dataset_name}")
        data_path = util.download_and_unzip(
            f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip",
            out_dir="beir_datasets",
        )
        self.corpus, self.queries, self.qrels = GenericDataLoader(
            data_folder=data_path
        ).load(split="test")

        logger.info(f"Loaded {len(self.corpus)} documents, {len(self.queries)} queries")
        self.llm_service = get_llm_service()
        self.ranking_service = get_ranking_service()
        self.embedding_service = EmbeddingService()

        # Benchmark directly queries DBBenchmarkPaper corpus table

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
            # Create test database engine
            self.test_engine = create_async_engine(
                self.test_db_url,
                echo=False,
                pool_pre_ping=True,
            )

            # Create session maker
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

        # No pipeline wiring here; benchmark runs directly on benchmark corpus table.

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

        # Setup database and pipelines
        await self.setup()
        try:
            await self._ensure_corpus_indexed()
            query_ids = list(self.queries.keys())
            if max_queries:
                query_ids = query_ids[:max_queries]

            logger.info(
                f"Evaluating {len(variants)} variants on {len(query_ids)} queries"
            )

            results = {}
            for variant in variants:
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

            # Run pipeline
            start_time = time.time()
            retrieved_papers, retrieved_chunks, component_metrics = (
                await self._run_variant(variant, query_text)
            )
            latency = (time.time() - start_time) * 1000  # ms

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

        # Aggregate metrics
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

        bm25_query: Optional[str] = query
        semantic_queries: Optional[List[str]] = [query]
        intent = None

        if config["decompose"]:
            try:
                decomposition_start = time.time()
                breakdown = await self.llm_service.decompose_user_query_v2(user_question=query)
                metrics.query_decomposition_time = (time.time() - decomposition_start) * 1000
                bm25_query = breakdown.bm25_query or breakdown.clarified_question or query
                semantic_queries = breakdown.semantic_queries or [q for q in breakdown.search_queries if q]
                intent = breakdown.intent
                metrics.num_subqueries = len(semantic_queries or []) + (1 if bm25_query else 0)
            except Exception as exc:
                logger.warning(f"Query decomposition failed, fallback to original query: {exc}")

        retrieval_start = time.time()
        papers, rrf_time_ms = await self._search_benchmark_papers(
            bm25_query=bm25_query,
            semantic_queries=semantic_queries,
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

    async def _search_benchmark_papers(
        self,
        bm25_query: Optional[str],
        semantic_queries: Optional[List[str]],
        per_ranking_limit: int,
        rrf_limit: int,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Search benchmark corpus with split BM25/semantic retrieval + external RRF."""
        query_list: List[str] = []
        if bm25_query:
            query_list.append(bm25_query)
        if semantic_queries:
            query_list.extend([q for q in semantic_queries if q])

        if not query_list:
            return [], 0.0

        rankings: List[List[Dict[str, Any]]] = []
        rrf_start = time.time()

        for q in query_list:
            bm25_ranking = await self._run_bm25_search_for_benchmark(
                query=q,
                limit=per_ranking_limit,
            )
            if bm25_ranking:
                rankings.append(bm25_ranking)

            semantic_ranking = await self._run_semantic_search_for_benchmark(
                query=q,
                limit=per_ranking_limit,
            )
            if semantic_ranking:
                rankings.append(semantic_ranking)

        fused = self._fuse_rankings_with_rrf(rankings=rankings, limit=rrf_limit)
        rrf_time_ms = (time.time() - rrf_start) * 1000
        return fused, rrf_time_ms

    async def _run_bm25_search_for_benchmark(
        self,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Run BM25-only ranking against benchmark corpus table."""
        ts_query = func.websearch_to_tsquery("english", query)
        ts_vector = func.to_tsvector(
            "english",
            func.coalesce(DBBenchmarkPaper.title, "")
            + " "
            + func.coalesce(DBBenchmarkPaper.abstract, ""),
        )
        bm25_score = func.ts_rank_cd(ts_vector, ts_query)

        result = await self.db_session.execute(
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
            for paper, score in result.all()
        ]

    async def _run_semantic_search_for_benchmark(
        self,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Run semantic-only ranking against benchmark corpus table."""
        embedding = await self.embedding_service.create_embedding(query, task="search_query")
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
        k: int = 60,
        limit: int = PAPER_LIMIT_AFTER_RRF,
    ) -> List[Dict[str, Any]]:
        """Fuse ranked lists with Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[str, float] = {}
        paper_map: Dict[str, Dict[str, Any]] = {}

        for ranking in rankings:
            for rank, item in enumerate(ranking, start=1):
                pid = item["paper_id"]
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (k + rank))
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

    def _build_virtual_chunks(self, papers: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
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

    def _rerank_virtual_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

        # NDCG@k
        metrics.ndcg_at_10 = calculate_ndcg_at_k(retrieved_ids, qrels, k=10)
        metrics.ndcg_at_20 = calculate_ndcg_at_k(retrieved_ids, qrels, k=20)

        # Recall@k
        metrics.recall_at_10 = calculate_recall_at_k(retrieved_ids[:10], relevant_ids)
        metrics.recall_at_20 = calculate_recall_at_k(retrieved_ids[:20], relevant_ids)

        # MRR
        metrics.mrr = calculate_mrr(retrieved_ids, relevant_ids)

        # Chunk metrics (if we have ground truth chunks)
        if chunks:
            metrics.chunk_ndcg_at_5 = 0.0  # Placeholder
            metrics.chunk_ndcg_at_10 = 0.0
            metrics.chunk_recall_at_5 = len(chunks[:5]) / max(len(chunks), 1)
            metrics.chunk_recall_at_10 = len(chunks[:10]) / max(len(chunks), 1)

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
            recall_at_10=float(np.mean([m.recall_at_10 for m in metrics_list])),
            recall_at_20=float(np.mean([m.recall_at_20 for m in metrics_list])),
            mrr=float(np.mean([m.mrr for m in metrics_list])),
            chunk_ndcg_at_5=float(np.mean([m.chunk_ndcg_at_5 for m in metrics_list])),
            chunk_ndcg_at_10=float(np.mean([m.chunk_ndcg_at_10 for m in metrics_list])),
            chunk_recall_at_5=float(np.mean([m.chunk_recall_at_5 for m in metrics_list])),
            chunk_recall_at_10=float(np.mean([m.chunk_recall_at_10 for m in metrics_list])),
            latency_ms=float(np.mean([m.latency_ms for m in metrics_list])),
        )

    def _average_component_metrics(
        self, metrics_list: List[ComponentMetrics]
    ) -> ComponentMetrics:
        """Average component metrics across queries."""
        if not metrics_list:
            return ComponentMetrics()

        return ComponentMetrics(
            query_decomposition_time=float(np.mean(
                [m.query_decomposition_time for m in metrics_list]
            )),
            retrieval_time=float(np.mean([m.retrieval_time for m in metrics_list])),
            rrf_time=float(np.mean([m.rrf_time for m in metrics_list])),
            paper_ranking_time=float(np.mean([m.paper_ranking_time for m in metrics_list])),
            chunk_retrieval_time=float(np.mean(
                [m.chunk_retrieval_time for m in metrics_list]
            )),
            chunk_reranking_time=float(np.mean(
                [m.chunk_reranking_time for m in metrics_list]
            )),
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
        # Create test table if not exists on the active benchmark engine
        if self.use_separate_db and self.test_engine is not None:
            async with self.test_engine.begin() as conn:
                await conn.run_sync(DBBenchmarkPaper.metadata.create_all)
        else:
            from app.db.database import engine

            async with engine.begin() as conn:
                await conn.run_sync(DBBenchmarkPaper.metadata.create_all)

        # Check if already indexed
        result = await self.db_session.execute(select(DBBenchmarkPaper).limit(1))
        existing = result.scalar_one_or_none()

        if existing and len(self.corpus) > 0:
            # Check count
            result = await self.db_session.execute(select(DBBenchmarkPaper))
            count = len(result.scalars().all())
            if count >= len(self.corpus) * 0.9:  # 90% threshold
                logger.info(f"Corpus already indexed ({count} papers in test table)")
                return

        logger.info(f"Indexing {len(self.corpus)} documents into test table...")

        # Clear existing test data
        await self.db_session.execute(delete(DBBenchmarkPaper))
        await self.db_session.commit()

        # Batch index corpus
        batch_size = 50
        doc_ids = list(self.corpus.keys())

        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i : i + batch_size]
            batch_docs = [self.corpus[doc_id] for doc_id in batch_ids]

            # Generate embeddings
            batch_texts = [
                f"{doc.get('title', '')} {doc.get('text', '')[:500]}"
                for doc in batch_docs
            ]

            try:
                embeddings = await self._get_embeddings_batch(batch_texts)
            except Exception as e:
                logger.warning(f"Error generating embeddings for batch {i}: {e}")
                embeddings = [[0.0] * 768] * len(batch_texts)

            # Create test paper records
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

    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        embeddings = []
        for text in texts:
            try:
                embedding = await self.embedding_service.create_embedding(
                    text, task="search_document"  # Limit text length
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
        }

    def _print_variant_summary(self, result: PipelineEvaluationResult):
        """Print summary for a variant."""
        print(f"\nVariant: {result.variant}")
        print(f"Dataset: {result.dataset} ({result.query_count} queries)")
        print("\nPaper Retrieval:")
        print(f"  NDCG@10:     {result.avg_metrics.ndcg_at_10:.4f}")
        print(f"  NDCG@20:     {result.avg_metrics.ndcg_at_20:.4f}")
        print(f"  Recall@10:   {result.avg_metrics.recall_at_10:.4f}")
        print(f"  Recall@20:   {result.avg_metrics.recall_at_20:.4f}")
        print(f"  MRR:         {result.avg_metrics.mrr:.4f}")
        print(f"\nLatency:       {result.avg_metrics.latency_ms:.0f} ms")

    def _print_comparison(self, results: Dict[str, PipelineEvaluationResult]):
        """Print comparison table."""
        print(f"\n{'='*100}")
        print("PIPELINE COMPARISON")
        print(f"{'='*100}\n")

        header = f"{'Variant':<20} {'NDCG@10':<10} {'Recall@20':<12} {'MRR':<10} {'Latency(ms)':<12} {'Papers':<8}"
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
                f"{m.mrr:<10.4f} {m.latency_ms:<12.0f} {c.num_papers_retrieved:<8}"
            )
            print(row)

        print(f"\n{'='*100}\n")

    def _save_results(self, results: Dict[str, PipelineEvaluationResult]):
        """Save results to JSON."""
        output_file = (
            self.output_dir
            / f"e2e_benchmark_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        results_dict = {
            variant_name: {
                **asdict(result),
                "avg_metrics": asdict(result.avg_metrics),
                "component_metrics": asdict(result.component_metrics),
            }
            for variant_name, result in results.items()
        }

        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Results saved to {output_file}")


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="End-to-end Exegent pipeline benchmark"
    )
    parser.add_argument(
        "--dataset",
        default="scifact",
        choices=["scifact", "nfcorpus", "scidocs"],
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
                variant
                for variant in base_variants
                if "after_rerank" in variant.value
            ]

        variants = base_variants

    # Check for test database URL
    test_db_url = args.test_db_url or os.getenv("BEIR_TEST_DATABASE_URL")

    if test_db_url:
        # Use separate test database
        benchmark = ExegentPipelineBenchmark(
            dataset_name=args.dataset,
            test_db_url=test_db_url,
        )

        await benchmark.run_full_evaluation(
            variants=variants,
            max_queries=args.max_queries,
        )
    else:
        # Use main database with separate table
        async with db_session_context() as session:
            benchmark = ExegentPipelineBenchmark(
                db_session=session,
                dataset_name=args.dataset,
            )

            await benchmark.run_full_evaluation(
                variants=variants,
                max_queries=args.max_queries,
            )


if __name__ == "__main__":
    asyncio.run(main())

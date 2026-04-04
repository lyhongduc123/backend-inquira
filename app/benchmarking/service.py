"""
Benchmark Service
Business logic for RAG pipeline benchmarking and evaluation.
"""
from typing import List, Dict, Any, Optional
import time
import statistics
from collections import Counter
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.models.benchmarks import (
    DBPaperRelevanceBenchmark,
    DBChunkQualityBenchmark,
    DBBreakdownQualityBenchmark,
    DBPipelineBenchmark,
    DBGroundTruthDataset,
)
from app.benchmarking.schemas import (
    PaperRelevanceBenchmarkCreate,
    PaperRelevanceBenchmarkResult,
    ChunkQualityBenchmarkCreate,
    ChunkQualityBenchmarkResult,
    BreakdownQualityBenchmarkCreate,
    BreakdownQualityBenchmarkResult,
    PipelineBenchmarkCreate,
    PipelineBenchmarkResult,
    GroundTruthDatasetCreate,
    GroundTruthDataset,
    BenchmarkComparisonReport,
    PipelineComparisonMetrics,
    AggregateMetrics,
    BenchmarkFilters,
)
from app.benchmarking import metrics as metrics_calc
from app.processor.schemas import RankedPaper
from app.domain.chunks.schemas import ChunkRetrieved
from app.llm.schemas import QuestionBreakdownResponse
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class BenchmarkService:
    """Service for benchmarking RAG pipeline components"""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    # ========================================================================
    # Ground Truth Management
    # ========================================================================

    async def create_ground_truth(
        self, data: GroundTruthDatasetCreate
    ) -> GroundTruthDataset:
        """Create a ground truth dataset entry."""
        db_entry = DBGroundTruthDataset(
            query=data.query,
            relevant_paper_ids=data.relevant_paper_ids,
            query_intent=data.query_intent,
            difficulty=data.difficulty,
            domain=data.domain,
            notes=data.notes,
            created_by=data.created_by,
        )
        
        self.db.add(db_entry)
        await self.db.commit()
        await self.db.refresh(db_entry)
        
        return GroundTruthDataset.model_validate(db_entry)

    async def get_ground_truth_for_query(
        self, query: str
    ) -> Optional[GroundTruthDataset]:
        """Get ground truth for a specific query (exact match)."""
        result = await self.db.execute(
            select(DBGroundTruthDataset).where(
                DBGroundTruthDataset.query == query
            )
        )
        db_entry = result.scalar_one_or_none()
        
        if db_entry:
            return GroundTruthDataset.model_validate(db_entry)
        return None

    # ========================================================================
    # Paper Relevance Benchmarking
    # ========================================================================

    async def track_paper_relevance(
        self,
        query: str,
        retrieved_papers: List[RankedPaper],
        ground_truth_paper_ids: List[str],
        pipeline_type: str = "standard",
        query_intent: Optional[str] = None,
        breakdown_response: Optional[QuestionBreakdownResponse] = None,
    ) -> PaperRelevanceBenchmarkResult:
        """
        Track paper relevance metrics.
        
        Args:
            query: User query
            retrieved_papers: List of RankedPaper objects from RAGResult
            ground_truth_paper_ids: List of relevant paper IDs
            pipeline_type: "standard" or "hybrid"
            query_intent: Detected query intent
            breakdown_response: Query breakdown
            
        Returns:
            Paper relevance benchmark result
        """
        # Extract paper IDs in rank order
        retrieved_paper_ids = [p.paper_id for p in retrieved_papers]
        relevant_set = set(ground_truth_paper_ids)
        
        # Calculate metrics
        k_values = [5, 10, 20]
        all_metrics = metrics_calc.analyze_retrieval_metrics(
            retrieved=retrieved_paper_ids,
            relevant=relevant_set,
            k_values=k_values
        )
        
        # Prepare metadata
        papers_metadata = {
            p.paper_id: {
                "relevance_score": p.relevance_score,
                "ranking_scores": p.ranking_scores,
                "title": p.paper.title if hasattr(p.paper, 'title') else None,
            }
            for p in retrieved_papers
        }
        
        # Create database record
        db_benchmark = DBPaperRelevanceBenchmark(
            query=query,
            ground_truth_paper_ids=ground_truth_paper_ids,
            retrieved_paper_ids=retrieved_paper_ids,
            retrieved_papers_metadata=papers_metadata,
            precision_at_5=all_metrics["precision"].get(5),
            precision_at_10=all_metrics["precision"].get(10),
            precision_at_20=all_metrics["precision"].get(20),
            recall_at_5=all_metrics["recall"].get(5),
            recall_at_10=all_metrics["recall"].get(10),
            recall_at_20=all_metrics["recall"].get(20),
            ndcg_at_5=all_metrics["ndcg"].get(5),
            ndcg_at_10=all_metrics["ndcg"].get(10),
            ndcg_at_20=all_metrics["ndcg"].get(20),
            mrr=all_metrics["mrr"],
            true_positives=all_metrics["true_positives"],
            false_positives=all_metrics["false_positives"],
            false_negatives=all_metrics["false_negatives"],
            pipeline_type=pipeline_type,
            query_intent=query_intent,
            breakdown_response=breakdown_response.model_dump() if breakdown_response else None,
        )
        
        self.db.add(db_benchmark)
        await self.db.commit()
        await self.db.refresh(db_benchmark)
        
        logger.info(
            f"Paper relevance benchmark saved: P@10={db_benchmark.precision_at_10:.2f}, "
            f"R@10={db_benchmark.recall_at_10:.2f}, NDCG@10={db_benchmark.ndcg_at_10:.2f}"
        )
        
        return PaperRelevanceBenchmarkResult.model_validate(db_benchmark)

    # ========================================================================
    # Chunk Quality Benchmarking
    # ========================================================================

    async def track_chunk_quality(
        self,
        query: str,
        retrieved_chunks: List[ChunkRetrieved],
        pipeline_type: str = "standard",
        search_scope: Optional[str] = None,
    ) -> ChunkQualityBenchmarkResult:
        """
        Track chunk quality metrics.
        
        Args:
            query: User query
            retrieved_chunks: List of ChunkRetrieved objects from RAGResult
            pipeline_type: "standard" or "hybrid"
            search_scope: "scoped" or "full_database"
            
        Returns:
            Chunk quality benchmark result
        """
        if not retrieved_chunks:
            logger.warning("No chunks to benchmark")
            return None
        
        # Extract relevance scores
        relevance_scores = [c.relevance_score for c in retrieved_chunks]
        
        # Quality metrics
        avg_relevance = statistics.mean(relevance_scores)
        median_relevance = statistics.median(relevance_scores)
        min_relevance = min(relevance_scores)
        max_relevance = max(relevance_scores)
        
        # Coverage metrics
        papers_covered = len(set(c.paper_id for c in retrieved_chunks))
        sections_covered = list(set(
            c.section_title for c in retrieved_chunks
            if c.section_title
        ))
        token_count_total = sum(c.token_count for c in retrieved_chunks)
        
        # Diversity metrics
        paper_distribution = dict(Counter(c.paper_id for c in retrieved_chunks))
        section_distribution = dict(Counter(
            c.section_title for c in retrieved_chunks
            if c.section_title
        ))
        
        # TODO: Semantic clustering (requires embeddings)
        chunk_clusters = None
        cluster_diversity_score = None
        
        # Prepare chunk metadata
        chunks_metadata = [
            {
                "chunk_id": c.chunk_id,
                "paper_id": c.paper_id,
                "relevance_score": c.relevance_score,
                "token_count": c.token_count,
                "section_title": c.section_title,
            }
            for c in retrieved_chunks
        ]
        
        # Create database record
        db_benchmark = DBChunkQualityBenchmark(
            query=query,
            retrieved_chunks=chunks_metadata,
            average_relevance=avg_relevance,
            median_relevance=median_relevance,
            min_relevance=min_relevance,
            max_relevance=max_relevance,
            papers_covered=papers_covered,
            sections_covered=sections_covered,
            token_count_total=token_count_total,
            paper_distribution=paper_distribution,
            section_distribution=section_distribution,
            chunk_clusters=chunk_clusters,
            cluster_diversity_score=cluster_diversity_score,
            pipeline_type=pipeline_type,
            search_scope=search_scope,
        )
        
        self.db.add(db_benchmark)
        await self.db.commit()
        await self.db.refresh(db_benchmark)
        
        logger.info(
            f"Chunk quality benchmark saved: avg_relevance={avg_relevance:.2f}, "
            f"papers_covered={papers_covered}, tokens={token_count_total}"
        )
        
        return ChunkQualityBenchmarkResult.model_validate(db_benchmark)

    # ========================================================================
    # Breakdown Quality Benchmarking
    # ========================================================================

    async def track_breakdown_quality(
        self,
        original_query: str,
        breakdown_response: QuestionBreakdownResponse,
        ground_truth_intent: Optional[str] = None,
    ) -> BreakdownQualityBenchmarkResult:
        """
        Track query breakdown quality metrics.
        
        Args:
            original_query: Original user query
            breakdown_response: QuestionBreakdownResponse from pipeline
            ground_truth_intent: Expected intent (if available)
            
        Returns:
            Breakdown quality benchmark result
        """
        # Query metrics
        num_queries = len(breakdown_response.search_queries)
        
        # TODO: Calculate query diversity (requires embeddings)
        query_diversity_score = None
        query_clarity_scores = None
        
        # Intent analysis
        predicted_intent = breakdown_response.intent.value if breakdown_response.intent else None
        intent_correct = (
            predicted_intent == ground_truth_intent
            if ground_truth_intent else None
        )
        intent_confidence = breakdown_response.intent_confidence
        
        # Complexity
        complexity = breakdown_response.complexity
        has_specific_papers = bool(breakdown_response.specific_papers)
        has_keyword_queries = bool(breakdown_response.bm25_query)
        
        # Create database record
        db_benchmark = DBBreakdownQualityBenchmark(
            original_query=original_query,
            breakdown_response=breakdown_response.model_dump(),
            num_queries=num_queries,
            query_diversity_score=query_diversity_score,
            query_clarity_scores=query_clarity_scores,
            predicted_intent=predicted_intent,
            ground_truth_intent=ground_truth_intent,
            intent_correct=intent_correct,
            intent_confidence=intent_confidence,
            complexity=complexity,
            has_specific_papers=has_specific_papers,
            has_keyword_queries=has_keyword_queries,
        )
        
        self.db.add(db_benchmark)
        await self.db.commit()
        await self.db.refresh(db_benchmark)
        
        logger.info(
            f"Breakdown quality benchmark saved: intent={predicted_intent}, "
            f"num_queries={num_queries}, complexity={complexity}"
        )
        
        return BreakdownQualityBenchmarkResult.model_validate(db_benchmark)

    # ========================================================================
    # Integrated Pipeline Benchmarking
    # ========================================================================

    async def create_pipeline_benchmark(
        self, data: PipelineBenchmarkCreate
    ) -> PipelineBenchmarkResult:
        """Create an integrated pipeline benchmark record."""
        db_benchmark = DBPipelineBenchmark(
            benchmark_id=data.benchmark_id,
            query=data.query,
            pipeline_type=data.pipeline_type,
            breakdown_time_ms=data.breakdown_time_ms,
            retrieval_time_ms=data.retrieval_time_ms,
            chunk_retrieval_time_ms=data.chunk_retrieval_time_ms,
            ranking_time_ms=data.ranking_time_ms,
            generation_time_ms=data.generation_time_ms,
            total_time_ms=data.total_time_ms,
            papers_retrieved=data.papers_retrieved,
            chunks_retrieved=data.chunks_retrieved,
            paper_relevance_benchmark_id=data.paper_relevance_benchmark_id,
            chunk_quality_benchmark_id=data.chunk_quality_benchmark_id,
            breakdown_quality_benchmark_id=data.breakdown_quality_benchmark_id,
            answer_validation_id=data.answer_validation_id,
            paper_precision_at_10=data.paper_precision_at_10,
            chunk_average_relevance=data.chunk_average_relevance,
            answer_relevance_score=data.answer_relevance_score,
            has_hallucination=data.has_hallucination,
            citation_accuracy=data.citation_accuracy,
            total_cost_usd=data.total_cost_usd,
            rag_result=data.rag_result,
            final_answer=data.final_answer,
        )
        
        self.db.add(db_benchmark)
        await self.db.commit()
        await self.db.refresh(db_benchmark)
        
        logger.info(f"Pipeline benchmark saved: {data.benchmark_id}")
        
        return PipelineBenchmarkResult.model_validate(db_benchmark)

    async def get_pipeline_benchmark(
        self, benchmark_id: str, include_components: bool = True
    ) -> Optional[PipelineBenchmarkResult]:
        """Get pipeline benchmark by ID with optional component loading."""
        result = await self.db.execute(
            select(DBPipelineBenchmark).where(
                DBPipelineBenchmark.benchmark_id == benchmark_id
            )
        )
        db_benchmark = result.scalar_one_or_none()
        
        if not db_benchmark:
            return None
        
        benchmark = PipelineBenchmarkResult.model_validate(db_benchmark)
        
        # Load component benchmarks if requested
        if include_components:
            if db_benchmark.paper_relevance_benchmark_id:
                result = await self.db.execute(
                    select(DBPaperRelevanceBenchmark).where(
                        DBPaperRelevanceBenchmark.id == db_benchmark.paper_relevance_benchmark_id
                    )
                )
                paper_bench = result.scalar_one_or_none()
                if paper_bench:
                    benchmark.paper_relevance_benchmark = PaperRelevanceBenchmarkResult.model_validate(paper_bench)
            
            if db_benchmark.chunk_quality_benchmark_id:
                result = await self.db.execute(
                    select(DBChunkQualityBenchmark).where(
                        DBChunkQualityBenchmark.id == db_benchmark.chunk_quality_benchmark_id
                    )
                )
                chunk_bench = result.scalar_one_or_none()
                if chunk_bench:
                    benchmark.chunk_quality_benchmark = ChunkQualityBenchmarkResult.model_validate(chunk_bench)
            
            if db_benchmark.breakdown_quality_benchmark_id:
                result = await self.db.execute(
                    select(DBBreakdownQualityBenchmark).where(
                        DBBreakdownQualityBenchmark.id == db_benchmark.breakdown_quality_benchmark_id
                    )
                )
                breakdown_bench = result.scalar_one_or_none()
                if breakdown_bench:
                    benchmark.breakdown_quality_benchmark = BreakdownQualityBenchmarkResult.model_validate(breakdown_bench)
        
        return benchmark

    # ========================================================================
    # Comparison & Analysis
    # ========================================================================

    async def compare_pipelines(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[BenchmarkComparisonReport]:
        """
        Compare standard vs hybrid pipeline performance for a query.
        
        Args:
            query: The query to compare
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Comparison report with metrics for each pipeline type
        """
        # Build query
        conditions = [DBPipelineBenchmark.query == query]
        if start_date:
            conditions.append(DBPipelineBenchmark.created_at >= start_date)
        if end_date:
            conditions.append(DBPipelineBenchmark.created_at <= end_date)
        
        result = await self.db.execute(
            select(DBPipelineBenchmark).where(and_(*conditions))
        )
        benchmarks = result.scalars().all()
        
        if not benchmarks:
            return None
        
        # Group by pipeline type
        by_pipeline = {"standard": [], "hybrid": []}
        for bench in benchmarks:
            if bench.pipeline_type in by_pipeline:
                by_pipeline[bench.pipeline_type].append(bench)
        
        # Calculate metrics for each pipeline
        pipeline_metrics = []
        for pipeline_type, pipeline_benchmarks in by_pipeline.items():
            if not pipeline_benchmarks:
                continue
            
            metrics = PipelineComparisonMetrics(
                pipeline_type=pipeline_type,
                avg_total_time_ms=statistics.mean([b.total_time_ms for b in pipeline_benchmarks]),
                avg_retrieval_time_ms=statistics.mean([
                    b.retrieval_time_ms for b in pipeline_benchmarks
                    if b.retrieval_time_ms
                ]) if any(b.retrieval_time_ms for b in pipeline_benchmarks) else None,
                avg_generation_time_ms=statistics.mean([
                    b.generation_time_ms for b in pipeline_benchmarks
                    if b.generation_time_ms
                ]) if any(b.generation_time_ms for b in pipeline_benchmarks) else None,
                avg_precision_at_10=statistics.mean([
                    b.paper_precision_at_10 for b in pipeline_benchmarks
                    if b.paper_precision_at_10 is not None
                ]) if any(b.paper_precision_at_10 is not None for b in pipeline_benchmarks) else None,
                avg_chunk_relevance=statistics.mean([
                    b.chunk_average_relevance for b in pipeline_benchmarks
                    if b.chunk_average_relevance is not None
                ]) if any(b.chunk_average_relevance is not None for b in pipeline_benchmarks) else None,
                avg_answer_relevance=statistics.mean([
                    b.answer_relevance_score for b in pipeline_benchmarks
                    if b.answer_relevance_score is not None
                ]) if any(b.answer_relevance_score is not None for b in pipeline_benchmarks) else None,
                hallucination_rate=sum(1 for b in pipeline_benchmarks if b.has_hallucination) / len(pipeline_benchmarks)
                if pipeline_benchmarks else None,
                avg_cost_usd=statistics.mean([
                    float(b.total_cost_usd) for b in pipeline_benchmarks
                    if b.total_cost_usd is not None
                ]) if any(b.total_cost_usd is not None for b in pipeline_benchmarks) else None,
            )
            pipeline_metrics.append(metrics)
        
        # Determine winners
        best_quality = None
        best_speed = None
        best_cost = None
        
        if len(pipeline_metrics) >= 2:
            # Quality: highest precision
            if all(m.avg_precision_at_10 for m in pipeline_metrics):
                best_quality = max(pipeline_metrics, key=lambda m: m.avg_precision_at_10 or 0).pipeline_type
            
            # Speed: lowest time
            best_speed = min(pipeline_metrics, key=lambda m: m.avg_total_time_ms).pipeline_type
            
            # Cost: lowest cost
            if all(m.avg_cost_usd for m in pipeline_metrics):
                best_cost = min(pipeline_metrics, key=lambda m: m.avg_cost_usd or float('inf')).pipeline_type
        
        return BenchmarkComparisonReport(
            query=query,
            run_count=len(benchmarks),
            pipelines=pipeline_metrics,
            best_quality_pipeline=best_quality,
            best_speed_pipeline=best_speed,
            best_cost_pipeline=best_cost,
            benchmark_ids=[b.benchmark_id for b in benchmarks],
            created_at=benchmarks[0].created_at,
        )

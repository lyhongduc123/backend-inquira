"""
Pydantic schemas for benchmarking RAG pipeline.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from app.core.model import CamelModel


# ============================================================================
# Paper Relevance Benchmarking
# ============================================================================

class PaperRelevanceBenchmarkCreate(CamelModel):
    """Request to create a paper relevance benchmark"""
    query: str
    retrieved_paper_ids: List[str]
    ground_truth_paper_ids: List[str]
    retrieved_papers_metadata: Dict[str, Any]  # {paper_id: RankedPaper dict}
    pipeline_type: str = "standard"  # "standard" or "hybrid"
    query_intent: Optional[str] = None
    breakdown_response: Optional[Dict[str, Any]] = None


class PaperRelevanceBenchmarkResult(CamelModel):
    """Paper relevance benchmark results"""
    id: int
    query: str
    
    # Metrics
    precision_at_5: Optional[float] = None
    precision_at_10: Optional[float] = None
    precision_at_20: Optional[float] = None
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    recall_at_20: Optional[float] = None
    ndcg_at_5: Optional[float] = None
    ndcg_at_10: Optional[float] = None
    ndcg_at_20: Optional[float] = None
    mrr: Optional[float] = None
    
    # Analysis
    true_positives: List[str]
    false_positives: List[str]
    false_negatives: List[str]
    
    # Context
    pipeline_type: str
    query_intent: Optional[str] = None
    created_at: datetime


# ============================================================================
# Chunk Quality Benchmarking
# ============================================================================

class ChunkQualityBenchmarkCreate(CamelModel):
    """Request to create a chunk quality benchmark"""
    query: str
    retrieved_chunks: List[Dict[str, Any]]  # ChunkRetrieved dicts
    pipeline_type: str = "standard"
    search_scope: Optional[str] = None  # "scoped" or "full_database"


class ChunkQualityBenchmarkResult(CamelModel):
    """Chunk quality benchmark results"""
    id: int
    query: str
    
    # Quality metrics
    average_relevance: Optional[float] = None
    median_relevance: Optional[float] = None
    min_relevance: Optional[float] = None
    max_relevance: Optional[float] = None
    
    # Coverage
    papers_covered: Optional[int] = None
    sections_covered: Optional[List[str]] = None
    token_count_total: Optional[int] = None
    
    # Diversity
    paper_distribution: Optional[Dict[str, int]] = None
    section_distribution: Optional[Dict[str, int]] = None
    chunk_clusters: Optional[int] = None
    cluster_diversity_score: Optional[float] = None
    
    # Context
    pipeline_type: str
    search_scope: Optional[str] = None
    created_at: datetime


# ============================================================================
# Breakdown Quality Benchmarking
# ============================================================================

class BreakdownQualityBenchmarkCreate(CamelModel):
    """Request to create a breakdown quality benchmark"""
    original_query: str
    breakdown_response: Dict[str, Any]  # QuestionBreakdownResponse dict
    ground_truth_intent: Optional[str] = None


class BreakdownQualityBenchmarkResult(CamelModel):
    """Breakdown quality benchmark results"""
    id: int
    original_query: str
    
    # Query metrics
    num_queries: int
    query_diversity_score: Optional[float] = None
    query_clarity_scores: Optional[List[float]] = None
    
    # Intent
    predicted_intent: str
    ground_truth_intent: Optional[str] = None
    intent_correct: Optional[bool] = None
    intent_confidence: Optional[float] = None
    
    # Effectiveness
    papers_retrieved_per_query: Optional[Dict[str, int]] = None
    total_unique_papers: Optional[int] = None
    query_overlap_papers: Optional[int] = None
    
    # Complexity
    complexity: Optional[str] = None
    has_specific_papers: Optional[bool] = None
    has_keyword_queries: Optional[bool] = None
    
    created_at: datetime


# ============================================================================
# Pipeline Benchmark (Integrated)
# ============================================================================

class PipelineBenchmarkCreate(CamelModel):
    """Request to create an integrated pipeline benchmark"""
    benchmark_id: str  # UUID
    query: str
    pipeline_type: str = "standard"
    
    # Timing
    breakdown_time_ms: Optional[int] = None
    retrieval_time_ms: Optional[int] = None
    chunk_retrieval_time_ms: Optional[int] = None
    ranking_time_ms: Optional[int] = None
    generation_time_ms: Optional[int] = None
    total_time_ms: int
    
    # Counts
    papers_retrieved: Optional[int] = None
    chunks_retrieved: Optional[int] = None
    
    # Component benchmark IDs
    paper_relevance_benchmark_id: Optional[int] = None
    chunk_quality_benchmark_id: Optional[int] = None
    breakdown_quality_benchmark_id: Optional[int] = None
    answer_validation_id: Optional[int] = None
    
    # Summary metrics
    paper_precision_at_10: Optional[float] = None
    chunk_average_relevance: Optional[float] = None
    answer_relevance_score: Optional[float] = None
    has_hallucination: Optional[bool] = None
    citation_accuracy: Optional[float] = None
    
    # Cost
    total_cost_usd: Optional[float] = None
    
    # Results
    rag_result: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None


class PipelineBenchmarkResult(CamelModel):
    """Integrated pipeline benchmark result"""
    id: int
    benchmark_id: str
    query: str
    pipeline_type: str
    
    # Timing
    breakdown_time_ms: Optional[int] = None
    retrieval_time_ms: Optional[int] = None
    chunk_retrieval_time_ms: Optional[int] = None
    ranking_time_ms: Optional[int] = None
    generation_time_ms: Optional[int] = None
    total_time_ms: int
    
    # Counts
    papers_retrieved: Optional[int] = None
    chunks_retrieved: Optional[int] = None
    
    # Summary metrics
    paper_precision_at_10: Optional[float] = None
    chunk_average_relevance: Optional[float] = None
    answer_relevance_score: Optional[float] = None
    has_hallucination: Optional[bool] = None
    citation_accuracy: Optional[float] = None
    
    # Cost
    total_cost_usd: Optional[float] = None
    
    # Component benchmarks (lazy loaded)
    paper_relevance_benchmark: Optional[PaperRelevanceBenchmarkResult] = None
    chunk_quality_benchmark: Optional[ChunkQualityBenchmarkResult] = None
    breakdown_quality_benchmark: Optional[BreakdownQualityBenchmarkResult] = None
    
    created_at: datetime


# ============================================================================
# Ground Truth Management
# ============================================================================

class GroundTruthDatasetCreate(CamelModel):
    """Request to create a ground truth dataset entry"""
    query: str
    relevant_paper_ids: List[str]
    query_intent: Optional[str] = None
    difficulty: Optional[str] = None  # "easy", "medium", "hard"
    domain: Optional[str] = None  # "NLP", "Computer Vision", etc.
    notes: Optional[str] = None
    created_by: Optional[str] = None


class GroundTruthDataset(CamelModel):
    """Ground truth dataset entry"""
    id: int
    query: str
    relevant_paper_ids: List[str]
    query_intent: Optional[str] = None
    difficulty: Optional[str] = None
    domain: Optional[str] = None
    notes: Optional[str] = None
    created_by: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


# ============================================================================
# Comparison & Analysis
# ============================================================================

class BenchmarkComparisonRequest(CamelModel):
    """Request to compare pipelines"""
    query: str
    pipeline_types: List[str] = ["standard", "hybrid"]
    ground_truth_paper_ids: Optional[List[str]] = None
    run_count: int = Field(1, ge=1, le=10, description="Number of runs per pipeline")


class PipelineComparisonMetrics(CamelModel):
    """Comparison metrics between two pipeline runs"""
    pipeline_type: str
    
    # Performance
    avg_total_time_ms: float
    avg_retrieval_time_ms: Optional[float] = None
    avg_generation_time_ms: Optional[float] = None
    
    # Quality
    avg_precision_at_10: Optional[float] = None
    avg_recall_at_10: Optional[float] = None
    avg_ndcg_at_10: Optional[float] = None
    avg_chunk_relevance: Optional[float] = None
    avg_answer_relevance: Optional[float] = None
    hallucination_rate: Optional[float] = None  # % of runs with hallucinations
    
    # Cost
    avg_cost_usd: Optional[float] = None


class BenchmarkComparisonReport(CamelModel):
    """Comparison report between pipelines"""
    query: str
    run_count: int
    
    pipelines: List[PipelineComparisonMetrics]
    
    # Winner analysis
    best_quality_pipeline: Optional[str] = None
    best_speed_pipeline: Optional[str] = None
    best_cost_pipeline: Optional[str] = None
    
    # Detailed results
    benchmark_ids: List[str]  # UUIDs of all benchmark runs
    
    created_at: datetime


class AggregateMetrics(CamelModel):
    """Aggregate metrics across multiple benchmarks"""
    total_benchmarks: int
    date_range: Dict[str, str]  # {"start": "2026-03-01", "end": "2026-03-02"}
    
    # By pipeline type
    metrics_by_pipeline: Dict[str, PipelineComparisonMetrics]
    
    # Quality trends
    avg_precision_at_10: float
    avg_recall_at_10: float
    avg_ndcg_at_10: float
    
    # Performance trends
    avg_total_time_ms: float
    avg_cost_usd: Optional[float] = None
    
    # Quality indicators
    hallucination_rate: float
    avg_citation_accuracy: float


# ============================================================================
# Benchmark Filters
# ============================================================================

class BenchmarkFilters(CamelModel):
    """Filters for querying benchmarks"""
    pipeline_type: Optional[str] = None
    query_intent: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_precision: Optional[float] = None
    has_hallucination: Optional[bool] = None
    limit: int = Field(50, ge=1, le=1000)
    offset: int = Field(0, ge=0)


class BenchmarkListResponse(CamelModel):
    """Paginated list of benchmarks"""
    total: int
    limit: int
    offset: int
    items: List[PipelineBenchmarkResult]


# ============================================================================
# Chunk Search Benchmarking
# ============================================================================


class SearchStageMetrics(CamelModel):
    """Retrieval metrics for a single ranking stage."""

    ndcg_at_10: float = 0.0
    ndcg_at_20: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    mrr: float = 0.0
    average_score: float = 0.0


class ChunkSearchBenchmarkRequest(CamelModel):
    """Request to run a chunk search benchmark."""

    dataset_name: str = "scifact"
    dataset_split: str = "test"
    max_queries: int = Field(50, ge=1, le=1000)
    top_papers: int = Field(20, ge=1, le=200)
    top_chunks: int = Field(10, ge=1, le=100)
    bm25_weight: float = Field(0.4, ge=0.0, le=1.0)
    semantic_weight: float = Field(0.6, ge=0.0, le=1.0)
    rerank: bool = True
    mmr: bool = True
    mmr_lambda: float = Field(0.5, ge=0.0, le=1.0)
    benchmark_db_url: Optional[str] = None
    force_reindex: bool = False
    output_dir: str = "evaluation_results/benchmark-runs"


class ChunkSearchModeResult(CamelModel):
    """Metrics for a single search strategy."""

    strategy: str
    paper_screening_latency_ms: float = 0.0
    chunk_search_latency_ms: float = 0.0
    rerank_latency_ms: Optional[float] = None
    mmr_latency_ms: Optional[float] = None
    candidate_count: int = 0
    retrieved_paper_ids: List[str] = []
    retrieved_chunk_ids: List[str] = []
    paper_metrics: SearchStageMetrics
    chunk_metrics_before_rerank: SearchStageMetrics
    chunk_metrics_after_rerank: Optional[SearchStageMetrics] = None
    chunk_metrics_after_mmr: Optional[SearchStageMetrics] = None


class ChunkSearchQueryComparison(CamelModel):
    """Per-query comparison between screened and flat chunk retrieval."""

    query_id: str
    query: str
    relevant_paper_count: int
    screened: ChunkSearchModeResult
    flat: ChunkSearchModeResult


class ChunkSearchBenchmarkSummary(CamelModel):
    """Aggregate metrics for a benchmark strategy."""

    strategy: str
    query_count: int
    average_paper_screening_latency_ms: float = 0.0
    average_chunk_search_latency_ms: float = 0.0
    average_rerank_latency_ms: Optional[float] = None
    average_mmr_latency_ms: Optional[float] = None
    paper_metrics: SearchStageMetrics
    chunk_metrics_before_rerank: SearchStageMetrics
    chunk_metrics_after_rerank: Optional[SearchStageMetrics] = None
    chunk_metrics_after_mmr: Optional[SearchStageMetrics] = None


class ChunkSearchBenchmarkReport(CamelModel):
    """Complete chunk-search benchmark report."""

    benchmark_id: str
    dataset_name: str
    dataset_split: str
    benchmark_db_url_used: str
    separate_database_used: bool
    query_count: int
    screened: ChunkSearchBenchmarkSummary
    flat: ChunkSearchBenchmarkSummary
    per_query: List[ChunkSearchQueryComparison]
    created_at: datetime
    artifact_path: Optional[str] = None

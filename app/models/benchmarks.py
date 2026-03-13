"""
Benchmark database models for RAG pipeline evaluation and comparison.
"""
from typing import TYPE_CHECKING
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import (
    Integer,
    String,
    DateTime,
    Text,
    Float,
    ForeignKey,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from app.models.base import DatabaseBase as Base
from datetime import datetime
import uuid

if TYPE_CHECKING:
    from app.models.answer_vaidations import DBAnswerValidation


class DBGroundTruthDataset(Base):
    """
    Ground truth dataset for paper relevance evaluation.
    Maps queries to known relevant papers (human-labeled).
    """
    __tablename__ = "ground_truth_datasets"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    
    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
        comment="User query or search question"
    )
    
    relevant_paper_ids: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        comment="List of paper IDs that are relevant to this query"
    )
    
    # Query metadata
    query_intent: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="Expected query intent (FOUNDATIONAL, COMPREHENSIVE_SEARCH, etc.)"
    )
    
    difficulty: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="Query difficulty: easy, medium, hard"
    )
    
    domain: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Academic domain (e.g., 'NLP', 'Computer Vision')"
    )
    
    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Additional notes about this ground truth entry"
    )
    
    # Metadata
    created_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=func.now(),
        nullable=True
    )


class DBPaperRelevanceBenchmark(Base):
    """
    Paper relevance benchmark results.
    Tracks how well retrieved papers match ground truth relevant papers.
    """
    __tablename__ = "paper_relevance_benchmarks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    
    # Query and results
    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
        comment="User query"
    )
    
    ground_truth_paper_ids: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        comment="List of relevant paper IDs (ground truth)"
    )
    
    retrieved_paper_ids: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        comment="List of retrieved paper IDs in order"
    )
    
    retrieved_papers_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        comment="Full RankedPaper snapshots with scores"
    )
    
    # Information Retrieval Metrics
    precision_at_5: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Precision at top 5 results"
    )
    
    precision_at_10: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Precision at top 10 results"
    )
    
    precision_at_20: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Precision at top 20 results"
    )
    
    recall_at_5: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Recall at top 5 results"
    )
    
    recall_at_10: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Recall at top 10 results"
    )
    
    recall_at_20: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Recall at top 20 results"
    )
    
    ndcg_at_5: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Normalized Discounted Cumulative Gain at 5"
    )
    
    ndcg_at_10: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Normalized Discounted Cumulative Gain at 10"
    )
    
    ndcg_at_20: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Normalized Discounted Cumulative Gain at 20"
    )
    
    mrr: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Mean Reciprocal Rank"
    )
    
    # Analysis
    true_positives: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=[],
        comment="Paper IDs correctly retrieved"
    )
    
    false_positives: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=[],
        comment="Paper IDs incorrectly retrieved (not relevant)"
    )
    
    false_negatives: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=[],
        comment="Relevant paper IDs that were missed"
    )
    
    # Pipeline context
    pipeline_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Pipeline type: 'standard' or 'hybrid'"
    )
    
    query_intent: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="Detected query intent"
    )
    
    breakdown_response: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="QuestionBreakdownResponse snapshot"
    )
    
    # Metadata
    ground_truth_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("ground_truth_datasets.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True
    )


class DBChunkQualityBenchmark(Base):
    """
    Chunk quality benchmark results.
    Tracks quality and diversity of retrieved chunks.
    """
    __tablename__ = "chunk_quality_benchmarks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    
    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
        comment="User query"
    )
    
    retrieved_chunks: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        comment="List of ChunkRetrieved snapshots with scores"
    )
    
    # Quality metrics
    average_relevance: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Average chunk relevance score"
    )
    
    median_relevance: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Median chunk relevance score"
    )
    
    min_relevance: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Minimum chunk relevance score"
    )
    
    max_relevance: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Maximum chunk relevance score"
    )
    
    # Coverage metrics
    papers_covered: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of unique papers represented"
    )
    
    sections_covered: Mapped[list | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="List of unique section titles"
    )
    
    token_count_total: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Total tokens across all chunks"
    )
    
    # Diversity metrics
    paper_distribution: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Distribution of chunks per paper {paper_id: count}"
    )
    
    section_distribution: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Distribution of chunks per section {section: count}"
    )
    
    chunk_clusters: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of semantic clusters detected"
    )
    
    cluster_diversity_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Diversity score (0-1, higher = more diverse)"
    )
    
    # Pipeline context
    pipeline_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Pipeline type: 'standard' or 'hybrid'"
    )
    
    search_scope: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="Search scope: 'scoped' (filtered papers) or 'full_database'"
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True
    )


class DBBreakdownQualityBenchmark(Base):
    """
    Query breakdown quality benchmark.
    Tracks quality of generated search queries and intent classification.
    """
    __tablename__ = "breakdown_quality_benchmarks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    
    original_query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
        comment="Original user query"
    )
    
    breakdown_response: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        comment="Full QuestionBreakdownResponse"
    )
    
    # Query quality metrics
    num_queries: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Number of generated search queries"
    )
    
    query_diversity_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Semantic diversity of generated queries (0-1)"
    )
    
    query_clarity_scores: Mapped[list | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="LLM-judged clarity score per query"
    )
    
    # Intent classification
    predicted_intent: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Predicted query intent"
    )
    
    ground_truth_intent: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="Ground truth intent (if available)"
    )
    
    intent_correct: Mapped[bool | None] = mapped_column(
        nullable=True,
        comment="Whether intent prediction was correct"
    )
    
    intent_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Intent classification confidence"
    )
    
    # Retrieval effectiveness
    papers_retrieved_per_query: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Number of papers retrieved per query {query: count}"
    )
    
    total_unique_papers: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Total unique papers across all queries"
    )
    
    query_overlap_papers: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Papers retrieved by multiple queries (overlap)"
    )
    
    # Complexity
    complexity: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="Query complexity: simple, intermediate, advanced"
    )
    
    has_specific_papers: Mapped[bool | None] = mapped_column(
        nullable=True,
        comment="Whether breakdown includes specific paper titles"
    )
    
    has_keyword_queries: Mapped[bool | None] = mapped_column(
        nullable=True,
        comment="Whether breakdown includes keyword queries"
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True
    )


class DBPipelineBenchmark(Base):
    """
    Integrated pipeline benchmark.
    Links together all benchmark components for end-to-end evaluation.
    """
    __tablename__ = "pipeline_benchmarks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    
    benchmark_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
        comment="UUID for this benchmark run"
    )
    
    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
        comment="User query"
    )
    
    pipeline_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Pipeline type: 'standard' or 'hybrid'"
    )
    
    # Timing breakdown (milliseconds)
    breakdown_time_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Time for query breakdown"
    )
    
    retrieval_time_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Time for paper retrieval (S2/OA)"
    )
    
    chunk_retrieval_time_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Time for chunk retrieval"
    )
    
    ranking_time_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Time for ranking papers"
    )
    
    generation_time_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Time for LLM answer generation"
    )
    
    total_time_ms: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Total pipeline execution time"
    )
    
    # Counts
    papers_retrieved: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of papers retrieved"
    )
    
    chunks_retrieved: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of chunks retrieved"
    )
    
    # Foreign keys to detailed benchmarks
    paper_relevance_benchmark_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("paper_relevance_benchmarks.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    chunk_quality_benchmark_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("chunk_quality_benchmarks.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    breakdown_quality_benchmark_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("breakdown_quality_benchmarks.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    answer_validation_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("answer_validations.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    # Summary metrics (denormalized for quick access)
    paper_precision_at_10: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Paper retrieval precision at 10"
    )
    
    chunk_average_relevance: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Average chunk relevance score"
    )
    
    answer_relevance_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Answer relevance score"
    )
    
    has_hallucination: Mapped[bool | None] = mapped_column(
        nullable=True,
        comment="Whether answer contains hallucinations"
    )
    
    citation_accuracy: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Citation accuracy (0-1)"
    )
    
    # Cost tracking
    total_cost_usd: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Total LLM API cost in USD"
    )
    
    # Full results snapshot
    rag_result: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Complete RAGResult snapshot"
    )
    
    final_answer: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Generated answer text"
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True
    )

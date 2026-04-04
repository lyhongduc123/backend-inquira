"""Pydantic schemas for validation module."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.model import CamelModel


class ValidationRequest(CamelModel):
    """Request for answer validation inspection"""
    query: str
    context: str  # Context with chunks, paper IDs, and prompt
    enhanced_query: Optional[str] = None
    context_chunks: Optional[List[Dict[str, Any]]] = None
    generated_answer: Optional[str] = None
    model_name: str = "gpt-4o-mini"
    conversation_id: Optional[int] = None
    message_id: Optional[int] = None


class CitationAccuracy(CamelModel):
    """Citation accuracy metrics"""
    total_citations: int
    correct_citations: int
    hallucinated_citations: int
    missing_citations: int
    accuracy: float


class TextMatchAnalysis(CamelModel):
    """Detailed text matching analysis for frontend diff display"""
    matched_terms: List[str]  # Terms from answer found in context
    missing_terms: List[str]  # Terms from answer NOT in context
    match_percentage: float  # 0-100 percentage of answer terms found in context
    suspicious_sentences: List[str]  # Sentences with low term match


class ValidationClaim(CamelModel):
    """Per-claim support analysis."""

    claim: str
    support_score: float  # 0-1
    supported: bool
    missing_terms: List[str]


class ValidationComponentScores(CamelModel):
    """V2 component scores to support thesis-oriented analysis."""

    grounding_score: float
    citation_faithfulness_score: float
    relevance_score: float
    perspective_coverage_score: float
    overall_score: float


class ContextEvidence(CamelModel):
    """Context evidence summary parsed from validation context."""

    paper_ids: List[str]
    chunk_ids: List[str]
    total_papers: int
    total_chunks: int


class ValidationResult(CamelModel):
    """Detailed validation result for inspection"""
    query: str
    generated_answer: str
    context_used: str
    
    # Detailed analysis for frontend display
    text_match: TextMatchAnalysis
    context_evidence: ContextEvidence
    
    # Metrics
    has_hallucination: bool
    hallucination_count: int = 0
    hallucination_details: Optional[List[str]] = None
    non_existent_facts: Optional[List[str]] = None
    incorrect_citations: Optional[List[Dict[str, Any]]] = None
    
    citation_accuracy: Optional[CitationAccuracy] = None
    relevance_score: float  # 0-1
    factual_accuracy_score: float  # 0-1
    component_scores: ValidationComponentScores
    claims_checked: List[ValidationClaim] = []
    
    # Execution info
    execution_time_ms: int
    model_used: str
    validation_id: Optional[int] = None  # DB record ID


class ValidationInspection(CamelModel):
    """Complete validation inspection response"""
    validation_id: int
    timestamp: datetime
    
    # The validation result
    result: ValidationResult
    
    # Quick summary
    summary: Dict[str, Any]  # Quick stats for display


class ValidationHistoryItem(CamelModel):
    """Summary of a validation record"""
    id: int
    message_id: Optional[int]
    conversation_id: Optional[str] = None
    conversation_title: Optional[str] = None
    assistant_answer_preview: Optional[str] = None
    query_text: str
    model_name: str
    has_hallucination: bool
    relevance_score: Optional[float]
    factual_accuracy_score: Optional[float]
    citation_accuracy: Optional[float]
    execution_time_ms: Optional[int] = None
    total_citations: int = 0
    correct_citations: int = 0
    hallucinated_citations: int = 0
    missing_citations: int = 0
    context_evidence: Optional[ContextEvidence] = None
    created_at: datetime
    validated_at: Optional[datetime]


class ValidationHistoryResponse(CamelModel):
    """Validation history response with pagination metadata."""

    total: int
    skip: int
    limit: int
    validations: List[ValidationHistoryItem]


class ValidationDetail(CamelModel):
    """Detailed record for one validation run."""

    id: int
    message_id: Optional[int]
    query_text: str
    enhanced_query: Optional[str] = None
    generated_answer: Optional[str] = None
    context_used: Optional[str] = None
    context_chunks: Optional[List[Dict[str, Any]]] = None
    context_evidence: Optional[ContextEvidence] = None
    has_hallucination: bool
    hallucination_count: int = 0
    hallucination_details: Optional[List[str]] = None
    non_existent_facts: Optional[List[str]] = None
    incorrect_citations: Optional[List[Dict[str, Any]]] = None
    relevance_score: Optional[float]
    factual_accuracy_score: Optional[float]
    citation_accuracy: Optional[float]
    total_citations: int
    correct_citations: int
    hallucinated_citations: int
    missing_citations: int
    execution_time_ms: Optional[int]
    model_name: Optional[str]
    status: str
    component_scores: Optional[ValidationComponentScores] = None
    created_at: datetime
    validated_at: Optional[datetime]


class ValidationStats(CamelModel):
    """Aggregate validation statistics"""
    total_validations: int
    hallucination_rate: float
    average_relevance_score: float
    average_factual_accuracy: float
    average_citation_accuracy: float
    total_hallucinations: int
    total_incorrect_citations: int
    average_grounding_score: float = 0.0
    average_perspective_coverage: float = 0.0

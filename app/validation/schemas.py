"""
Pydantic Schemas for Validation Module
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ValidationRequest(BaseModel):
    """Request for answer validation inspection"""
    query: str
    context: str  # Context with chunks, paper IDs, and prompt
    generated_answer: Optional[str] = None  # If provided, validate this answer; otherwise generate new one
    model_name: str = "gpt-4o-mini"
    conversation_id: Optional[int] = None
    message_id: Optional[int] = None


class CitationAccuracy(BaseModel):
    """Citation accuracy metrics"""
    total_citations: int
    correct_citations: int
    hallucinated_citations: int
    missing_citations: int
    accuracy: float


class TextMatchAnalysis(BaseModel):
    """Detailed text matching analysis for frontend diff display"""
    matched_terms: List[str]  # Terms from answer found in context
    missing_terms: List[str]  # Terms from answer NOT in context
    match_percentage: float  # % of answer terms found in context
    suspicious_sentences: List[str]  # Sentences with low term match


class ValidationResult(BaseModel):
    """Detailed validation result for inspection"""
    query: str
    generated_answer: str
    context_used: str
    
    # Detailed analysis for frontend display
    text_match: TextMatchAnalysis
    
    # Metrics
    has_hallucination: bool
    hallucination_count: int = 0
    hallucination_details: Optional[List[str]] = None
    non_existent_facts: Optional[List[str]] = None
    incorrect_citations: Optional[List[Dict[str, Any]]] = None
    
    citation_accuracy: Optional[CitationAccuracy] = None
    relevance_score: float  # 0-1
    factual_accuracy_score: float  # 0-1
    
    # Execution info
    execution_time_ms: int
    model_used: str
    validation_id: Optional[int] = None  # DB record ID


class ValidationInspection(BaseModel):
    """Complete validation inspection response"""
    validation_id: int
    timestamp: datetime
    
    # The validation result
    result: ValidationResult
    
    # Quick summary
    summary: Dict[str, Any]  # Quick stats for display


class ValidationHistoryItem(BaseModel):
    """Summary of a validation record"""
    id: int
    message_id: Optional[int]
    query_text: str
    model_name: str
    has_hallucination: bool
    relevance_score: Optional[float]
    citation_accuracy: Optional[float]
    created_at: datetime
    validated_at: Optional[datetime]


class ValidationStats(BaseModel):
    """Aggregate validation statistics"""
    total_validations: int
    hallucination_rate: float
    average_relevance_score: float
    average_factual_accuracy: float
    average_citation_accuracy: float
    total_hallucinations: int
    total_incorrect_citations: int

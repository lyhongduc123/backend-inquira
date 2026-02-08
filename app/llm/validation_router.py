"""
Router for LLM Generation Validation and Testing
Provides endpoints for testing LLM outputs against ground truth
"""
from typing import List, Optional, Dict, Any, Set
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime
import json
import re

from app.db.database import get_db_session
from app.llm.lite_llm_provider import LiteLLMProvider
from app.models.queries import DBAnswerValidation
from app.extensions.logger import create_logger
from app.core.config import settings

logger = create_logger(__name__)

router = APIRouter()


# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class ValidationRequest(BaseModel):
    """Request for answer validation inspection"""
    query: str
    context: str  # Context with chunks, paper IDs, and prompt
    generated_answer: Optional[str] = None  # If provided, validate this answer; otherwise generate new one
    model_name: str = "gpt-4o-mini"
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None


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


# ============================================================================
# VALIDATION PROMPTS
# ============================================================================


# ============================================================================
# VALIDATION PROMPTS
# ============================================================================

RELEVANCE_SCORE_PROMPT = """You are a relevance evaluator. Rate how well the answer addresses the query on a scale of 0 to 1.

Scale:
- 1.0: Perfect answer, directly addresses all aspects of the query
- 0.7-0.9: Good answer, addresses main points
- 0.4-0.6: Partial answer, misses some key points
- 0.0-0.3: Poor answer, barely relevant or off-topic

Respond in JSON format with:
{
  "score": float between 0 and 1
}
"""


# ============================================================================
# VERIFICATION HELPER FUNCTIONS
# ============================================================================

def extract_citations(text: str) -> List[str]:
    """
    Extract citation IDs from text in formats like [1], [2], etc.
    Returns list of citation IDs found.
    """
    # Pattern matches [1], [2], [123] etc
    pattern = r'\[(\d+)\]'
    citations = re.findall(pattern, text)
    return list(set(citations))  # Remove duplicates


def extract_paper_ids_from_context(context: str) -> Set[str]:
    """
    Extract paper IDs from context string.
    Assumes context contains paper metadata with IDs in format 'id: <paper_id>'
    or 'paperId: <paper_id>' or 'corpus_id: <id>'.
    """
    ids = set()
    
    # Pattern for various ID formats
    patterns = [
        r'(?:^|\n)id:\s*(\S+)',
        r'(?:^|\n)paperId:\s*(\S+)',
        r'(?:^|\n)corpus_id:\s*(\d+)',
        r'(?:^|\n)corpusId:\s*(\d+)',
        r'"paperId":\s*"([^"]+)"',
        r'"id":\s*"([^"]+)"',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, context, re.MULTILINE | re.IGNORECASE)
        ids.update(matches)
    
    return ids


def verify_citations_against_context(
    answer: str, 
    context: str
) -> Dict[str, Any]:
    """
    Verify that citations in answer reference papers actually in context.
    
    Returns:
        {
            'total_citations': int,
            'correct_citations': int,
            'hallucinated_citations': int,
            'missing_citations': int,
            'incorrect_citation_details': List[Dict],
            'citation_accuracy': float
        }
    """
    # Extract citation markers from answer
    citations = extract_citations(answer)
    total_citations = len(citations)
    
    # Extract paper IDs from context
    context_paper_ids = extract_paper_ids_from_context(context)
    expected_ids_set = context_paper_ids
    
    # Check each citation
    incorrect_citations = []
    hallucinated_count = 0
    
    for citation_num in citations:
        # Check if citation number references a valid paper
        # This is heuristic - assumes citation [1] maps to 1st paper, etc.
        # More robust: parse answer for actual paper ID associations
        is_valid = len(expected_ids_set) >= int(citation_num)
        
        if not is_valid:
            hallucinated_count += 1
            incorrect_citations.append({
                'citation': f'[{citation_num}]',
                'reason': 'Citation number exceeds available papers',
                'expected_range': f'1-{len(expected_ids_set)}'
            })
    
    correct_citations = total_citations - hallucinated_count
    missing_citations = max(0, len(expected_ids_set) - total_citations)
    
    citation_accuracy = correct_citations / total_citations if total_citations > 0 else 0.0
    
    return {
        'total_citations': total_citations,
        'correct_citations': correct_citations,
        'hallucinated_citations': hallucinated_count,
        'missing_citations': missing_citations,
        'incorrect_citation_details': incorrect_citations,
        'citation_accuracy': citation_accuracy
    }


def analyze_text_matches(answer: str, context: str) -> TextMatchAnalysis:
    """
    Analyze which terms from answer are found in context.
    Returns detailed match info for frontend diff visualization.
    """
    # Extract all meaningful terms from answer
    answer_terms = [
        word.lower() for word in answer.split()
        if len(word) > 3 and word.isalnum()
    ]
    
    if not answer_terms:
        return TextMatchAnalysis(
            matched_terms=[],
            missing_terms=[],
            match_percentage=0.0,
            suspicious_sentences=[]
        )
    
    context_lower = context.lower()
    
    # Check which terms are in context
    matched = [term for term in answer_terms if term in context_lower]
    missing = [term for term in answer_terms if term not in context_lower]
    
    match_percentage = len(matched) / len(answer_terms) if answer_terms else 0.0
    
    # Find sentences with low match rate
    sentences = [s.strip() for s in answer.split('.') if s.strip()]
    suspicious = []
    
    for sentence in sentences:
        if len(sentence.split()) < 4:
            continue
        
        sent_terms = [
            word.lower() for word in sentence.split()
            if len(word) > 3 and word.isalnum()
        ]
        
        if sent_terms:
            sent_matched = [t for t in sent_terms if t in context_lower]
            sent_match_rate = len(sent_matched) / len(sent_terms)
            
            if sent_match_rate < 0.4:  # Less than 40% match
                suspicious.append(sentence)
    
    return TextMatchAnalysis(
        matched_terms=list(set(matched)),
        missing_terms=list(set(missing)),
        match_percentage=match_percentage,
        suspicious_sentences=suspicious
    )


def check_facts_in_context(answer: str, context: str) -> Dict[str, Any]:
    """
    Verify factual claims in answer against context using text analysis.
    Returns list of potential non-existent facts.
    
    Simple heuristic: extract key phrases from answer and check if they
    appear or are semantically similar to content in context.
    """
    # Split answer into sentences
    sentences = [s.strip() for s in answer.split('.') if s.strip()]
    
    non_existent_facts = []
    hallucination_details = []
    
    for sentence in sentences:
        # Skip very short sentences
        if len(sentence.split()) < 4:
            continue
        
        # Extract key terms (simple approach: words longer than 4 chars)
        key_terms = [
            word.lower() for word in sentence.split() 
            if len(word) > 4 and word.isalnum()
        ]
        
        # Check if key terms appear in context
        context_lower = context.lower()
        missing_terms = [term for term in key_terms if term not in context_lower]
        
        # If more than 60% of key terms missing, flag as potential hallucination
        if len(key_terms) > 0:
            missing_ratio = len(missing_terms) / len(key_terms)
            if missing_ratio > 0.6:
                non_existent_facts.append(sentence)
                hallucination_details.append(f"Missing terms: {', '.join(missing_terms)}")
    
    has_hallucination = len(non_existent_facts) > 0
    hallucination_count = len(non_existent_facts)
    
    return {
        'has_hallucination': has_hallucination,
        'hallucination_count': hallucination_count,
        'non_existent_facts': non_existent_facts,
        'hallucination_details': hallucination_details
    }


async def save_validation_result(
    db: AsyncSession,
    request: ValidationRequest,
    result: ValidationResult
) -> DBAnswerValidation:
    """
    Save validation result to database.
    """
    db_validation = DBAnswerValidation(
        query_text=request.query,
        context_text=request.context,
        generated_answer=result.generated_answer,
        model_name=result.model_used,
        conversation_id=request.conversation_id,
        message_id=request.message_id,
        
        # Metrics
        has_hallucination=result.has_hallucination,
        hallucination_count=result.hallucination_count,
        hallucination_details=result.hallucination_details,
        non_existent_facts=result.non_existent_facts,
        incorrect_citations=result.incorrect_citations,
        
        # Citation metrics
        total_citations=result.citation_accuracy.total_citations if result.citation_accuracy else 0,
        correct_citations=result.citation_accuracy.correct_citations if result.citation_accuracy else 0,
        hallucinated_citations=result.citation_accuracy.hallucinated_citations if result.citation_accuracy else 0,
        missing_citations=result.citation_accuracy.missing_citations if result.citation_accuracy else 0,
        
        # Scores
        relevance_score=result.relevance_score,
        factual_accuracy_score=result.factual_accuracy_score,
        citation_accuracy=result.citation_accuracy.accuracy if result.citation_accuracy else 0.0,
        
        # Execution info
        execution_time_ms=result.execution_time_ms,
    )
    
    db.add(db_validation)
    await db.commit()
    await db.refresh(db_validation)
    
    return db_validation


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/validate", response_model=ValidationInspection)
async def validate_answer(
    request: ValidationRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Validate an LLM-generated answer with detailed inspection.
    
    Returns detailed analysis for frontend visualization:
    - Text matching (terms found/missing)
    - Citation verification
    - Hallucination detection
    - Relevance scoring
    
    Context should contain: chunks text + paper IDs + prompt
    """
    import time
    
    start_time = time.time()
    
    # If answer not provided, generate it
    if not request.generated_answer:
        provider = LiteLLMProvider(model=request.model_name)
        
        messages = [
            {"role": "system", "content": "You are a helpful research assistant. Answer based ONLY on the provided context."},
            {"role": "user", "content": f"Context: {request.context}\\n\\nQuestion: {request.query}\\n\\nProvide a concise answer with citations if applicable."}
        ]
        
        generated = provider.simple_prompt(
            messages=messages,
            temperature=0.1
        )
        
        generated_answer = generated.choices[0].message.content
    else:
        generated_answer = request.generated_answer
    
    # ========================================================================
    # DETERMINISTIC VERIFICATION
    # ========================================================================
    
    # Analyze text matching for diff visualization
    text_match = analyze_text_matches(
        answer=generated_answer,
        context=request.context
    )
    
    # Verify citations against context (paper IDs extracted from context)
    citation_verification = verify_citations_against_context(
        answer=generated_answer,
        context=request.context
    )
    
    citation_accuracy_obj = CitationAccuracy(
        total_citations=citation_verification['total_citations'],
        correct_citations=citation_verification['correct_citations'],
        hallucinated_citations=citation_verification['hallucinated_citations'],
        missing_citations=citation_verification['missing_citations'],
        accuracy=citation_verification['citation_accuracy']
    )
    
    # Check facts in context using text analysis
    fact_verification = check_facts_in_context(
        answer=generated_answer,
        context=request.context
    )
    
    # Factual accuracy based on text matching (no LLM)
    factual_score = 1.0 - (fact_verification['hallucination_count'] * 0.2)
    factual_score = max(0.0, min(1.0, factual_score))
    
    # ========================================================================
    # LLM RELEVANCE SCORING
    # ========================================================================
    
    provider = LiteLLMProvider(model=request.model_name)
    relevance_check = provider.simple_prompt(
        messages=[
            {"role": "system", "content": RELEVANCE_SCORE_PROMPT},
            {"role": "user", "content": f"Query: {request.query}\\n\\nAnswer: {generated_answer}\\n\\nRate relevance 0-1. Respond with JSON: {{\\\"score\\\": float}}"}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    relevance_data = json.loads(relevance_check.choices[0].message.content)
    
    execution_time = int((time.time() - start_time) * 1000)
    
    # Create validation result
    validation_result = ValidationResult(
        query=request.query,
        generated_answer=generated_answer,
        context_used=request.context,
        
        # Text matching analysis
        text_match=text_match,
        
        # Hallucination metrics (deterministic only)
        has_hallucination=fact_verification['has_hallucination'],
        hallucination_count=fact_verification['hallucination_count'],
        hallucination_details=fact_verification['hallucination_details'],
        non_existent_facts=fact_verification['non_existent_facts'],
        incorrect_citations=citation_verification['incorrect_citation_details'] if citation_verification['incorrect_citation_details'] else None,
        
        # Accuracy metrics
        citation_accuracy=citation_accuracy_obj,
        relevance_score=relevance_data.get("score", 0.0),
        factual_accuracy_score=factual_score,
        
        # Execution info
        execution_time_ms=execution_time,
        model_used=request.model_name
    )
    
    # Save to database
    db_record = await save_validation_result(db, request, validation_result)
    validation_result.validation_id = db_record.id
    logger.info(f"Saved validation to database with ID: {db_record.id}")
    
    # Create summary for quick display
    summary = {
        "has_issues": fact_verification['has_hallucination'] or citation_verification['hallucinated_citations'] > 0,
        "text_match_percentage": text_match.match_percentage,
        "citation_accuracy": citation_accuracy_obj.accuracy,
        "relevance": relevance_data.get("score", 0.0),
        "issues_count": fact_verification['hallucination_count'] + citation_verification['hallucinated_citations']
    }
    
    return ValidationInspection(
        validation_id=db_record.id,
        timestamp=datetime.now(),
        result=validation_result,
        summary=summary
    )


@router.get("/history")
async def get_validation_history(
    skip: int = 0,
    limit: int = 50,
    conversation_id: Optional[str] = None,
    model_name: Optional[str] = None,
    has_hallucination: Optional[bool] = None,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get validation history with filtering.
    """
    from sqlalchemy import select, desc
    
    # Build query
    query = select(DBAnswerValidation)
    
    # Apply filters
    if conversation_id:
        query = query.where(DBAnswerValidation.conversation_id == conversation_id)
    if model_name:
        query = query.where(DBAnswerValidation.model_name == model_name)
    if has_hallucination is not None:
        query = query.where(DBAnswerValidation.has_hallucination == has_hallucination)
    
    # Order by most recent first
    query = query.order_by(desc(DBAnswerValidation.created_at))
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    validations = result.scalars().all()
    
    # Get total count
    count_query = select(DBAnswerValidation)
    if conversation_id:
        count_query = count_query.where(DBAnswerValidation.conversation_id == conversation_id)
    if model_name:
        count_query = count_query.where(DBAnswerValidation.model_name == model_name)
    if has_hallucination is not None:
        count_query = count_query.where(DBAnswerValidation.has_hallucination == has_hallucination)
    
    from sqlalchemy import func
    count_result = await db.execute(select(func.count()).select_from(count_query.subquery()))
    total = count_result.scalar()
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "test_cases": [
            {
                "id": tc.id,
                "test_case_name": tc.test_case_name,
                "test_suite_name": tc.test_suite_name,
                "query_text": tc.query_text,
                "generated_answer": tc.generated_answer,

    }


@router.get("/history/{validation_id}")
async def get_validation_detail(
    validation_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get detailed information about a specific validation.
    """
    from sqlalchemy import select
    
    result = await db.execute(
        select(DBAnswerValidation).where(DBAnswerValidation.id == validation_id)
    )
    validation = result.scalar_one_or_none()
    
    if not validation:
        raise HTTPException(status_code=404, detail="Validation not found")
    
    return {
        "id": validation.id,
        "query_text": validation.query_text,
        "context_text": validation.context_text,
        "generated_answer": validation.generated_answer,
        "model_name": validation.model_name,
        
        # Metrics
        "has_hallucination": test_case.has_hallucination,
        "hallucination_count": test_case.hallucination_count,
        "hallucination_details": test_case.hallucination_details,
        "non_existent_facts": test_case.non_existent_facts,
        "incorrect_citations": test_case.incorrect_citations,
        
        # Citation metrics
        "total_citations": test_case.total_citations,
        "correct_citations": test_case.correct_citations,
        "hallucinated_citations": test_case.hallucinated_citations,
        "missing_citations": test_case.missing_citations,
        
        # Scores
        "relevance_score": test_case.relevance_score,
        "factual_accuracy_score": test_case.factual_accuracy_score,
        "citation_accuracy": test_case.citation_accuracy,
        
        # Execution info
        "execution_time_ms": test_case.execution_time_ms,
        "created_at": test_case.created_at,
    }


@router.get("/stats")
async def get_validation_stats(
    conversation_id: Optional[str] = None,
    model_name: Optional[str] = None,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get aggregate statistics for validations.
    """
    from sqlalchemy import select, func
    
    # Build base query
    query = select(DBAnswerValidation)
    
    if conversation_id:
        query = query.where(DBAnswerValidation.conversation_id == conversation_id)
    if model_name:
        query = query.where(DBAnswerValidation.model_name == model_name)
    
    result = await db.execute(query)
    validations = result.scalars().all()
    
    if not validations:
        return {
            "total_validations": 0,
            "hallucination_rate": 0.0,
            "average_relevance_score": 0.0,
            "average_factual_accuracy": 0.0,
            "average_citation_accuracy": 0.0,
            "total_hallucinations": 0,
            "total_incorrect_citations": 0,
        }
    
    total = len(validations)
    with_hallucination = sum(1 for v in validations if v.has_hallucination)
    total_hallucination_count = sum(v.hallucination_count or 0 for v in validations)
    total_incorrect_citations = sum(v.hallucinated_citations or 0 for v in validations)
    
    avg_relevance = sum(v.relevance_score or 0.0 for v in validations) / total
    avg_factual = sum(v.factual_accuracy_score or 0.0 for v in validations) / total
    avg_citation = sum(v.citation_accuracy or 0.0 for v in validations) / total
    
    return {
        "total_validations": total,
        "hallucination_rate": with_hallucination / total,
        "average_relevance_score": avg_relevance,
        "average_factual_accuracy": avg_factual,
        "average_citation_accuracy": avg_citation,
        "total_hallucinations": total_hallucination_count,
        "total_incorrect_citations": total_incorrect_citations,
        "conversation_id": conversation_id,
        "model_name": model_name,
    }


@router.delete("/history/{validation_id}")
async def delete_validation(
    validation_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Delete a validation from history.
    """
    from sqlalchemy import select, delete
    
    result = await db.execute(
        select(DBAnswerValidation).where(DBAnswerValidation.id == validation_id)
    )
    validation = result.scalar_one_or_none()
    
    if not validation:
        raise HTTPException(status_code=404, detail="Validation not found")
    
    await db.execute(
        delete(DBAnswerValidation).where(DBAnswerValidation.id == validation_id)
    )
    await db.commit()
    
    return {"message": "Validation deleted successfully"}


@router.get("/test-sets")
async def get_test_sets():
    """Get predefined test sets for common scenarios"""
    return {
        "sets": [
            {
                "name": "Citation Accuracy",
                "description": "Tests accurate citation generation",
                "test_count": 10
            },
            {
                "name": "Hallucination Detection",
                "description": "Tests for unsupported claims",
                "test_count": 15
            },
            {
                "name": "Relevance Check",
                "description": "Tests answer relevance to query",
                "test_count": 12
            }
        ]
    }


# Validation prompts
HALLUCINATION_CHECK_PROMPT = """You are an expert fact-checker. Your task is to identify if the generated answer contains any claims that are NOT supported by the provided context.

A hallucination is any statement, fact, or claim that:
1. Cannot be verified from the given context
2. Adds information not present in the context
3. Misrepresents information from the context

Be strict in your evaluation."""

RELEVANCE_SCORE_PROMPT = """You are an expert evaluator. Rate how relevant the answer is to the query on a scale of 0 to 1.

0 = Completely irrelevant
0.5 = Partially relevant
1.0 = Highly relevant and directly answers the query

Consider:
- Does it address the query?
- Is it on-topic?
- Does it provide useful information?"""

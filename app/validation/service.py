"""
Validation Service
Business logic for answer validation, citation verification, and hallucination detection.
"""
from typing import List, Dict, Any, Set
import re
import json
import time
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from app.models.answer_vaidations import DBAnswerValidation
from app.llm.lite_llm_provider import LiteLLMProvider
from app.validation.schemas import (
    ValidationRequest,
    ValidationResult,
    CitationAccuracy,
    TextMatchAnalysis,
)
from app.extensions.logger import create_logger

logger = create_logger(__name__)


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
# HELPER FUNCTIONS
# ============================================================================

def extract_citations(text: str) -> List[str]:
    """
    Extract citation IDs from text in formats like [1], [2], etc.
    Returns list of citation IDs found.
    """
    pattern = r'\[(\d+)\]'
    citations = re.findall(pattern, text)
    return list(set(citations))


def extract_paper_ids_from_context(context: str) -> Set[str]:
    """
    Extract paper IDs from context string.
    """
    ids = set()
    
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
    """
    citations = extract_citations(answer)
    total_citations = len(citations)
    
    context_paper_ids = extract_paper_ids_from_context(context)
    expected_ids_set = context_paper_ids
    
    incorrect_citations = []
    hallucinated_count = 0
    
    for citation_num in citations:
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
    """
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
    
    matched = [term for term in answer_terms if term in context_lower]
    missing = [term for term in answer_terms if term not in context_lower]
    
    match_percentage = len(matched) / len(answer_terms) if answer_terms else 0.0
    
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
            
            if sent_match_rate < 0.4:
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
    """
    sentences = [s.strip() for s in answer.split('.') if s.strip()]
    
    non_existent_facts = []
    hallucination_details = []
    
    for sentence in sentences:
        if len(sentence.split()) < 4:
            continue
        
        key_terms = [
            word.lower() for word in sentence.split() 
            if len(word) > 4 and word.isalnum()
        ]
        
        context_lower = context.lower()
        missing_terms = [term for term in key_terms if term not in context_lower]
        
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


def generate_answer(
    query: str,
    context: str,
    model_name: str
) -> str:
    """Generate answer using LLM."""
    provider = LiteLLMProvider(model=model_name)
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful research assistant. Answer based ONLY on the provided context."
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {query}\n\nProvide a concise answer with citations if applicable."
        }
    ]
    
    # Use simple_prompt without streaming
    response = provider.simple_prompt(
        messages=messages,
        temperature=0.1  # type: ignore
    )
    
    return response.choices[0].message.content or ""  # type: ignore


def evaluate_relevance(
    query: str,
    answer: str,
    model_name: str
) -> float:
    """Evaluate answer relevance using LLM."""
    provider = LiteLLMProvider(model=model_name)
    
    messages = [
        {"role": "system", "content": RELEVANCE_SCORE_PROMPT},
        {
            "role": "user",
            "content": f"Query: {query}\n\nAnswer: {answer}\n\nRate relevance 0-1. Respond with JSON: {{\"score\": float}}"
        }
    ]
    
    response = provider.simple_prompt(
        messages=messages,
        response_format={"type": "json_object"},  # type: ignore
        temperature=0.0  # type: ignore
    )
    
    content = response.choices[0].message.content or "{}"  # type: ignore
    relevance_data = json.loads(content)
    
    return relevance_data.get("score", 0.0)


async def validate_answer(
    request: ValidationRequest
) -> ValidationResult:
    """
    Perform comprehensive answer validation.
    Returns detailed validation result.
    """
    start_time = time.time()
    
    # Generate answer if not provided
    if not request.generated_answer:
        generated_answer = generate_answer(
            request.query,
            request.context,
            request.model_name
        )
    else:
        generated_answer = request.generated_answer
    
    # Analyze text matching
    text_match = analyze_text_matches(
        answer=generated_answer,
        context=request.context
    )
    
    # Verify citations
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
    
    # Check facts
    fact_verification = check_facts_in_context(
        answer=generated_answer,
        context=request.context
    )
    
    # Calculate factual accuracy
    factual_score = 1.0 - (fact_verification['hallucination_count'] * 0.2)
    factual_score = max(0.0, min(1.0, factual_score))
    
    # Evaluate relevance using LLM
    relevance_score = evaluate_relevance(
        request.query,
        generated_answer,
        request.model_name
    )
    
    execution_time = int((time.time() - start_time) * 1000)
    
    return ValidationResult(
        query=request.query,
        generated_answer=generated_answer,
        context_used=request.context,
        text_match=text_match,
        has_hallucination=fact_verification['has_hallucination'],
        hallucination_count=fact_verification['hallucination_count'],
        hallucination_details=fact_verification['hallucination_details'],
        non_existent_facts=fact_verification['non_existent_facts'],
        incorrect_citations=citation_verification['incorrect_citation_details'] if citation_verification['incorrect_citation_details'] else None,
        citation_accuracy=citation_accuracy_obj,
        relevance_score=relevance_score,
        factual_accuracy_score=factual_score,
        execution_time_ms=execution_time,
        model_used=request.model_name
    )


async def save_validation_result(
    db: AsyncSession,
    request: ValidationRequest,
    result: ValidationResult
) -> DBAnswerValidation:
    """Save validation result to database."""
    db_validation = DBAnswerValidation(
        message_id=request.message_id,
        query_text=request.query,
        model_name=result.model_used,
        has_hallucination=result.has_hallucination,
        hallucination_count=result.hallucination_count,
        hallucination_details=result.hallucination_details,
        non_existent_facts=result.non_existent_facts,
        incorrect_citations=result.incorrect_citations,
        relevance_score=result.relevance_score,
        factual_accuracy_score=result.factual_accuracy_score,
        citation_accuracy=result.citation_accuracy.accuracy if result.citation_accuracy else 0.0,
        total_citations=result.citation_accuracy.total_citations if result.citation_accuracy else 0,
        correct_citations=result.citation_accuracy.correct_citations if result.citation_accuracy else 0,
        hallucinated_citations=result.citation_accuracy.hallucinated_citations if result.citation_accuracy else 0,
        missing_citations=result.citation_accuracy.missing_citations if result.citation_accuracy else 0,
        execution_time_ms=result.execution_time_ms,
        status="completed",
    )
    
    db.add(db_validation)
    await db.commit()
    await db.refresh(db_validation)
    
    return db_validation


async def get_validation_history(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 50,
    message_id: int | None = None,
    has_hallucination: bool | None = None
):
    """Get validation history with filters."""
    query = select(DBAnswerValidation)
    
    if message_id is not None:
        query = query.where(DBAnswerValidation.message_id == message_id)
    if has_hallucination is not None:
        query = query.where(DBAnswerValidation.has_hallucination == has_hallucination)
    
    query = query.order_by(desc(DBAnswerValidation.created_at))
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    validations = result.scalars().all()
    
    # Get total count
    count_query = select(func.count()).select_from(DBAnswerValidation)
    if message_id is not None:
        count_query = count_query.where(DBAnswerValidation.message_id == message_id)
    if has_hallucination is not None:
        count_query = count_query.where(DBAnswerValidation.has_hallucination == has_hallucination)
    
    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "validations": validations
    }


async def get_validation_stats(
    db: AsyncSession,
    message_id: int | None = None
):
    """Get aggregate validation statistics."""
    query = select(DBAnswerValidation)
    
    if message_id is not None:
        query = query.where(DBAnswerValidation.message_id == message_id)
    
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
    }

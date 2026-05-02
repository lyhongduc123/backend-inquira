"""Validation service for answer quality and faithfulness checks."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Set
import re
import time

from app.validation.schemas import (
    ContextEvidence,
    ValidationRequest,
    ValidationResult,
    ValidationInspection,
    ValidationComponentScores,
    CitationAccuracy,
    TextMatchAnalysis,
)
from app.validation.utils import extract_context_evidence
from app.extensions.citation_extractor import CitationExtractor
from app.extensions.logger import create_logger

logger = create_logger(__name__)

_STOPWORDS = {
    "the", "and", "that", "with", "from", "this", "those", "these", "their",
    "there", "where", "when", "which", "what", "about", "into", "than", "then",
    "were", "been", "being", "have", "has", "had", "does", "did", "while",
    "using", "used", "also", "such", "more", "most", "less", "many", "very",
    "some", "over", "under", "between", "among", "across", "within", "without",
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_terms(text: str, min_len: int = 4) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
    return [
        token for token in tokens
        if len(token) >= min_len and token not in _STOPWORDS
    ]


def extract_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# `extract_context_evidence` moved to app.validation.utils to avoid circular imports


def verify_citations_against_context(
    answer: str,
    context: str
) -> Dict[str, Any]:
    """
    Verify that citations in answer reference papers actually in context.
    """
    citations = CitationExtractor.extract_numeric_citations(answer)
    total_citations = len(citations)
    
    context_evidence = extract_context_evidence(context)
    expected_ids_set = set(context_evidence.paper_ids)
    
    incorrect_citations = []
    hallucinated_count = 0
    
    for citation_num in citations:
        is_valid = len(expected_ids_set) >= int(citation_num)
        
        if not is_valid:
            hallucinated_count += 1
            incorrect_citations.append(
                {
                    'citation': f'(cite:{citation_num})',
                    'reason': 'Citation number exceeds available papers',
                    'expected_range': f'1-{len(expected_ids_set)}',
                }
            )
    
    correct_citations = total_citations - hallucinated_count
    missing_citations = max(0, context_evidence.total_papers - total_citations)
    
    citation_accuracy = correct_citations / total_citations if total_citations > 0 else 0.0
    
    return {
        'total_citations': total_citations,
        'correct_citations': correct_citations,
        'hallucinated_citations': hallucinated_count,
        'missing_citations': missing_citations,
        'incorrect_citation_details': incorrect_citations,
        'citation_accuracy': citation_accuracy,
    }


def analyze_text_matches(answer: str, context: str) -> TextMatchAnalysis:
    """
    Analyze which terms from answer are found in context.
    """
    answer_terms = _normalize_terms(answer, min_len=4)
    
    if not answer_terms:
        return TextMatchAnalysis(
            matched_terms=[],
            missing_terms=[],
            match_percentage=0.0,
            suspicious_sentences=[],
        )
    
    context_lower = context.lower()
    
    matched = [term for term in answer_terms if term in context_lower]
    missing = [term for term in answer_terms if term not in context_lower]
    
    match_percentage = (
        (len(matched) / len(answer_terms)) * 100.0
        if answer_terms else 0.0
    )
    
    sentences = extract_sentences(answer)
    suspicious = []
    
    for sentence in sentences:
        if len(sentence.split()) < 4:
            continue
        
        sent_terms = _normalize_terms(sentence, min_len=4)
        
        if sent_terms:
            sent_matched = [t for t in sent_terms if t in context_lower]
            sent_match_rate = len(sent_matched) / len(sent_terms)
            
            if sent_match_rate < 0.4:
                suspicious.append(sentence)
    
    return TextMatchAnalysis(
        matched_terms=list(set(matched)),
        missing_terms=list(set(missing)),
        match_percentage=match_percentage,
        suspicious_sentences=suspicious,
    )


def build_validation_inspection(
    result: ValidationResult,
    validation_id: int | None,
) -> ValidationInspection:
    """Build validation inspection payload used by frontend."""
    return ValidationInspection(
        validation_id=validation_id or 0,
        timestamp=datetime.now(timezone.utc),
        result=result.model_copy(update={"validation_id": validation_id}),
        summary={
            "has_issues": result.has_hallucination
            or (result.citation_accuracy.hallucinated_citations > 0 if result.citation_accuracy else False),
            "text_match_percentage": result.text_match.match_percentage,
            "citation_accuracy": result.citation_accuracy.accuracy if result.citation_accuracy else 0.0,
            "relevance": result.relevance_score,
            "issues_count": result.hallucination_count + (
                result.citation_accuracy.hallucinated_citations if result.citation_accuracy else 0
            ),
            "overall_score": result.component_scores.overall_score,
            "grounding_score": result.component_scores.grounding_score,
            "perspective_coverage_score": result.component_scores.perspective_coverage_score,
        },
    )


async def validate_answer(
    request: ValidationRequest
) -> ValidationResult:
    """
    Perform comprehensive answer validation.
    Returns detailed validation result.
    """
    start_time = time.time()
    generated_answer = (request.generated_answer or "").strip()
    
    text_match = analyze_text_matches(
        answer=generated_answer,
        context=request.context
    )

    context_evidence = extract_context_evidence(request.context)
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
    
    citation_faithfulness_score = citation_verification['citation_accuracy']
    grounding_score = citation_faithfulness_score
    factual_score = citation_faithfulness_score
    relevance_score = 0.0
    perspective_coverage_score = 0.0

    overall_score = _clamp01(
        (0.80 * citation_faithfulness_score)
        + (0.20 * grounding_score)
    )

    component_scores = ValidationComponentScores(
        grounding_score=grounding_score,
        citation_faithfulness_score=citation_faithfulness_score,
        relevance_score=relevance_score,
        perspective_coverage_score=perspective_coverage_score,
        overall_score=overall_score,
    )
    
    execution_time = int((time.time() - start_time) * 1000)

    hallucinated_citations = citation_verification['hallucinated_citations']
    has_hallucination = hallucinated_citations > 0
    hallucination_details = (
        [
            f"Invalid citation {item.get('citation')}: {item.get('reason')}"
            for item in citation_verification['incorrect_citation_details']
        ]
        if citation_verification['incorrect_citation_details']
        else None
    )

    return ValidationResult(
        query=request.query,
        generated_answer=generated_answer,
        context_used=request.context,
        text_match=text_match,
        context_evidence=context_evidence,
        has_hallucination=has_hallucination,
        hallucination_count=hallucinated_citations,
        hallucination_details=hallucination_details,
        non_existent_facts=None,
        incorrect_citations=citation_verification['incorrect_citation_details'] if citation_verification['incorrect_citation_details'] else None,
        citation_accuracy=citation_accuracy_obj,
        relevance_score=relevance_score,
        factual_accuracy_score=factual_score,
        component_scores=component_scores,
        claims_checked=[],
        execution_time_ms=execution_time,
        model_used=request.model_name,
    )


# Repository wrapper helpers (moved from repository module for clearer service layer)
async def save_validation_result(db, request: ValidationRequest, result: ValidationResult):
    from app.validation.repository import ValidationRepository

    return await ValidationRepository(db).save_validation_result(request=request, result=result)


async def get_validation_history(db, **kwargs):
    from app.validation.repository import ValidationRepository

    return await ValidationRepository(db).get_validation_history(**kwargs)


async def get_validation_detail(db, validation_id: int):
    from app.validation.repository import ValidationRepository

    return await ValidationRepository(db).get_validation_detail(validation_id=validation_id)


async def get_validation_stats(db, message_id: int | None = None):
    from app.validation.repository import ValidationRepository

    return await ValidationRepository(db).get_validation_stats(message_id=message_id)


async def delete_validation(db, validation_id: int):
    from app.validation.repository import ValidationRepository

    return await ValidationRepository(db).delete_validation(validation_id=validation_id)

"""Validation service for answer quality and faithfulness checks."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Set, Tuple
import re
import json
import time
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from sqlalchemy.orm import joinedload

from app.models.answer_vaidations import DBAnswerValidation
from app.models.messages import DBMessage
from app.llm.lite_llm_provider import LiteLLMProvider
from app.validation.schemas import (
    ContextEvidence,
    ValidationRequest,
    ValidationResult,
    ValidationInspection,
    ValidationDetail,
    ValidationHistoryItem,
    ValidationHistoryResponse,
    ValidationComponentScores,
    ValidationClaim,
    CitationAccuracy,
    TextMatchAnalysis,
    ValidationStats,
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
# HELPER FUNCTIONS (V2)
# ============================================================================

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

def extract_citations(text: str) -> List[str]:
    """
    Extract citation IDs from text in formats like [1], [2], etc.
    Returns list of citation IDs found.
    """
    citations = re.findall(r'\[(\d+)\]', text)
    return sorted(list(set(citations)), key=int)


def extract_paper_ids_from_context(context: str) -> Set[str]:
    """
    Extract paper IDs from context string.
    """
    ids = set()
    
    patterns = [
        r'(?:^|\n)SOURCE_ID:\s*([^\n\r]+)',
        r'(?:^|\n)PAPER\s+ID:\s*([^\n\r]+)',
        r'(?:^|\n)paper_id:\s*([^\n\r]+)',
        r'(?:^|\n)paperId:\s*([^\n\r]+)',
        r'"paperId":\s*"([^"]+)"',
        r'"paper_id":\s*"([^"]+)"',
        r'"id":\s*"([^"]+)"',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, context, re.MULTILINE)
        ids.update(match.strip().strip(',') for match in matches)
    
    return ids


def extract_chunk_ids_from_context(context: str) -> Set[str]:
    """Extract chunk IDs from context string."""
    ids: Set[str] = set()
    patterns = [
        r'(?:^|\n)CHUNK_ID:\s*([^\n\r]+)',
        r'(?:^|\n)chunk_id:\s*([^\n\r]+)',
        r'"chunkId":\s*"([^"]+)"',
        r'"chunk_id":\s*"([^"]+)"',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, context, re.MULTILINE)
        ids.update(match.strip().strip(',') for match in matches)

    return ids


def extract_context_evidence(context: str) -> ContextEvidence:
    """Extract paper/chunk evidence lists from context."""
    paper_ids = sorted(list(extract_paper_ids_from_context(context)))
    chunk_ids = sorted(list(extract_chunk_ids_from_context(context)))
    return ContextEvidence(
        paper_ids=paper_ids,
        chunk_ids=chunk_ids,
        total_papers=len(paper_ids),
        total_chunks=len(chunk_ids),
    )


def verify_citations_against_context(
    answer: str,
    context: str
) -> Dict[str, Any]:
    """
    Verify that citations in answer reference papers actually in context.
    """
    citations = extract_citations(answer)
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
                    'citation': f'[{citation_num}]',
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


def extract_query_aspects(query: str) -> List[str]:
    """Heuristically derive aspects from user query."""
    parts = re.split(r"\b(?:and|or|vs|versus|compare|including|with|without)\b", query)
    aspects: List[str] = []
    for part in parts:
        terms = _normalize_terms(part, min_len=4)
        if terms:
            aspects.append(" ".join(terms[:3]))
    unique_aspects: List[str] = []
    seen: Set[str] = set()
    for aspect in aspects:
        if aspect not in seen:
            unique_aspects.append(aspect)
            seen.add(aspect)
    return unique_aspects[:8]


def analyze_claim_support(answer: str, context: str) -> List[ValidationClaim]:
    """Analyze claim-level support for answer sentences against context."""
    context_lower = context.lower()
    claims: List[ValidationClaim] = []
    for sentence in extract_sentences(answer):
        if len(sentence.split()) < 5:
            continue
        key_terms = _normalize_terms(sentence, min_len=5)
        if not key_terms:
            continue
        missing_terms = [term for term in key_terms if term not in context_lower]
        support_score = _clamp01(1.0 - (len(missing_terms) / len(key_terms)))
        claims.append(
            ValidationClaim(
                claim=sentence,
                support_score=support_score,
                supported=support_score >= 0.5,
                missing_terms=missing_terms,
            )
        )
    return claims


def check_facts_in_context(answer: str, context: str) -> Dict[str, Any]:
    """Verify factual claims in answer against context using claim-level analysis."""
    claims = analyze_claim_support(answer, context)
    unsupported_claims = [claim for claim in claims if not claim.supported]

    return {
        'has_hallucination': len(unsupported_claims) > 0,
        'hallucination_count': len(unsupported_claims),
        'non_existent_facts': [claim.claim for claim in unsupported_claims],
        'hallucination_details': [
            f"Missing terms: {', '.join(claim.missing_terms[:8])}" if claim.missing_terms
            else "Insufficient support in context"
            for claim in unsupported_claims
        ],
        'claims_checked': claims,
    }


def compute_perspective_coverage(query: str, answer: str) -> Tuple[float, List[str]]:
    """Compute lightweight perspective coverage from query aspects."""
    aspects = extract_query_aspects(query)
    if not aspects:
        return 1.0, []

    answer_lower = answer.lower()
    covered = [aspect for aspect in aspects if any(term in answer_lower for term in aspect.split())]
    coverage = len(covered) / len(aspects)
    return _clamp01(coverage), aspects


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
    
    try:
        response = provider.simple_prompt(
            messages=messages,
            response_format={"type": "json_object"},  # type: ignore
            temperature=0.0,  # type: ignore
        )

        content = response.choices[0].message.content or "{}"  # type: ignore
        relevance_data = json.loads(content)
        return _clamp01(float(relevance_data.get("score", 0.0)))
    except Exception:
        logger.warning("Failed relevance evaluation, fallback to lexical estimate", exc_info=True)
        query_terms = set(_normalize_terms(query, min_len=4))
        answer_terms = set(_normalize_terms(answer, min_len=4))
        if not query_terms:
            return 0.0
        overlap = len(query_terms & answer_terms) / len(query_terms)
        return _clamp01(overlap)


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

    context_evidence = extract_context_evidence(request.context)
    
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

    claims_checked = fact_verification['claims_checked']
    if claims_checked:
        grounding_score = _clamp01(
            sum(claim.support_score for claim in claims_checked) / len(claims_checked)
        )
    else:
        grounding_score = 0.0

    factual_score = grounding_score
    
    # Evaluate relevance using LLM
    relevance_score = evaluate_relevance(
        request.query,
        generated_answer,
        request.model_name
    )

    perspective_coverage_score, _ = compute_perspective_coverage(
        request.query,
        generated_answer,
    )

    citation_faithfulness_score = citation_verification['citation_accuracy']
    overall_score = _clamp01(
        (0.40 * grounding_score)
        + (0.25 * citation_faithfulness_score)
        + (0.20 * relevance_score)
        + (0.15 * perspective_coverage_score)
    )

    component_scores = ValidationComponentScores(
        grounding_score=grounding_score,
        citation_faithfulness_score=citation_faithfulness_score,
        relevance_score=relevance_score,
        perspective_coverage_score=perspective_coverage_score,
        overall_score=overall_score,
    )
    
    execution_time = int((time.time() - start_time) * 1000)
    
    return ValidationResult(
        query=request.query,
        generated_answer=generated_answer,
        context_used=request.context,
        text_match=text_match,
        context_evidence=context_evidence,
        has_hallucination=fact_verification['has_hallucination'],
        hallucination_count=fact_verification['hallucination_count'],
        hallucination_details=fact_verification['hallucination_details'],
        non_existent_facts=fact_verification['non_existent_facts'],
        incorrect_citations=citation_verification['incorrect_citation_details'] if citation_verification['incorrect_citation_details'] else None,
        citation_accuracy=citation_accuracy_obj,
        relevance_score=relevance_score,
        factual_accuracy_score=factual_score,
        component_scores=component_scores,
        claims_checked=claims_checked,
        execution_time_ms=execution_time,
        model_used=request.model_name,
    )


async def save_validation_result(
    db: AsyncSession,
    request: ValidationRequest,
    result: ValidationResult
) -> DBAnswerValidation:
    """Save validation result to database."""
    if request.message_id is None:
        raise ValueError("message_id is required to persist validation result")

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
        validated_at=datetime.now(timezone.utc),
    )

    db.add(db_validation)

    # Persist validation context snapshot inside message metadata for per-query audit.
    message_result = await db.execute(
        select(DBMessage).where(DBMessage.id == request.message_id)
    )
    message = message_result.scalar_one_or_none()
    if message:
        metadata = dict(message.message_metadata or {})
        metadata["validation_snapshot"] = {
            "query": request.query,
            "context": request.context,
            "generated_answer": result.generated_answer,
            "paper_ids": result.context_evidence.paper_ids,
            "chunk_ids": result.context_evidence.chunk_ids,
            "component_scores": result.component_scores.model_dump(),
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        message.message_metadata = metadata

    await db.commit()
    await db.refresh(db_validation)
    
    return db_validation


async def get_validation_history(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 50,
    message_id: int | None = None,
    has_hallucination: bool | None = None
) -> ValidationHistoryResponse:
    """Get validation history with filters."""
    query = select(DBAnswerValidation)
    
    if message_id is not None:
        query = query.where(DBAnswerValidation.message_id == message_id)
    if has_hallucination is not None:
        query = query.where(DBAnswerValidation.has_hallucination == has_hallucination)
    
    query = query.options(joinedload(DBAnswerValidation.message))
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
    
    history_items: List[ValidationHistoryItem] = []
    for validation in validations:
        context_evidence = None
        if validation.message and validation.message.message_metadata:
            snapshot = validation.message.message_metadata.get("validation_snapshot")
            if snapshot:
                context_evidence = ContextEvidence(
                    paper_ids=snapshot.get("paper_ids", []),
                    chunk_ids=snapshot.get("chunk_ids", []),
                    total_papers=len(snapshot.get("paper_ids", [])),
                    total_chunks=len(snapshot.get("chunk_ids", [])),
                )

        history_items.append(
            ValidationHistoryItem(
                id=validation.id,
                message_id=validation.message_id,
                query_text=validation.query_text,
                model_name=validation.model_name or "unknown",
                has_hallucination=validation.has_hallucination,
                relevance_score=validation.relevance_score,
                factual_accuracy_score=validation.factual_accuracy_score,
                citation_accuracy=validation.citation_accuracy,
                execution_time_ms=validation.execution_time_ms,
                total_citations=validation.total_citations,
                correct_citations=validation.correct_citations,
                hallucinated_citations=validation.hallucinated_citations,
                missing_citations=validation.missing_citations,
                context_evidence=context_evidence,
                created_at=validation.created_at,
                validated_at=validation.validated_at,
            )
        )

    return ValidationHistoryResponse(
        total=total,
        skip=skip,
        limit=limit,
        validations=history_items,
    )


async def get_validation_detail(
    db: AsyncSession,
    validation_id: int,
) -> ValidationDetail | None:
    """Get detailed validation record with context snapshot for per-query audit."""
    result = await db.execute(
        select(DBAnswerValidation)
        .options(joinedload(DBAnswerValidation.message))
        .where(DBAnswerValidation.id == validation_id)
    )
    validation = result.scalar_one_or_none()
    if not validation:
        return None

    context_used: str | None = None
    generated_answer: str | None = None
    context_evidence: ContextEvidence | None = None
    component_scores: ValidationComponentScores | None = None

    if validation.message and validation.message.message_metadata:
        snapshot = validation.message.message_metadata.get("validation_snapshot")
        if snapshot:
            context_used = snapshot.get("context")
            generated_answer = snapshot.get("generated_answer")
            paper_ids = snapshot.get("paper_ids", [])
            chunk_ids = snapshot.get("chunk_ids", [])
            context_evidence = ContextEvidence(
                paper_ids=paper_ids,
                chunk_ids=chunk_ids,
                total_papers=len(paper_ids),
                total_chunks=len(chunk_ids),
            )
            raw_component_scores = snapshot.get("component_scores")
            if isinstance(raw_component_scores, dict):
                component_scores = ValidationComponentScores.model_validate(raw_component_scores)

    if generated_answer is None and validation.message:
        generated_answer = validation.message.content

    if context_evidence is None and context_used:
        context_evidence = extract_context_evidence(context_used)

    hallucination_details: List[str] | None = None
    if isinstance(validation.hallucination_details, list):
        hallucination_details = [str(item) for item in validation.hallucination_details]

    non_existent_facts: List[str] | None = None
    if isinstance(validation.non_existent_facts, list):
        non_existent_facts = [str(item) for item in validation.non_existent_facts]

    incorrect_citations: List[Dict[str, Any]] | None = None
    if isinstance(validation.incorrect_citations, list):
        incorrect_citations = [
            item for item in validation.incorrect_citations if isinstance(item, dict)
        ]

    return ValidationDetail(
        id=validation.id,
        message_id=validation.message_id,
        query_text=validation.query_text,
        generated_answer=generated_answer,
        context_used=context_used,
        context_evidence=context_evidence,
        has_hallucination=validation.has_hallucination,
        hallucination_count=validation.hallucination_count,
        hallucination_details=hallucination_details,
        non_existent_facts=non_existent_facts,
        incorrect_citations=incorrect_citations,
        relevance_score=validation.relevance_score,
        factual_accuracy_score=validation.factual_accuracy_score,
        citation_accuracy=validation.citation_accuracy,
        total_citations=validation.total_citations,
        correct_citations=validation.correct_citations,
        hallucinated_citations=validation.hallucinated_citations,
        missing_citations=validation.missing_citations,
        execution_time_ms=validation.execution_time_ms,
        model_name=validation.model_name,
        status=validation.status,
        component_scores=component_scores,
        created_at=validation.created_at,
        validated_at=validation.validated_at,
    )


async def get_validation_stats(
    db: AsyncSession,
    message_id: int | None = None
) -> ValidationStats:
    """
    Get aggregate validation statistics.
    Includes breakdown by pipeline type for comparison.
    """
    query = select(DBAnswerValidation).options(
        joinedload(DBAnswerValidation.message)
    )
    
    if message_id is not None:
        query = query.where(DBAnswerValidation.message_id == message_id)
    
    result = await db.execute(query)
    validations = result.scalars().all()
    
    if not validations:
        return ValidationStats(
            total_validations=0,
            hallucination_rate=0.0,
            average_relevance_score=0.0,
            average_factual_accuracy=0.0,
            average_citation_accuracy=0.0,
            total_hallucinations=0,
            total_incorrect_citations=0,
            average_grounding_score=0.0,
            average_perspective_coverage=0.0,
        )
    
    total = len(validations)
    with_hallucination = sum(1 for v in validations if v.has_hallucination)
    total_hallucination_count = sum(v.hallucination_count or 0 for v in validations)
    total_incorrect_citations = sum(v.hallucinated_citations or 0 for v in validations)
    
    avg_relevance = sum(v.relevance_score or 0.0 for v in validations) / total
    avg_factual = sum(v.factual_accuracy_score or 0.0 for v in validations) / total
    avg_citation = sum(v.citation_accuracy or 0.0 for v in validations) / total
    
    grounding_scores: List[float] = []
    perspective_scores: List[float] = []
    for validation in validations:
        if validation.message and validation.message.message_metadata:
            snapshot = validation.message.message_metadata.get("validation_snapshot")
            if snapshot and isinstance(snapshot.get("component_scores"), dict):
                comp = snapshot["component_scores"]
                grounding_scores.append(float(comp.get("grounding_score", 0.0)))
                perspective_scores.append(float(comp.get("perspective_coverage_score", 0.0)))

    avg_grounding = sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0.0
    avg_perspective = sum(perspective_scores) / len(perspective_scores) if perspective_scores else 0.0

    return ValidationStats(
        total_validations=total,
        hallucination_rate=with_hallucination / total,
        average_relevance_score=avg_relevance,
        average_factual_accuracy=avg_factual,
        average_citation_accuracy=avg_citation,
        total_hallucinations=total_hallucination_count,
        total_incorrect_citations=total_incorrect_citations,
        average_grounding_score=avg_grounding,
        average_perspective_coverage=avg_perspective,
    )

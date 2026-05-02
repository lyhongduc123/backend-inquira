"""Repository layer for validation persistence and query operations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, func, select, delete as sql_delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models.answer_vaidations import DBAnswerValidation
from app.models.conversations import DBConversation
from app.models.messages import DBMessage
from app.validation.schemas import (
    ContextEvidence,
    ValidationComponentScores,
    ValidationDetail,
    ValidationHistoryItem,
    ValidationHistoryResponse,
    ValidationRequest,
    ValidationResult,
    ValidationStats,
)
from app.validation.utils import extract_context_evidence


class ValidationRepository:
    """Database access operations for answer validation records."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def save_validation_result(
        self,
        request: ValidationRequest,
        result: ValidationResult,
    ) -> DBAnswerValidation:
        if request.message_id is None:
            raise ValueError("message_id is required to persist validation result")

        db_validation = DBAnswerValidation(
            message_id=request.message_id,
            query_text=request.query,
            enhanced_query=request.enhanced_query,
            context_used=request.context,
            context_chunks=request.context_chunks,
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

        self.db.add(db_validation)
        await self.db.commit()
        await self.db.refresh(db_validation)
        return db_validation

    async def get_validation_history(
        self,
        *,
        skip: int = 0,
        limit: int = 50,
        message_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
        model_name: Optional[str] = None,
        query_text: Optional[str] = None,
        has_hallucination: Optional[bool] = None,
    ) -> ValidationHistoryResponse:
        query = select(DBAnswerValidation)

        if conversation_id is not None:
            query = query.join(DBAnswerValidation.message).join(DBMessage.conversation)

        if message_id is not None:
            query = query.where(DBAnswerValidation.message_id == message_id)
        if conversation_id is not None:
            query = query.where(DBConversation.conversation_id == conversation_id)
        if model_name is not None:
            query = query.where(DBAnswerValidation.model_name == model_name)
        if query_text is not None:
            query = query.where(DBAnswerValidation.query_text.ilike(f"%{query_text}%"))
        if has_hallucination is not None:
            query = query.where(DBAnswerValidation.has_hallucination == has_hallucination)

        query = query.options(joinedload(DBAnswerValidation.message).joinedload(DBMessage.conversation))
        query = query.order_by(desc(DBAnswerValidation.created_at))
        query = query.offset(skip).limit(limit)

        result = await self.db.execute(query)
        validations = result.scalars().all()

        count_query = select(func.count()).select_from(DBAnswerValidation)
        if conversation_id is not None:
            count_query = count_query.join(DBAnswerValidation.message).join(DBMessage.conversation)
        if message_id is not None:
            count_query = count_query.where(DBAnswerValidation.message_id == message_id)
        if conversation_id is not None:
            count_query = count_query.where(DBConversation.conversation_id == conversation_id)
        if model_name is not None:
            count_query = count_query.where(DBAnswerValidation.model_name == model_name)
        if query_text is not None:
            count_query = count_query.where(DBAnswerValidation.query_text.ilike(f"%{query_text}%"))
        if has_hallucination is not None:
            count_query = count_query.where(DBAnswerValidation.has_hallucination == has_hallucination)

        count_result = await self.db.execute(count_query)
        total = count_result.scalar() or 0

        history_items: List[ValidationHistoryItem] = []
        for validation in validations:
            conversation = validation.message.conversation if validation.message else None
            assistant_answer_preview = None
            if validation.message and validation.message.content:
                assistant_answer_preview = validation.message.content[:280]

            context_evidence = None
            if validation.context_used:
                context_evidence = extract_context_evidence(validation.context_used)
            elif validation.message and validation.message.message_metadata:
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
                    conversation_id=conversation.conversation_id if conversation else None,
                    conversation_title=conversation.title if conversation else None,
                    assistant_answer_preview=assistant_answer_preview,
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

    async def get_validation_detail(self, validation_id: int) -> ValidationDetail | None:
        result = await self.db.execute(
            select(DBAnswerValidation)
            .options(joinedload(DBAnswerValidation.message))
            .where(DBAnswerValidation.id == validation_id)
        )
        validation = result.scalar_one_or_none()
        if not validation:
            return None

        context_used: str | None = validation.context_used
        generated_answer: str | None = validation.message.content if validation.message else None
        context_evidence: ContextEvidence | None = None
        context_chunks: List[Dict[str, Any]] | None = None
        component_scores: ValidationComponentScores | None = None

        if isinstance(validation.context_chunks, list):
            context_chunks = [item for item in validation.context_chunks if isinstance(item, dict)]

        if context_used:
            context_evidence = extract_context_evidence(context_used)

        if (context_used is None or generated_answer is None or context_chunks is None) and validation.message and validation.message.message_metadata:
            snapshot = validation.message.message_metadata.get("validation_snapshot")
            if snapshot:
                context_used = context_used or snapshot.get("context")
                generated_answer = generated_answer or snapshot.get("generated_answer")
                paper_ids = snapshot.get("paper_ids", [])
                chunk_ids = snapshot.get("chunk_ids", [])
                if context_evidence is None:
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
            enhanced_query=validation.enhanced_query,
            generated_answer=generated_answer,
            context_used=context_used,
            context_chunks=context_chunks,
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

    async def get_validation_stats(self, message_id: Optional[int] = None) -> ValidationStats:
        query = select(DBAnswerValidation).options(joinedload(DBAnswerValidation.message))

        if message_id is not None:
            query = query.where(DBAnswerValidation.message_id == message_id)

        result = await self.db.execute(query)
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

    async def delete_validation(self, validation_id: int) -> bool:
        result = await self.db.execute(
            select(DBAnswerValidation).where(DBAnswerValidation.id == validation_id)
        )
        validation = result.scalar_one_or_none()
        if not validation:
            return False

        await self.db.execute(
            sql_delete(DBAnswerValidation).where(DBAnswerValidation.id == validation_id)
        )
        await self.db.commit()
        return True


async def save_validation_result(
    db: AsyncSession,
    request: ValidationRequest,
    result: ValidationResult,
) -> DBAnswerValidation:
    return await ValidationRepository(db).save_validation_result(request=request, result=result)


async def get_validation_history(
    db: AsyncSession,
    *,
    skip: int = 0,
    limit: int = 50,
    message_id: Optional[int] = None,
    conversation_id: Optional[str] = None,
    model_name: Optional[str] = None,
    query_text: Optional[str] = None,
    has_hallucination: Optional[bool] = None,
) -> ValidationHistoryResponse:
    return await ValidationRepository(db).get_validation_history(
        skip=skip,
        limit=limit,
        message_id=message_id,
        conversation_id=conversation_id,
        model_name=model_name,
        query_text=query_text,
        has_hallucination=has_hallucination,
    )


async def get_validation_detail(
    db: AsyncSession,
    validation_id: int,
) -> ValidationDetail | None:
    return await ValidationRepository(db).get_validation_detail(validation_id=validation_id)


async def get_validation_stats(
    db: AsyncSession,
    message_id: Optional[int] = None,
) -> ValidationStats:
    return await ValidationRepository(db).get_validation_stats(message_id=message_id)


async def delete_validation(
    db: AsyncSession,
    validation_id: int,
) -> bool:
    return await ValidationRepository(db).delete_validation(validation_id=validation_id)

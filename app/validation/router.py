"""
Validation Router
API endpoints for viewing validation statistics and benchmarking results.

Validation happens automatically after each chat response.
These endpoints are for viewing and analyzing the validation results.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from app.db.database import get_db_session
from app.models.answer_vaidations import DBAnswerValidation
from app.validation import schemas
from app.validation import service
from app.extensions.logger import create_logger

logger = create_logger(__name__)

router = APIRouter()


@router.get("/history")
async def get_validation_history(
    skip: int = 0,
    limit: int = 50,
    message_id: Optional[int] = None,
    has_hallucination: Optional[bool] = None,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get validation history with filtering.
    
    Validations are created automatically after each chat response.
    Use this endpoint to view and analyze validation results for benchmarking.
    """
    try:
        result = await service.get_validation_history(
            db=db,
            skip=skip,
            limit=limit,
            message_id=message_id,
            has_hallucination=has_hallucination
        )
        
        return {
            "total": result["total"],
            "skip": skip,
            "limit": limit,
            "validations": [
                {
                    "id": v.id,
                    "message_id": v.message_id,
                    "query_text": v.query_text,
                    "has_hallucination": v.has_hallucination,
                    "relevance_score": v.relevance_score,
                    "citation_accuracy": v.citation_accuracy,
                    "created_at": v.created_at,
                    "validated_at": v.validated_at,
                }
                for v in result["validations"]
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching validation history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{validation_id}")
async def get_validation_detail(
    validation_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get detailed information about a specific validation.
    
    View full validation results including hallucination details,
    citation accuracy, and relevance scores.
    """
    result = await db.execute(
        select(DBAnswerValidation).where(DBAnswerValidation.id == validation_id)
    )
    validation = result.scalar_one_or_none()
    
    if not validation:
        raise HTTPException(status_code=404, detail="Validation not found")
    
    return {
        "id": validation.id,
        "message_id": validation.message_id,
        "query_text": validation.query_text,
        "has_hallucination": validation.has_hallucination,
        "hallucination_count": validation.hallucination_count,
        "hallucination_details": validation.hallucination_details,
        "relevance_score": validation.relevance_score,
        "factual_accuracy_score": validation.factual_accuracy_score,
        "citation_accuracy": validation.citation_accuracy,
        "total_citations": validation.total_citations,
        "correct_citations": validation.correct_citations,
        "hallucinated_citations": validation.hallucinated_citations,
        "missing_citations": validation.missing_citations,
        "execution_time_ms": validation.execution_time_ms,
        "model_name": validation.model_name,
        "status": validation.status,
        "created_at": validation.created_at,
        "validated_at": validation.validated_at,
    }


@router.get("/stats")
async def get_validation_stats(
    message_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get aggregate validation statistics for benchmarking.
    
    Returns:
    - Total validations
    - Hallucination rate
    - Average relevance score
    - Average citation accuracy
    - Statistics by pipeline type (if available)
    """
    try:
        stats = await service.get_validation_stats(db=db, message_id=message_id)
        return stats
    except Exception as e:
        logger.error(f"Errorecord r fetching validation stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/{validation_id}")
async def delete_validation(
    validation_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """Delete a validation from history."""
    from sqlalchemy import delete as sql_delete
    
    result = await db.execute(
        select(DBAnswerValidation).where(DBAnswerValidation.id == validation_id)
    )
    validation = result.scalar_one_or_none()
    
    if not validation:
        raise HTTPException(status_code=404, detail="Validation not found")
    
    await db.execute(
        sql_delete(DBAnswerValidation).where(DBAnswerValidation.id == validation_id)
    )
    await db.commit()
    
    return {"message": "Validation deleted successfully"}

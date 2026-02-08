"""
Validation Router
API endpoints for LLM answer validation, citation verification, and hallucination detection.
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


@router.post("/validate", response_model=schemas.ValidationInspection)
async def validate_answer(
    request: schemas.ValidationRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Validate an LLM-generated answer with detailed inspection.
    
    Returns detailed analysis for frontend visualization:
    - Text matching (terms found/missing)
    - Citation verification
    - Hallucination detection
    - Relevance scoring
    """
    try:
        # Perform validation
        validation_result = await service.validate_answer(request)
        
        # Save to database
        db_record = await service.save_validation_result(db, request, validation_result)
        validation_result.validation_id = db_record.id
        
        logger.info(f"Validation completed and saved with ID: {db_record.id}")
        
        # Create summary
        summary = {
            "has_issues": validation_result.has_hallucination or 
                         (validation_result.citation_accuracy.hallucinated_citations > 0 if validation_result.citation_accuracy else False),
            "text_match_percentage": validation_result.text_match.match_percentage,
            "citation_accuracy": validation_result.citation_accuracy.accuracy if validation_result.citation_accuracy else 0.0,
            "relevance": validation_result.relevance_score,
            "issues_count": validation_result.hallucination_count + 
                          (validation_result.citation_accuracy.hallucinated_citations if validation_result.citation_accuracy else 0)
        }
        
        return schemas.ValidationInspection(
            validation_id=db_record.id,
            timestamp=datetime.now(),
            result=validation_result,
            summary=summary
        )
    except Exception as e:
        logger.error(f"Validation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/history")
async def get_validation_history(
    skip: int = 0,
    limit: int = 50,
    message_id: Optional[int] = None,
    has_hallucination: Optional[bool] = None,
    db: AsyncSession = Depends(get_db_session)
):
    """Get validation history with filtering."""
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
    """Get detailed information about a specific validation."""
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
    """Get aggregate statistics for validations."""
    try:
        stats = await service.get_validation_stats(db=db, message_id=message_id)
        return stats
    except Exception as e:
        logger.error(f"Error fetching validation stats: {str(e)}", exc_info=True)
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

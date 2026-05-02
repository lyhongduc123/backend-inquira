"""Validation router with inspection, history and stats endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db_session
from app.validation import schemas
from app.validation import service
from app.validation import repository
from app.extensions.logger import create_logger

logger = create_logger(__name__)

router = APIRouter()


@router.post("/validate", response_model=schemas.ValidationInspection)
async def validate_answer(
    request: schemas.ValidationRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Run validation v2 and optionally persist it when `message_id` is provided."""
    try:
        validation_result = await service.validate_answer(request)

        validation_id: int | None = None
        if request.message_id is not None:
            db_validation = await repository.save_validation_result(
                db=db,
                request=request,
                result=validation_result,
            )
            validation_id = db_validation.id

        return service.build_validation_inspection(validation_result, validation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Error validating answer: %s", str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail="Validation failed") from exc


@router.get("/history", response_model=schemas.ValidationHistoryResponse)
async def get_validation_history(
    skip: int = 0,
    limit: int = 50,
    message_id: Optional[int] = None,
    conversation_id: Optional[str] = None,
    model_name: Optional[str] = None,
    query_text: Optional[str] = None,
    has_hallucination: Optional[bool] = None,
    db: AsyncSession = Depends(get_db_session)
) -> schemas.ValidationHistoryResponse:
    """
    Get validation history with filtering.
    
    Validations are created automatically after each chat response.
    Use this endpoint to view and analyze validation results for benchmarking.
    """
    try:
        result = await repository.get_validation_history(
            db=db,
            skip=skip,
            limit=limit,
            message_id=message_id,
            conversation_id=conversation_id,
            model_name=model_name,
            query_text=query_text,
            has_hallucination=has_hallucination
        )
        return result
    except Exception as e:
        logger.error(f"Error fetching validation history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{validation_id}", response_model=schemas.ValidationDetail)
async def get_validation_detail(
    validation_id: int,
    db: AsyncSession = Depends(get_db_session)
) -> schemas.ValidationDetail:
    """
    Get detailed information about a specific validation.
    
    View full validation results including hallucination details,
    citation accuracy, and relevance scores.
    """
    validation = await repository.get_validation_detail(db=db, validation_id=validation_id)

    if not validation:
        raise HTTPException(status_code=404, detail="Validation not found")

    return validation


@router.get("/stats", response_model=schemas.ValidationStats)
async def get_validation_stats(
    message_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db_session)
) -> schemas.ValidationStats:
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
        stats = await repository.get_validation_stats(db=db, message_id=message_id)
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
    deleted = await repository.delete_validation(db=db, validation_id=validation_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Validation not found")

    return {"message": "Validation deleted successfully"}

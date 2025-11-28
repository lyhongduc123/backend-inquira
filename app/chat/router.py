"""
Chat router for handling chatbot interactions
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.chat.schemas import (
    ChatMessageRequest, 
    FeedbackRequest,
    FeedbackResponse
)
from app.llm.schemas import CitationBasedResponse
from app.chat.services import ChatService
from app.db.database import get_db_session

router = APIRouter()


@router.post("/stream")
async def stream_message(
    request: ChatMessageRequest,
    # user_id: int = Depends(get_current_user)  # TODO: Add auth dependency
    db: AsyncSession = Depends(get_db_session)
):
    """
    Stream chat message response in real-time with paper metadata
    
    Returns Server-Sent Events (SSE) stream with:
    1. event: sources - Paper metadata (JSON array)
    2. event: chunk - Response text chunks
    3. event: done - Completion signal
    
    Frontend should:
    - Parse 'sources' event to display citations
    - Accumulate 'chunk' events to build the response
    - Stop listening on 'done' event
    
    - **query**: User's message/question
    - **conversation_id**: Optional ID of existing conversation
    """
    chat_service = ChatService(db_session=db)
    try:
        print(f"[DEBUG] Stream endpoint called with query: {request.query[:50]}...")
        return StreamingResponse(
            chat_service.stream_message(
                request=request,
                user_id=None,  # TODO: Use actual user_id from auth
                db_session=db
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        print(f"[ERROR] Stream endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    # user_id: int = Depends(get_current_user)  # TODO: Add auth dependency
    db: AsyncSession = Depends(get_db_session)
):
    """
    Submit feedback/rating for a chat message
    
    - **message_id**: ID of the message being rated
    - **rating**: Rating from 1-5
    - **comment**: Optional feedback comment
    """
    try:
        chat_service = ChatService(db_session=db)  # TODO: Pass actual db_session if needed
        success = await chat_service.save_feedback(
            message_id=request.message_id,
            rating=request.rating,
            comment=request.comment
        )
        
        return FeedbackResponse(
            success=success,
            message="Feedback submitted successfully" if success else "Failed to submit feedback"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
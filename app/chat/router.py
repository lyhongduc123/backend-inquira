"""
Chat router for handling chatbot interactions
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import Optional
from app.chat.schemas import (
    ChatMessageRequest, 
    FeedbackRequest,
    FeedbackResponse
)
from app.llm.schemas import CitationBasedResponse
from app.chat.services import chat_service

router = APIRouter()


@router.post("/message", response_model=CitationBasedResponse)
async def send_message(
    request: ChatMessageRequest,
    # user_id: int = Depends(get_current_user)  # TODO: Add auth dependency
):
    """
    Send a chat message and get AI response with thought process and citations
    
    - **query**: User's message/question
    - **conversation_id**: Optional ID of existing conversation
    - **stream**: Whether to stream the response (use /stream endpoint instead)
    
    Returns response with:
    - Thought process steps showing reasoning
    - Citations from research papers
    - Final synthesized answer
    """
    try:
        response = await chat_service.process_message(
            request=request,
            user_id=None  # TODO: Use actual user_id from auth
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_message(
    request: ChatMessageRequest,
    # user_id: int = Depends(get_current_user)  # TODO: Add auth dependency
):
    """
    Stream chat message response in real-time
    
    - **query**: User's message/question
    - **conversation_id**: Optional ID of existing conversation
    """
    try:
        print(f"[DEBUG] Stream endpoint called with query: {request.query[:50]}...")
        return StreamingResponse(
            chat_service.stream_message(
                request=request,
                user_id=None  # TODO: Use actual user_id from auth
            ),
            media_type="text/plain; charset=utf-8"
        )
    except Exception as e:
        print(f"[ERROR] Stream endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    # user_id: int = Depends(get_current_user)  # TODO: Add auth dependency
):
    """
    Submit feedback/rating for a chat message
    
    - **message_id**: ID of the message being rated
    - **rating**: Rating from 1-5
    - **comment**: Optional feedback comment
    """
    try:
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

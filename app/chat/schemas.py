from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ChatMessageRequest(BaseModel):
    """Request model for sending a chat message"""
    query: str = Field(..., min_length=1, max_length=5000, description="User's message/question")
    conversation_id: Optional[int] = Field(None, description="ID of existing conversation")
    model: Optional[str] = Field(None, description="Optional model override")
    stream: bool = Field(True, description="Whether to stream the response")


class ChatMessageResponse(BaseModel):
    """Response model for chat message"""
    message: str = Field(..., description="AI assistant's response")
    conversation_id: int = Field(..., description="Conversation ID")
    message_id: int = Field(..., description="Message ID")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved sources/references")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class FeedbackRequest(BaseModel):
    """Request model for message feedback"""
    message_id: int = Field(..., description="ID of the message being rated")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    comment: Optional[str] = Field(None, max_length=1000, description="Optional feedback comment")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    success: bool
    message: str

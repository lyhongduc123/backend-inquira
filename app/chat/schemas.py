from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.core.model import CamelModel
from app.papers.schemas import PaperMetadata


class ChatMessageRequest(CamelModel):
    """Request model for sending a chat message"""
    query: str = Field(..., min_length=1, max_length=5000, description="User's message/question")
    conversation_id: Optional[str] = Field(None, description="UUID of existing conversation")
    filter: Optional[Dict[str, Any]] = Field(None, description="Optional filters for retrieval")
    model: Optional[str] = Field(None, description="Optional model override")
    stream: bool = Field(True, description="Whether to stream the response")
    is_retry: bool = Field(False, description="Whether this is a retry of a failed request")
    client_message_id: Optional[str] = Field(None, description="Client-generated message ID for deduplication on retry")


class ChatMessageResponse(CamelModel):
    """Response model for chat message"""
    message: str = Field(..., description="AI assistant's response")
    conversation_id: int = Field(..., description="Conversation ID")
    message_id: int = Field(..., description="Message ID")
    sources: Optional[List[PaperMetadata]] = Field(None, description="Retrieved paper sources with metadata")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class FeedbackRequest(CamelModel):
    """Request model for message feedback"""
    message_id: int = Field(..., description="ID of the message being rated")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    comment: Optional[str] = Field(None, max_length=1000, description="Optional feedback comment")


class FeedbackResponse(CamelModel):
    """Response model for feedback submission"""
    success: bool
    message: str


class PaperDetailChatRequest(CamelModel):
    """Request model for single-paper detail chat"""
    query: str = Field(..., min_length=1, max_length=5000, description="User's question about the paper")
    conversation_id: Optional[str] = Field(None, description="UUID of existing conversation (null = create new)")
    model: Optional[str] = Field(None, description="Optional model override")

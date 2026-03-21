from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from app.core.model import CamelModel
from app.domain.papers.schemas import PaperMetadata


class ChatMessageRequest(CamelModel):
    """Request model for sending a chat message"""
    query: str = Field(..., min_length=1, max_length=5000, description="User's message/question")
    conversation_id: Optional[str] = Field(None, description="UUID of existing conversation")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters for retrieval")
    model: Optional[str] = Field(None, description="Optional model override")
    stream: bool = Field(True, description="Whether to stream the response")
    is_retry: bool = Field(False, description="Whether this is a retry of a failed request")
    client_message_id: Optional[str] = Field(None, description="Client-generated message ID for deduplication on retry")
    pipeline: Literal["database", "hybrid", "standard"] = Field("database", description="Pipeline type: 'database' (fast DB-only), 'hybrid' (BM25+semantic with S2/OA), 'standard' (legacy)")
    use_hybrid_pipeline: bool = Field(False, description="Deprecated: use 'pipeline' field instead")


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


class ChatSubmitFilters(CamelModel):
    """Typed filters for chat submit requests."""
    author: Optional[str] = Field(None, description="Author name filter")
    year_min: Optional[int] = Field(None, description="Minimum publication year")
    year_max: Optional[int] = Field(None, description="Maximum publication year")
    venue: Optional[str] = Field(None, description="Venue/journal/conference filter")
    min_citations: Optional[int] = Field(None, description="Minimum citation count")
    max_citations: Optional[int] = Field(None, description="Maximum citation count")
    paper_ids: Optional[List[str]] = Field(
        None,
        description="Scoped paper IDs for paper-constrained answering (serialized as paperIds)",
    )


class ChatSubmitRequest(CamelModel):
    """Request model for submitting a chat message for async processing"""
    query: str = Field(..., min_length=1, max_length=5000, description="User's message/question")
    conversation_id: Optional[str] = Field(None, description="UUID of existing conversation")
    filters: Optional[ChatSubmitFilters] = Field(None, description="Optional typed filters for retrieval")
    model: Optional[str] = Field(None, description="Optional model override")
    client_message_id: Optional[str] = Field(None, description="Client-generated message ID for deduplication")
    pipeline: Literal["database", "hybrid", "standard"] = Field("database", description="Pipeline type")


class ChatSubmitResponse(CamelModel):
    """Response model for chat submission"""
    task_id: str = Field(..., description="Unique task identifier for tracking")
    conversation_id: str = Field(..., description="Conversation this task belongs to")
    status: str = Field(..., description="Initial task status (pending)")
    message: str = Field(..., description="Success message")


class PipelineTaskResponse(CamelModel):
    """Response model for pipeline task status"""
    task_id: str
    user_id: int
    conversation_id: str
    message_id: Optional[int]
    query: str
    pipeline_type: str
    status: str
    current_phase: Optional[str]
    progress_percent: int
    error_message: Optional[str]
    retry_count: int
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]


class PaperDetailChatRequest(CamelModel):
    """Request model for single-paper detail chat"""
    query: str = Field(..., min_length=1, max_length=5000, description="User's question about the paper")
    conversation_id: Optional[str] = Field(None, description="UUID of existing conversation (null = create new)")
    model: Optional[str] = Field(None, description="Optional model override")

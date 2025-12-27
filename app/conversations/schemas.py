from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List
from datetime import datetime


class ConversationCreate(BaseModel):
    """Request model for creating a new conversation"""
    title: Optional[str] = Field(None, max_length=200, description="Optional conversation title")


class ConversationUpdate(BaseModel):
    """Request model for updating a conversation"""
    title: Optional[str] = Field(None, max_length=200, description="New title")
    is_archived: Optional[bool] = Field(None, description="Archive status")


class Message(BaseModel):
    """Message within a conversation"""
    id: int
    role: str  # "user" or "assistant"
    content: str
    sources: Optional[List[Dict[str, Any]]] = None  # List of paper IDs
    created_at: datetime


class ConversationDetail(BaseModel):
    """Detailed conversation with messages"""
    id: str  # UUID string
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int
    is_archived: bool
    messages: List[Message]


class ConversationSummary(BaseModel):
    """Summary of a conversation for list view"""
    id: str  # UUID string
    title: Optional[str]
    preview: Optional[str] = None  # First message or summary
    last_updated: datetime
    message_count: int
    is_archived: bool


class ConversationListResponse(BaseModel):
    """Response model for conversation list"""
    conversations: List[ConversationSummary]
    total: int
    page: int
    page_size: int


class DeleteResponse(BaseModel):
    """Response for delete operations"""
    message: str

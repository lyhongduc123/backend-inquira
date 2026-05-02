"""
Pydantic schemas for message API contracts
"""
from pydantic import Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.core.model import CamelModel


class MessageCreate(CamelModel):
    """Request model for creating a new message"""
    conversation_id: str
    content: str
    role: str = Field(default="user", description="Message role: user or assistant")
    client_message_id: Optional[str] = Field(None, description="Client-side message ID for deduplication")


class MessageUpdate(CamelModel):
    """Request model for updating a message"""
    status: Optional[str] = Field(None, description="Message status: pending, sent, or failed")
    is_active: Optional[bool] = Field(None, description="Whether message is active")


class MessagePaperLink(CamelModel):
    """Request model for linking papers to a message"""
    paper_ids: List[str] = Field(description="List of paper IDs to link")
    paper_snapshots: Optional[List[Dict[str, Any]]] = Field(None, description="Optional paper metadata snapshots")


class MessageResponse(CamelModel):
    """Response model for a single message"""
    id: int
    conversation_id: str
    user_id: int
    role: str
    content: str
    status: str = "sent"
    pipeline_type: Optional[str] = None
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    message_metadata: Optional[Dict[str, Any]] = None


class MessageWithPapersResponse(MessageResponse):
    """Response model for message with linked papers"""
    paper_snapshots: Optional[List[Dict[str, Any]]] = None
    progress_events: Optional[List[Dict[str, Any]]] = None
    scoped_quote_refs: Optional[List[Dict[str, Any]]] = None


class MessageListResponse(CamelModel):
    """Response model for message list"""
    messages: List[MessageWithPapersResponse]
    total: int
    page: int
    page_size: int

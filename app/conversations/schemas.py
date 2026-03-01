from pydantic import BaseModel, Field, ConfigDict, field_serializer
from typing import Any, Dict, Optional, List
from datetime import datetime
from app.core.model import CamelModel
from pydantic.alias_generators import to_camel


def convert_dict_to_camel(data: Any) -> Any:
    """Recursively convert dictionary keys from snake_case to camelCase"""
    if isinstance(data, dict):
        return {to_camel(k): convert_dict_to_camel(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_dict_to_camel(item) for item in data]
    else:
        return data


class ConversationCreate(CamelModel):
    """Request model for creating a new conversation"""
    title: Optional[str] = Field(None, max_length=200, description="Optional conversation title")
    conversation_type: Optional[str] = Field("multi_paper_rag", description="Type of conversation")
    primary_paper_id: Optional[str] = Field(None, description="Primary paper ID for single-paper conversations")


class ConversationUpdate(CamelModel):
    """Request model for updating a conversation"""
    title: Optional[str] = Field(None, max_length=200, description="New title")
    is_archived: Optional[bool] = Field(None, description="Archive status")


class Message(CamelModel):
    """Message within a conversation"""
    id: int
    role: str  # "user" or "assistant"
    content: str
    sources: Optional[List[Dict[str, Any]]] = None  # Deprecated: use paper_snapshots
    paper_snapshots: Optional[List[Dict[str, Any]]] = None  # Paper metadata snapshots
    progress_events: Optional[List[Dict[str, Any]]] = None  # RAG pipeline progress events
    created_at: datetime
    
    @field_serializer('paper_snapshots', 'sources', 'progress_events')
    def serialize_metadata(self, value: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Convert nested dict keys to camelCase"""
        if value is None:
            return None
        return convert_dict_to_camel(value)


class ConversationDetail(CamelModel):
    """Detailed conversation with messages"""
    conversation_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int
    is_archived: bool
    conversation_type: str = "multi_paper_rag"
    primary_paper_id: Optional[str] = None
    messages: List[Message]


class ConversationSummary(CamelModel):
    """Summary of a conversation for list view"""
    id: str  # UUID string
    title: Optional[str]
    preview: Optional[str] = None  # First message or summary
    last_updated: datetime
    message_count: int
    is_archived: bool
    conversation_type: str = "multi_paper_rag"
    primary_paper_id: Optional[str] = None


class ConversationListResponse(CamelModel):
    """Response model for conversation list"""
    conversations: List[ConversationSummary]
    total: int
    page: int
    page_size: int


class DeleteResponse(CamelModel):
    """Response for delete operations"""
    message: str

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class PaperMetadata(BaseModel):
    """Paper metadata for citations and references"""
    paper_id: Optional[str] = Field(None, description="External paper ID (e.g., DOI, arXiv ID)")
    title: str = Field(..., description="Paper title")
    authors: Optional[List[str]] = Field(None, description="List of author names")
    year: Optional[int] = Field(None, description="Publication year")
    venue: Optional[str] = Field(None, description="Publication venue (journal, conference)")
    abstract: Optional[str] = Field(None, description="Paper abstract")
    pdf_url: Optional[str] = Field(None, description="Link to PDF")
    citation_count: Optional[int] = Field(None, description="Number of citations")
    influential_citation_count: Optional[int] = Field(None, description="Number of influential citations")
    source: Optional[str] = Field(None, description="Source database (semantic_scholar, arxiv, etc.)")


class ChatMessageRequest(BaseModel):
    """Request model for sending a chat message"""
    query: str = Field(..., min_length=1, max_length=5000, description="User's message/question")
    conversation_id: Optional[str] = Field(None, description="UUID of existing conversation")
    filter: Optional[Dict[str, Any]] = Field(None, description="Optional filters for retrieval")
    model: Optional[str] = Field(None, description="Optional model override")
    stream: bool = Field(True, description="Whether to stream the response")


class ChatMessageResponse(BaseModel):
    """Response model for chat message"""
    message: str = Field(..., description="AI assistant's response")
    conversation_id: int = Field(..., description="Conversation ID")
    message_id: int = Field(..., description="Message ID")
    sources: Optional[List[PaperMetadata]] = Field(None, description="Retrieved paper sources with metadata")
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

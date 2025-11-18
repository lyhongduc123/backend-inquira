"""
Chat and conversation response schemas
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class SearchSummaryResponse(BaseModel):
    """Response model for search results summary"""
    query: str = Field(..., description="Original search query")
    results_processed: int = Field(..., description="Number of results processed")
    total_results: int = Field(..., description="Total results available")
    summary: str = Field(..., description="Generated summary")
    model_used: str = Field(..., description="Model used for summarization")


class QuestionBreakdownResponse(BaseModel):
    """Response model for question breakdown"""
    original_question: str = Field(..., description="Original user question")
    clarified_question: str = Field(..., description="Clarified/refined question")
    subtopics: List[str] = Field(..., description="Breakdown sub-topics")
    num_subtopics: int = Field(..., description="Number of sub-topics")
    complexity: Literal["simple", "intermediate", "advanced"] = Field(..., description="Question complexity level")
    explanations: Optional[List[str]] = Field(None, description="Optional explanations for each subtopic")
    has_explanations: bool = Field(..., description="Whether explanations are included")
    model_used: str = Field(..., description="Model used for breakdown")


class ChatResponse(BaseModel):
    """Response model for chat interactions"""
    answer: str = Field(..., description="Generated answer")
    query: str = Field(..., description="User's query")
    sources_used: int = Field(..., description="Number of sources used")
    model_used: str = Field(..., description="Model used for response")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class RelatedTopicsResponse(BaseModel):
    """Response model for related topics suggestions"""
    current_topic: str = Field(..., description="Current research topic")
    suggestions: List[str] = Field(..., description="Related topic suggestions")
    num_suggestions: int = Field(..., description="Number of suggestions")
    model_used: str = Field(..., description="Model used for suggestions")


class Citation(BaseModel):
    """Citation reference from a paper or source"""
    paper_id: str = Field(..., description="Unique identifier for the paper")
    title: str = Field(..., description="Title of the paper")
    authors: Optional[List[str]] = Field(None, description="Authors of the paper")
    year: Optional[int] = Field(None, description="Publication year")
    quote: Optional[str] = Field(None, description="Direct quote from the paper")
    relevance: Optional[str] = Field(None, description="Why this citation is relevant")
    page: Optional[str] = Field(None, description="Page number or section")


class ThoughtStep(BaseModel):
    """Individual thought step in the reasoning process"""
    step_number: int = Field(..., description="Order of this thought step")
    thought: str = Field(..., description="The reasoning or thought process")
    citations: List[Citation] = Field(default_factory=list, description="Citations supporting this thought")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence level (0-1)")


class CitationBasedResponse(BaseModel):
    """Response with thought process and citations"""
    query: str = Field(..., description="User's original question")
    thought_process: List[ThoughtStep] = Field(..., description="Step-by-step thought process")
    final_answer: str = Field(..., description="Final synthesized answer")
    all_citations: List[Citation] = Field(..., description="All citations used in response")
    sources_count: int = Field(..., description="Number of unique sources cited")
    model_used: str = Field(..., description="Model used for generation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

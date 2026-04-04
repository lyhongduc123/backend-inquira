"""
Chat and conversation response schemas
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


class QueryIntent(str, Enum):
    """Query intent types for pipeline optimization"""
    COMPREHENSIVE_SEARCH = "comprehensive_search"  # Full pipeline: retrieve, rank, filter, embed
    AUTHOR_PAPERS = "author_papers"  # Author-specific: simple retrieval, skip scoring
    COMPARISON = "comparison"  # Compare specific papers: targeted retrieval
    FOUNDATIONAL = "foundational"  # Original/seminal papers: use specific paper titles


class SearchSummaryResponse(BaseModel):
    """Response model for search results summary"""
    query: str = Field(..., description="Original search query")
    results_processed: int = Field(..., description="Number of results processed")
    total_results: int = Field(..., description="Total results available")
    summary: str = Field(..., description="Generated summary")
    model_used: str = Field(..., description="Model used for summarization")


class QuestionBreakdownResponse(BaseModel):
    """Response model for question breakdown with intent classification"""
    original_question: str = Field(..., description="Original user question")
    clarified_question: str = Field(..., description="Clarified/refined question")
    search_queries: List[str] = Field(..., description="Optimized search queries for academic retrieval")
    bm25_query: Optional[str] = Field(default=None, description="Keyword queries for title/abstract matching")
    semantic_queries: Optional[List[str]] = Field(default=None, description="Semantic queries for contextual retrieval")
    specific_papers: Optional[List[str]] = Field(default=None, description="Specific paper titles to search for exact matches")
    num_queries: int = Field(..., description="Number of search queries")
    complexity: Literal["simple", "intermediate", "advanced"] = Field(..., description="Question complexity level")
    reasoning_content: Optional[str] = Field(None, description="LLM's reasoning process for generating the breakdown")
    model_used: str = Field(..., description="Model used for breakdown")
    
    # Query Intent Classification (merged from QueryIntentResponse)
    intent: Optional[QueryIntent] = Field(default=None, description="Classified query intent")
    intent_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Intent classification confidence")
    
    skip_ranking: bool = Field(default=False, description="Skip paper ranking step")
    skip_title_abstract_filter: bool = Field(default=False, description="Skip title/abstract similarity filter")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Extracted filters (author, year, venue)")


class ChatResponse(BaseModel):
    """Response model for chat interactions"""
    answer: str = Field(..., description="Generated answer")
    query: str = Field(..., description="User's query")
    sources_used: int = Field(..., description="Number of sources used")
    model_used: str = Field(..., description="Model used for response")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    llm_params_used: Optional[Dict[str, Any]] = Field(None, description="LLM parameters used for generation")


class RelatedTopicsResponse(BaseModel):
    """Response model for related topics suggestions"""
    current_topic: str = Field(..., description="Current research topic")
    suggestions: List[str] = Field(..., description="Related topic suggestions")
    num_suggestions: int = Field(..., description="Number of suggestions")
    model_used: str = Field(..., description="Model used for suggestions")

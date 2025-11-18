"""
Pydantic schemas for LLM service responses
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


# Base Response Models
class BaseResponse(BaseModel):
    """Base response model with common fields"""
    model_config = ConfigDict(from_attributes=True)
    
    model_used: str = Field(..., description="The LLM model used for generation")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class TokenUsage(BaseModel):
    """Token usage information"""
    prompt_tokens: Optional[int] = Field(None, description="Number of tokens in prompt")
    completion_tokens: Optional[int] = Field(None, description="Number of tokens in completion")
    total_tokens: Optional[int] = Field(None, description="Total tokens used")


# Analysis Response Models
class PaperAnalysisResponse(BaseResponse):
    """Response model for paper analysis"""
    analysis: str = Field(..., description="Comprehensive analysis of the paper")
    analysis_type: Literal["comprehensive", "methodology", "results", "literature_review"] = Field(
        ..., 
        description="Type of analysis performed"
    )
    paper_title: Optional[str] = Field(None, description="Title of the analyzed paper")
    

class KeywordExtractionResponse(BaseModel):
    """Response model for keyword extraction"""
    keywords: List[str] = Field(..., description="Extracted keywords")
    max_keywords: int = Field(..., description="Maximum number of keywords requested")
    include_phrases: bool = Field(..., description="Whether phrases were included")
    domain: Optional[str] = Field(None, description="Domain context used")
    model_used: str = Field(..., description="Model used for extraction")


class PaperComparisonResponse(BaseResponse):
    """Response model for paper comparison"""
    comparison: str = Field(..., description="Detailed comparison analysis")
    papers_compared: int = Field(..., description="Number of papers compared")
    comparison_aspects: List[str] = Field(..., description="Aspects compared")


class ResearchGapsResponse(BaseResponse):
    """Response model for research gap identification"""
    gaps_analysis: str = Field(..., description="Identified research gaps")
    research_area: str = Field(..., description="Research area analyzed")
    papers_analyzed: int = Field(..., description="Number of papers analyzed")


class MethodologyAnalysisResponse(BaseResponse):
    """Response model for methodology analysis"""
    methodology_analysis: str = Field(..., description="Detailed methodology analysis")
    focus_areas: List[str] = Field(..., description="Focus areas of the analysis")


class SentimentAnalysisResponse(BaseResponse):
    """Response model for sentiment analysis"""
    sentiment_analysis: str = Field(..., description="Sentiment analysis results")
    granularity: Literal["overall", "detailed", "aspect-based"] = Field(
        ..., 
        description="Granularity level of analysis"
    )


# Reading & Comprehension Response Models
class ExplanationResponse(BaseResponse):
    """Response model for content explanation"""
    explanation: str = Field(..., description="Generated explanation")
    explanation_level: Literal["beginner", "intermediate", "advanced"] = Field(
        ..., 
        description="Level of explanation"
    )
    target_audience: Literal["general", "students", "researchers", "professionals"] = Field(
        ..., 
        description="Target audience"
    )


class Question(BaseModel):
    """Single question model"""
    question: str = Field(..., description="The question text")
    answer: Optional[str] = Field(None, description="The answer")
    explanation: Optional[str] = Field(None, description="Explanation of the answer")
    question_type: Optional[str] = Field(None, description="Type of question")


class QuestionGenerationResponse(BaseResponse):
    """Response model for question generation"""
    questions: str = Field(..., description="Generated questions with answers")
    question_types: List[str] = Field(..., description="Types of questions generated")
    num_questions: int = Field(..., description="Number of questions generated")
    difficulty: str = Field(..., description="Difficulty level")


class StudyGuideResponse(BaseResponse):
    """Response model for study guide creation"""
    study_guide: str = Field(..., description="Complete study guide")
    sections_included: List[str] = Field(..., description="Sections included in the guide")


class InteractiveReadingResponse(BaseResponse):
    """Response model for interactive reading"""
    responses: List[str] = Field(..., description="Responses to user questions")
    context_mode: bool = Field(..., description="Whether context was maintained")
    num_questions: int = Field(..., description="Number of questions answered")


class MainIdeasResponse(BaseResponse):
    """Response model for main ideas extraction"""
    main_ideas: str = Field(..., description="Extracted main ideas")
    num_ideas: int = Field(..., description="Number of ideas extracted")
    include_supporting_details: bool = Field(..., description="Whether details were included")


class ConceptMapResponse(BaseResponse):
    """Response model for concept map creation"""
    concept_map: str = Field(..., description="Generated concept map")
    format_type: Literal["hierarchical", "network", "sequential"] = Field(
        ..., 
        description="Format type of the concept map"
    )


class ComprehensionTestResponse(BaseResponse):
    """Response model for comprehension test"""
    comprehension_test: str = Field(..., description="Generated comprehension test")
    num_questions: int = Field(..., description="Number of questions in test")
    include_answers: bool = Field(..., description="Whether answers are included")


# Summarization Response Models
class SummaryResponse(BaseResponse):
    """Response model for text summarization"""
    summary: str = Field(..., description="Generated summary")
    style: Literal["concise", "detailed", "bullet-points", "executive", "academic", "narrative"] = Field(
        ..., 
        description="Summary style used"
    )
    max_length: int = Field(..., description="Maximum length requested")
    original_length: Optional[int] = Field(None, description="Original text length")
    compression_ratio: Optional[float] = Field(None, description="Compression ratio achieved")


class ExecutiveSummaryResponse(BaseResponse):
    """Response model for executive summary"""
    executive_summary: str = Field(..., description="Executive summary")
    target_audience: str = Field(..., description="Target audience")
    key_points: List[str] = Field(default_factory=list, description="Key points emphasized")


class SummaryWithQuestionsResponse(BaseResponse):
    """Response model for summary with questions"""
    content: str = Field(..., description="Summary and questions")
    num_questions: int = Field(..., description="Number of follow-up questions")


class ProgressiveSummaryResponse(BaseResponse):
    """Response model for progressive summarization"""
    final_summary: str = Field(..., description="Final progressive summary")
    num_chunks: int = Field(..., description="Number of chunks processed")
    chunk_summaries: List[str] = Field(..., description="Individual chunk summaries")
    original_length: int = Field(..., description="Original text length")
    final_length: int = Field(..., description="Final summary length")
    compression_ratio: float = Field(..., description="Compression ratio")


# Service-Level Response Models
class SearchResultsSummaryResponse(BaseResponse):
    """Response model for search results summarization"""
    query: str = Field(..., description="Original search query")
    summary: Any = Field(..., description="Summary of search results (can be generator)")
    results_processed: int = Field(..., description="Number of results processed")
    total_results: int = Field(..., description="Total number of results")


class PaperContentAnalysisResponse(BaseResponse):
    """Response model for paper content analysis"""
    title: str = Field(..., description="Paper title")
    analysis: str = Field(..., description="Content analysis")
    keywords: List[str] = Field(..., description="Extracted keywords")
    analysis_type: str = Field(..., description="Type of analysis performed")


class ResearchQuestionsResponse(BaseModel):
    """Response model for research questions generation"""
    questions: List[str] = Field(..., description="Generated research questions")
    topic: str = Field(..., description="Research topic")
    num_questions: int = Field(..., description="Number of questions generated")
    model_used: str = Field(..., description="Model used")


class RelatedTopicsResponse(BaseModel):
    """Response model for related topic suggestions"""
    suggestions: List[str] = Field(..., description="Related topic suggestions")
    current_topic: str = Field(..., description="Current research topic")
    num_suggestions: int = Field(..., description="Number of suggestions")
    model_used: str = Field(..., description="Model used")


# Streaming Response Models
class StreamChunk(BaseModel):
    """Model for streaming response chunks"""
    chunk: str = Field(..., description="Text chunk")
    is_final: bool = Field(False, description="Whether this is the final chunk")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


# Error Response Models
class LLMErrorResponse(BaseModel):
    """Response model for LLM errors"""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now)


# Batch Processing Models
class BatchAnalysisRequest(BaseModel):
    """Request model for batch paper analysis"""
    papers: List[Dict[str, str]] = Field(..., description="List of papers to analyze")
    analysis_type: str = Field("comprehensive", description="Type of analysis")
    max_workers: int = Field(3, description="Maximum concurrent workers")


class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis"""
    results: List[PaperAnalysisResponse] = Field(..., description="Analysis results")
    total_papers: int = Field(..., description="Total papers processed")
    successful: int = Field(..., description="Successfully analyzed")
    failed: int = Field(..., description="Failed analyses")
    errors: List[str] = Field(default_factory=list, description="Error messages")


# Validation Models
class PromptValidationResponse(BaseModel):
    """Response for prompt validation"""
    is_valid: bool = Field(..., description="Whether prompt is valid")
    issues: List[str] = Field(default_factory=list, description="Validation issues")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")


# Usage Statistics
class UsageStatistics(BaseModel):
    """Model for tracking LLM usage"""
    total_requests: int = Field(0, description="Total number of requests")
    total_tokens: int = Field(0, description="Total tokens used")
    total_cost: float = Field(0.0, description="Total estimated cost")
    requests_by_type: Dict[str, int] = Field(
        default_factory=dict, 
        description="Request count by type"
    )
    average_response_time: float = Field(0.0, description="Average response time in seconds")


# Configuration Models
class LLMConfiguration(BaseModel):
    """Configuration for LLM services"""
    default_model: str = Field("gpt-4o-mini", description="Default model to use")
    default_temperature: float = Field(0.7, description="Default temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens per request")
    timeout: int = Field(30, description="Request timeout in seconds")
    retry_attempts: int = Field(3, description="Number of retry attempts")

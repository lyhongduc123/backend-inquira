"""
Analysis-related response schemas
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from .base import BaseResponse


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

"""
Batch processing and validation schemas
"""
from typing import List, Dict
from pydantic import BaseModel, Field
from .analysis import PaperAnalysisResponse


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


class PromptValidationResponse(BaseModel):
    """Response for prompt validation"""
    is_valid: bool = Field(..., description="Whether prompt is valid")
    issues: List[str] = Field(default_factory=list, description="Validation issues")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")

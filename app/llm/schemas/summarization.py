"""
Summarization response schemas
"""
from typing import List, Optional, Literal
from pydantic import Field
from .base import BaseResponse


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

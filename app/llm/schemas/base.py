"""
Base schemas for LLM responses
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


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


class StreamChunk(BaseModel):
    """Model for streaming response chunks"""
    chunk: str = Field(..., description="Text chunk")
    is_final: bool = Field(False, description="Whether this is the final chunk")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class LLMErrorResponse(BaseModel):
    """Response model for LLM errors"""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now)


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


class LLMConfiguration(BaseModel):
    """Configuration for LLM services"""
    default_model: str = Field("gpt-4o-mini", description="Default model to use")
    default_temperature: float = Field(0.7, description="Default temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens per request")
    timeout: int = Field(30, description="Request timeout in seconds")
    retry_attempts: int = Field(3, description="Number of retry attempts")

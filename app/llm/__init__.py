from typing import Optional
from .openai_client import OpenaiClient, ModelType
from .ollama_client import OllamaClient
from .provider import LLMProvider
from .lite_llm_provider import LiteLLMProvider
from .services import LLMService
from .schemas import (
    # Base Models
    BaseResponse,
    TokenUsage,
    StreamChunk,
    LLMErrorResponse,
    
    # Analysis Models
    PaperAnalysisResponse,
    KeywordExtractionResponse,
    PaperComparisonResponse,
    ResearchGapsResponse,
    MethodologyAnalysisResponse,
    SentimentAnalysisResponse,
    
    # Reading Models
    ExplanationResponse,
    QuestionGenerationResponse,
    StudyGuideResponse,
    InteractiveReadingResponse,
    MainIdeasResponse,
    ConceptMapResponse,
    ComprehensionTestResponse,
    
    # Summary Models
    SummaryResponse,
    ExecutiveSummaryResponse,
    SummaryWithQuestionsResponse,
    ProgressiveSummaryResponse,
    
    # Batch Models
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    
    # Config Models
    LLMConfiguration,
    UsageStatistics,
)

__all__ = [
    # Core Classes
    "OpenaiClient", 
    "ModelType", 
    "LLMProvider", 
    "LiteLLMProvider",
    "LLMService",
    "get_llm_service",  # Lazy getter
    
    # Base Models
    "BaseResponse",
    "TokenUsage",
    "StreamChunk",
    "LLMErrorResponse",
    
    # Analysis Models
    "PaperAnalysisResponse",
    "KeywordExtractionResponse",
    "PaperComparisonResponse",
    "ResearchGapsResponse",
    "MethodologyAnalysisResponse",
    "SentimentAnalysisResponse",
    
    # Reading Models
    "ExplanationResponse",
    "QuestionGenerationResponse",
    "StudyGuideResponse",
    "InteractiveReadingResponse",
    "MainIdeasResponse",
    "ConceptMapResponse",
    "ComprehensionTestResponse",
    
    # Summary Models
    "SummaryResponse",
    "ExecutiveSummaryResponse",
    "SummaryWithQuestionsResponse",
    "ProgressiveSummaryResponse",
    
    # Batch Models
    "BatchAnalysisRequest",
    "BatchAnalysisResponse",
    
    # Config Models
    "LLMConfiguration",
    "UsageStatistics",
]

# Lazy initialization to avoid slow startup
_llm_service: Optional["LLMService"] = None

def get_llm_service() -> "LLMService":
    """Get or create the singleton LLM service instance (lazy initialization)"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service

# For backward compatibility - returns the service when called
llm_service = get_llm_service()
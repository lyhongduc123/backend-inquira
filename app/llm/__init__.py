from typing import Optional
from .openai_client import OpenaiClient, ModelType
from .ollama_client import OllamaClient
from .provider import LLMProvider
from .lite_llm_provider import LiteLLMProvider
from .summarizer import Summarizer
from .analyzer import Analyzer
from .reader import Reader
from .services import LLMService
from .configs import PromptConfig, PromptPresets
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
    
    # Service Models
    RelatedTopicsResponse,
    
    # Batch Models
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    
    # Config Models
    LLMConfiguration,
    UsageStatistics,
)
from .schema_utils import (
    ResponseConverter,
    ResponseBuilder,
    dict_to_model,
    validate_response,
    response_to_dict,
    merge_responses,
    create_analysis_response,
    create_summary_response,
)

__all__ = [
    # Core Classes
    "OpenaiClient", 
    "ModelType", 
    "LLMProvider", 
    "LiteLLMProvider",
    "Summarizer", 
    "Analyzer", 
    "Reader", 
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
    
    # Service Models
    "RelatedTopicsResponse",
    
    # Batch Models
    "BatchAnalysisRequest",
    "BatchAnalysisResponse",
    
    # Config Models
    "LLMConfiguration",
    "UsageStatistics",
    
    # Prompt Configuration
    "PromptConfig",
    "PromptPresets",
    
    # Utilities
    "ResponseConverter",
    "ResponseBuilder",
    "dict_to_model",
    "validate_response",
    "response_to_dict",
    "merge_responses",
    "create_analysis_response",
    "create_summary_response",
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
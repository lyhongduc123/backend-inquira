"""
LLM schemas package - organized by category
"""
from .analysis import (
    PaperAnalysisResponse,
    KeywordExtractionResponse,
    PaperComparisonResponse,
    ResearchGapsResponse,
    MethodologyAnalysisResponse,
    SentimentAnalysisResponse
)
from .reading import (
    ExplanationResponse,
    Question,
    QuestionGenerationResponse,
    StudyGuideResponse,
    InteractiveReadingResponse,
    MainIdeasResponse,
    ConceptMapResponse,
    ComprehensionTestResponse
)
from .summarization import (
    SummaryResponse,
    ExecutiveSummaryResponse,
    SummaryWithQuestionsResponse,
    ProgressiveSummaryResponse
)
from .chat import (
    SearchSummaryResponse,
    QuestionBreakdownResponse,
    ChatResponse,
    QueryIntent,
    RelatedTopicsResponse,
)
from .base import (
    BaseResponse,
    TokenUsage,
    StreamChunk,
    LLMErrorResponse,
    UsageStatistics,
    LLMConfiguration
)
from .batch import (
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    PromptValidationResponse
)

__all__ = [
    # Analysis
    "PaperAnalysisResponse",
    "KeywordExtractionResponse",
    "PaperComparisonResponse",
    "ResearchGapsResponse",
    "MethodologyAnalysisResponse",
    "SentimentAnalysisResponse",
    
    # Reading
    "ExplanationResponse",
    "Question",
    "QuestionGenerationResponse",
    "StudyGuideResponse",
    "InteractiveReadingResponse",
    "MainIdeasResponse",
    "ConceptMapResponse",
    "ComprehensionTestResponse",
    
    # Summarization
    "SummaryResponse",
    "ExecutiveSummaryResponse",
    "SummaryWithQuestionsResponse",
    "ProgressiveSummaryResponse",
    
    # Chat
    "SearchSummaryResponse",
    "QuestionBreakdownResponse",
    "ChatResponse",
    "QueryIntent",
    "RelatedTopicsResponse",
    
    # Base
    "BaseResponse",
    "TokenUsage",
    "StreamChunk",
    "LLMErrorResponse",
    "UsageStatistics",
    "LLMConfiguration",
    
    # Batch
    "BatchAnalysisRequest",
    "BatchAnalysisResponse",
    "PromptValidationResponse",
]

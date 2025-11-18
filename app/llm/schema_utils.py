"""
Utility functions for working with LLM schemas and responses
"""
from typing import Dict, Any, List, Optional, Type, TypeVar, Union, cast
from datetime import datetime
from .schemas import (
    PaperAnalysisResponse,
    KeywordExtractionResponse,
    PaperComparisonResponse,
    ResearchGapsResponse,
    MethodologyAnalysisResponse,
    SentimentAnalysisResponse,
    ExplanationResponse,
    QuestionGenerationResponse,
    StudyGuideResponse,
    MainIdeasResponse,
    ConceptMapResponse,
    ComprehensionTestResponse,
    SummaryResponse,
    ExecutiveSummaryResponse,
    SummaryWithQuestionsResponse,
    ProgressiveSummaryResponse,
    SearchResultsSummaryResponse,
    PaperContentAnalysisResponse,
    ResearchQuestionsResponse,
)

T = TypeVar('T')


def dict_to_model(data: Dict[str, Any], model_class: Type[T]) -> T:
    """
    Convert dictionary response to Pydantic model
    
    Args:
        data: Dictionary data
        model_class: Pydantic model class to convert to
    
    Returns:
        Instance of the model class
    """
    # Add timestamp if not present
    if 'timestamp' not in data and hasattr(model_class, 'timestamp'):
        data['timestamp'] = datetime.now()
    
    return model_class(**data)


class ResponseConverter:
    """Helper class to convert dict responses to Pydantic models"""
    
    @staticmethod
    def to_paper_analysis(data: Dict[str, Any]) -> PaperAnalysisResponse:
        """Convert dict to PaperAnalysisResponse"""
        return dict_to_model(data, PaperAnalysisResponse)
    
    @staticmethod
    def to_keyword_extraction(keywords: List[str], **kwargs) -> KeywordExtractionResponse:
        """Convert keyword list to KeywordExtractionResponse"""
        return KeywordExtractionResponse(
            keywords=keywords,
            **kwargs
        )
    
    @staticmethod
    def to_paper_comparison(data: Dict[str, Any]) -> PaperComparisonResponse:
        """Convert dict to PaperComparisonResponse"""
        return dict_to_model(data, PaperComparisonResponse)
    
    @staticmethod
    def to_research_gaps(data: Dict[str, Any]) -> ResearchGapsResponse:
        """Convert dict to ResearchGapsResponse"""
        return dict_to_model(data, ResearchGapsResponse)
    
    @staticmethod
    def to_methodology_analysis(data: Dict[str, Any]) -> MethodologyAnalysisResponse:
        """Convert dict to MethodologyAnalysisResponse"""
        return dict_to_model(data, MethodologyAnalysisResponse)
    
    @staticmethod
    def to_sentiment_analysis(data: Dict[str, Any]) -> SentimentAnalysisResponse:
        """Convert dict to SentimentAnalysisResponse"""
        return dict_to_model(data, SentimentAnalysisResponse)
    
    @staticmethod
    def to_explanation(data: Dict[str, Any]) -> ExplanationResponse:
        """Convert dict to ExplanationResponse"""
        return dict_to_model(data, ExplanationResponse)
    
    @staticmethod
    def to_question_generation(data: Dict[str, Any]) -> QuestionGenerationResponse:
        """Convert dict to QuestionGenerationResponse"""
        return dict_to_model(data, QuestionGenerationResponse)
    
    @staticmethod
    def to_study_guide(data: Dict[str, Any]) -> StudyGuideResponse:
        """Convert dict to StudyGuideResponse"""
        return dict_to_model(data, StudyGuideResponse)
    
    @staticmethod
    def to_main_ideas(data: Dict[str, Any]) -> MainIdeasResponse:
        """Convert dict to MainIdeasResponse"""
        return dict_to_model(data, MainIdeasResponse)
    
    @staticmethod
    def to_concept_map(data: Dict[str, Any]) -> ConceptMapResponse:
        """Convert dict to ConceptMapResponse"""
        return dict_to_model(data, ConceptMapResponse)
    
    @staticmethod
    def to_comprehension_test(data: Dict[str, Any]) -> ComprehensionTestResponse:
        """Convert dict to ComprehensionTestResponse"""
        return dict_to_model(data, ComprehensionTestResponse)
    
    @staticmethod
    def to_summary(summary: str, style: str, max_length: int, 
                   model_used: str, **kwargs) -> SummaryResponse:
        """Convert summary data to SummaryResponse"""
        # Cast to proper Literal type
        valid_styles = ["concise", "detailed", "bullet-points", "executive", "academic", "narrative"]
        if style not in valid_styles:
            style = "concise"  # Default fallback
        
        return SummaryResponse(
            summary=summary,
            style=cast(Any, style),  # type: ignore
            max_length=max_length,
            model_used=model_used,
            **kwargs
        )
    
    @staticmethod
    def to_executive_summary(data: Dict[str, Any]) -> ExecutiveSummaryResponse:
        """Convert dict to ExecutiveSummaryResponse"""
        return dict_to_model(data, ExecutiveSummaryResponse)
    
    @staticmethod
    def to_summary_with_questions(data: Dict[str, Any]) -> SummaryWithQuestionsResponse:
        """Convert dict to SummaryWithQuestionsResponse"""
        return dict_to_model(data, SummaryWithQuestionsResponse)
    
    @staticmethod
    def to_progressive_summary(data: Dict[str, Any]) -> ProgressiveSummaryResponse:
        """Convert dict to ProgressiveSummaryResponse"""
        return dict_to_model(data, ProgressiveSummaryResponse)
    
    @staticmethod
    def to_search_summary(data: Dict[str, Any]) -> SearchResultsSummaryResponse:
        """Convert dict to SearchResultsSummaryResponse"""
        return dict_to_model(data, SearchResultsSummaryResponse)
    
    @staticmethod
    def to_paper_content_analysis(data: Dict[str, Any]) -> PaperContentAnalysisResponse:
        """Convert dict to PaperContentAnalysisResponse"""
        return dict_to_model(data, PaperContentAnalysisResponse)
    
    @staticmethod
    def to_research_questions(questions: List[str], topic: str, 
                              model_used: str) -> ResearchQuestionsResponse:
        """Convert research questions to ResearchQuestionsResponse"""
        return ResearchQuestionsResponse(
            questions=questions,
            topic=topic,
            num_questions=len(questions),
            model_used=model_used
        )


def validate_response(response: Any, expected_fields: Optional[List[str]] = None) -> bool:
    """
    Validate that a response has expected fields
    
    Args:
        response: Response to validate
        expected_fields: List of expected field names
    
    Returns:
        True if valid, False otherwise
    """
    if expected_fields is None:
        return True
    
    if isinstance(response, dict):
        return all(field in response for field in expected_fields)
    
    # For Pydantic models
    if hasattr(response, 'model_fields'):
        response_fields = set(response.model_fields.keys())
        return all(field in response_fields for field in expected_fields)
    
    return False


def response_to_dict(response: Any) -> Dict[str, Any]:
    """
    Convert Pydantic response to dictionary
    
    Args:
        response: Pydantic model instance
    
    Returns:
        Dictionary representation
    """
    if hasattr(response, 'model_dump'):
        return response.model_dump()
    elif hasattr(response, 'dict'):
        return response.dict()
    elif isinstance(response, dict):
        return response
    else:
        raise ValueError(f"Cannot convert {type(response)} to dict")


def merge_responses(*responses: Any) -> Dict[str, Any]:
    """
    Merge multiple responses into a single dictionary
    
    Args:
        *responses: Variable number of responses to merge
    
    Returns:
        Merged dictionary
    """
    merged = {}
    for response in responses:
        response_dict = response_to_dict(response)
        merged.update(response_dict)
    return merged


class ResponseBuilder:
    """Builder pattern for constructing complex responses"""
    
    def __init__(self, base_model_type: Type[Any]):
        self.model_type = base_model_type
        self.data: Dict[str, Any] = {}
    
    def add(self, key: str, value: Any) -> 'ResponseBuilder':
        """Add a field to the response"""
        self.data[key] = value
        return self
    
    def add_multiple(self, **kwargs) -> 'ResponseBuilder':
        """Add multiple fields at once"""
        self.data.update(kwargs)
        return self
    
    def build(self) -> Any:
        """Build the final response model"""
        return self.model_type(**self.data)


# Example usage helpers
def create_analysis_response(
    analysis: str,
    analysis_type: str,
    model_used: str,
    paper_title: Optional[str] = None
) -> PaperAnalysisResponse:
    """Helper to create PaperAnalysisResponse"""
    # Validate analysis_type
    valid_types = ["comprehensive", "methodology", "results", "literature_review"]
    if analysis_type not in valid_types:
        analysis_type = "comprehensive"  # Default fallback
    
    return PaperAnalysisResponse(
        analysis=analysis,
        analysis_type=cast(Any, analysis_type),  # type: ignore
        model_used=model_used,
        paper_title=paper_title
    )


def create_summary_response(
    summary: str,
    style: str,
    max_length: int,
    model_used: str,
    original_length: Optional[int] = None
) -> SummaryResponse:
    """Helper to create SummaryResponse"""
    # Validate style
    valid_styles = ["concise", "detailed", "bullet-points", "executive", "academic", "narrative"]
    if style not in valid_styles:
        style = "concise"  # Default fallback
    
    response_data: Dict[str, Any] = {
        "summary": summary,
        "style": cast(Any, style),  # type: ignore
        "max_length": max_length,
        "model_used": model_used
    }
    
    if original_length:
        response_data["original_length"] = original_length
        response_data["compression_ratio"] = len(summary) / original_length
    
    return SummaryResponse(**response_data)

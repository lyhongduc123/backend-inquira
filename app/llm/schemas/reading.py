"""
Reading and comprehension response schemas
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from .base import BaseResponse


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

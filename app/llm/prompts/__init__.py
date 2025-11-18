"""
Centralized prompts for LLM services
"""
from .analysis_prompts import AnalysisPrompts
from .reading_prompts import ReadingPrompts
from .summary_prompts import SummaryPrompts
from .base_prompts import PromptTemplate

__all__ = [
    'AnalysisPrompts',
    'ReadingPrompts',
    'SummaryPrompts',
    'PromptTemplate'
]

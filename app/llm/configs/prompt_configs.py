"""
Centralized configuration for LLM prompt parameters
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class PromptConfig(BaseModel):
    """Configuration for LLM prompt parameters"""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    # presence_penalty: Optional[float] = None
    # frequency_penalty: Optional[float] = None
    reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high", "default"]] = None
    
    class Config:
        frozen = False 
        validate_assignment = True
    
    def __init__(self, **data):
        super().__init__(**data)
        # Post-init validation
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.top_p is not None and (self.top_p < 0.0 or self.top_p > 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")
        # if self.presence_penalty is not None and (self.presence_penalty < -2.0 or self.presence_penalty > 2.0):
        #     raise ValueError("presence_penalty must be between -2.0 and 2.0")
        # if self.frequency_penalty is not None and (self.frequency_penalty < -2.0 or self.frequency_penalty > 2.0):
        #     raise ValueError("frequency_penalty must be between -2.0 and 2.0")


class PromptPresets:
    """Predefined prompt configurations for different tasks"""
    
    # Factual, deterministic tasks (citation-based responses, question breakdown)
    FACTUAL = PromptConfig(
        temperature=0.3,
        max_tokens=4000,
        top_p=0.9,
        # presence_penalty=0.1,
        # frequency_penalty=0.1
    )
    
    # Creative, exploratory tasks (related topics, brainstorming)
    CREATIVE = PromptConfig(
        temperature=0.8,
        max_tokens=2000,
        top_p=0.95,
        # presence_penalty=0.3,
        # frequency_penalty=0.3
    )
    
    # Analytical, reasoning tasks (paper analysis, comparison)
    ANALYTICAL = PromptConfig(
        temperature=0.4,
        max_tokens=3000,
        top_p=0.9,
        reasoning_effort="medium"
    )
    
    # Summarization tasks
    SUMMARIZATION = PromptConfig(
        temperature=0.3,
        max_tokens=2000,
        top_p=0.9
    )
    
    # Reading comprehension tasks
    READING = PromptConfig(
        temperature=0.4,
        max_tokens=2500,
        top_p=0.9
    )
    
    # Default balanced configuration
    DEFAULT = PromptConfig(
        temperature=0.7,
        max_tokens=2000,
        top_p=0.95
    )
    
    @classmethod
    def get_preset(cls, preset_name: str) -> PromptConfig:
        """
        Get a preset by name
        
        Args:
            preset_name: Name of the preset (factual, creative, analytical, etc.)
            
        Returns:
            PromptConfig instance
        """
        preset_map = {
            "factual": cls.FACTUAL,
            "creative": cls.CREATIVE,
            "analytical": cls.ANALYTICAL,
            "summarization": cls.SUMMARIZATION,
            "reading": cls.READING,
            "default": cls.DEFAULT
        }
        return preset_map.get(preset_name.lower(), cls.DEFAULT)
    
    @classmethod
    def merge_with_overrides(
        cls, 
        preset: PromptConfig, 
        **overrides
    ) -> dict:
        """
        Merge preset with runtime overrides
        
        Args:
            preset: Base PromptConfig preset
            **overrides: Runtime parameter overrides
            
        Returns:
            Dictionary of merged parameters (excluding None values)
        """
        config = preset.model_dump(exclude_none=True)
        
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
        
        return config

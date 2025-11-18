"""
Base prompt template functionality
"""
from typing import Dict, Any, Optional


class PromptTemplate:
    """Base class for creating and managing prompt templates"""
    
    @staticmethod
    def format_prompt(template: str, **kwargs) -> str:
        """
        Format a prompt template with provided variables
        
        Args:
            template: The prompt template string
            **kwargs: Variables to substitute in template
        
        Returns:
            Formatted prompt string
        """
        return template.format(**kwargs)
    
    @staticmethod
    def create_system_message(role: str, task: str, **constraints) -> str:
        """
        Create a system message with role and task
        
        Args:
            role: The role of the AI (e.g., "research assistant", "teacher")
            task: The main task description
            **constraints: Additional constraints or requirements
        
        Returns:
            Formatted system message
        """
        message = f"You are a {role}. {task}"
        
        if constraints:
            constraint_text = "\n".join([f"{k}: {v}" for k, v in constraints.items()])
            message += f"\n\n{constraint_text}"
        
        return message
    
    @staticmethod
    def build_structured_response(sections: list) -> str:
        """
        Build a prompt requesting structured response
        
        Args:
            sections: List of section names/descriptions
        
        Returns:
            Structured response template
        """
        structure = "\n\n".join([f"## {section}" for section in sections])
        return f"\nStructure your response as:\n\n{structure}"

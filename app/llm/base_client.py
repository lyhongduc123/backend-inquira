"""
Abstract base class for LLM clients to ensure consistent interface
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator, Union


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients (OpenAI, Ollama, etc.)"""
    
    def __init__(self):
        self.default_model: str = ""
    
    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """
        Generate chat completion
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (defaults to instance default)
            temperature: Randomness in response (0.0 to 2.0)
            max_tokens: Maximum tokens in response (optional, for OpenAI)
            stream: Whether to stream the response
            **kwargs: Additional parameters
        
        Returns:
            Response from LLM API
        """
        raise NotImplementedError
    
    @abstractmethod
    def simple_prompt(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Simple prompt-response interaction
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            model: Model to use
            temperature: Response randomness
            max_tokens: Maximum tokens in response (optional, for OpenAI)
            **kwargs: Additional parameters
        
        Returns:
            Response text
        """
        raise NotImplementedError
    
    @abstractmethod
    def stream_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream chat completion
        
        Args:
            messages: List of message dictionaries
            model: Model to use
            temperature: Response randomness
            **kwargs: Additional parameters
        
        Yields:
            Text chunks from the response
        """
        raise NotImplementedError
        yield  # Make it a generator
    
    @abstractmethod
    def get_model(self) -> str:
        """Get current model name"""
        raise NotImplementedError


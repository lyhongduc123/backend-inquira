"""
Ollama LLM client for local model interactions
"""
from typing import List, Dict, Any, Optional, Generator, Union
import ollama

from .base_client import BaseLLMClient
from app.extensions.logger import create_logger

logger = create_logger(__name__)

class OllamaClient(BaseLLMClient):
    """Client for Ollama API interactions"""
    
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "llama3.2"):
        """
        Initialize Ollama client
        
        Args:
            base_url: Ollama server URL
            default_model: Default model to use
        """
        self.client = ollama.Client(host=base_url)
        self.default_model = default_model
        logger.info(f"Initialized Ollama client with model: {default_model}")
    
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
        Generate chat completion using Ollama API
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (defaults to instance default)
            temperature: Randomness in response (0.0 to 2.0)
            max_tokens: Maximum tokens in response (ignored for Ollama, for compatibility)
            stream: Whether to stream the response
            **kwargs: Additional parameters for Ollama API
        
        Returns:
            Response from Ollama API (dict if not streaming, generator if streaming)
        """
        model_name = model or self.default_model
        
        # Build options dict
        options = {
            'temperature': temperature,
        }
        
        # Add max_tokens if provided (Ollama uses num_predict)
        if max_tokens:
            options['num_predict'] = max_tokens
        
        # Add any additional kwargs to options
        for key, value in kwargs.items():
            if key not in ['messages', 'model', 'stream', 'max_tokens']:
                options[key] = value
        
        response = self.client.chat(
            model=model_name,
            messages=messages,
            stream=stream,
            options=options
        )
        
        return response
    
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
            max_tokens: Maximum tokens in response (ignored for Ollama, for compatibility)
            **kwargs: Additional parameters
        
        Returns:
            Response text
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        response = self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **kwargs
        )
        
        # Ollama returns dict with 'message' key containing 'content'
        return response['message']['content']  # type: ignore
    
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
        response_stream = self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            stream=True,
            **kwargs
        )
        
        # Ollama streaming returns chunks with 'message' containing 'content'
        for chunk in response_stream:  # type: ignore
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
    
    def get_model(self) -> str:
        """Get current model name"""
        return self.default_model

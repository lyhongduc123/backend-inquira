"""
Main LLM Provider - Factory for all LLM services
"""
from typing import Optional
from .base import BaseLLMClient, ModelType
from .summarizer import Summarizer
from .analyzer import Analyzer
from .reader import Reader

class LLMProvider:
    """
    Main LLM Provider that creates specialized services
    Factory pattern for accessing different LLM capabilities
    """
    
    def __init__(self, api_key: str, default_model: str = ModelType.GPT_4O_MINI.value):
        self.base_client = BaseLLMClient(api_key=api_key, default_model=default_model)
        
        # Initialize specialized services
        self._summarizer = None
        self._analyzer = None
        self._reader = None
    
    @property
    def summarizer(self) -> Summarizer:
        """Get summarizer service"""
        if self._summarizer is None:
            self._summarizer = Summarizer(self.base_client)
        return self._summarizer
    
    @property
    def analyzer(self) -> Analyzer:
        """Get analyzer service"""
        if self._analyzer is None:
            self._analyzer = Analyzer(self.base_client)
        return self._analyzer
    
    @property
    def reader(self) -> Reader:
        """Get reader service"""
        if self._reader is None:
            self._reader = Reader(self.base_client)
        return self._reader
    
    def get_base_client(self) -> BaseLLMClient:
        """Access the underlying base LLM client"""
        return self.base_client
    
    def get_model(self) -> str:
        """Get the default model used by the base client"""
        return self.base_client.default_model
    
    # Convenience methods for direct access to base client functionality
    def simple_prompt(self, prompt: str, system_message: Optional[str] = None, **kwargs):
        """Direct access to simple prompt functionality"""
        return self.base_client.simple_prompt(prompt, system_message, **kwargs)
    
    def chat_completion(self, messages, **kwargs):
        """Direct access to chat completion functionality"""
        return self.base_client.chat_completion(messages, **kwargs)
    
    def stream_completion(self, messages, **kwargs):
        """Direct access to streaming functionality"""
        return self.base_client.stream_completion(messages, **kwargs)
    
    def count_tokens(self, text: str) -> int:
        """Direct access to token counting"""
        return self.base_client.count_tokens(text)
    
    def validate_messages(self, messages) -> bool:
        """Direct access to message validation"""
        return self.base_client.validate_messages(messages)
    
    def get_available_models(self):
        """Direct access to available models"""
        return self.base_client.get_available_models()


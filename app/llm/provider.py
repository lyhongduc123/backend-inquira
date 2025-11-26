"""
Main LLM Provider - Factory for all LLM services
"""
from typing import Optional, Union
from .openai_client import OpenaiClient, ModelType
from .ollama_client import OllamaClient
from .summarizer import Summarizer
from .analyzer import Analyzer
from .reader import Reader
from app.core.config import settings

class LLMProvider:
    """
    Main LLM Provider that creates specialized services
    Factory pattern for accessing different LLM capabilities
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        default_model: str = ModelType.GPT_4O_MINI.value,
        provider: str = "openai"
    ):
        """
        Initialize LLM provider
        
        Args:
            api_key: API key for OpenAI (only needed if provider is 'openai')
            default_model: Default model to use
            provider: LLM provider to use ('openai' or 'ollama')
        """
        self.provider_type = provider
        
        if provider == "ollama":
            self.base_client = OllamaClient(
                base_url=settings.OLLAMA_BASE_URL,
                default_model=settings.OLLAMA_MODEL
            )  # type: ignore
        else:  # openai
            if not api_key:
                raise ValueError("api_key is required for OpenAI provider")
            self.base_client = OpenaiClient(api_key=api_key, default_model=default_model)  # type: ignore
        
        self._summarizer = None
        self._analyzer = None
        self._reader = None
    
    @property
    def summarizer(self) -> Summarizer:
        """Get summarizer service"""
        if self._summarizer is None:
            self._summarizer = Summarizer(self.base_client)  # type: ignore
        return self._summarizer
    
    @property
    def analyzer(self) -> Analyzer:
        """Get analyzer service"""
        if self._analyzer is None:
            self._analyzer = Analyzer(self.base_client)  # type: ignore
        return self._analyzer
    
    @property
    def reader(self) -> Reader:
        """Get reader service"""
        if self._reader is None:
            self._reader = Reader(self.base_client)  # type: ignore
        return self._reader
    
    def get_base_client(self) -> Union[OpenaiClient, OllamaClient]:
        """Access the underlying base LLM client"""
        return self.base_client
    
    def get_model(self) -> str:
        """Get the default model used by the base client"""
        return self.base_client.default_model  # type: ignore
    
    # Convenience methods for direct access to base client functionality
    def simple_prompt(self, prompt: str, system_message: Optional[str] = None, **kwargs):
        """Direct access to simple prompt functionality"""
        return self.base_client.simple_prompt(prompt, system_message, **kwargs)  # type: ignore
    
    def chat_completion(self, messages, **kwargs):
        """Direct access to chat completion functionality"""
        return self.base_client.chat_completion(messages, **kwargs)  # type: ignore
    
    def stream_completion(self, messages, **kwargs):
        """Direct access to streaming functionality"""
        return self.base_client.stream_completion(messages, **kwargs)  # type: ignore
    
    def count_tokens(self, text: str) -> int:
        """Direct access to token counting"""
        if hasattr(self.base_client, 'count_tokens'):
            return self.base_client.count_tokens(text)  # type: ignore
        # Fallback for Ollama (rough estimate)
        return len(text.split()) * 1.3  # type: ignore
    
    def validate_messages(self, messages) -> bool:
        """Direct access to message validation"""
        if hasattr(self.base_client, 'validate_messages'):
            return self.base_client.validate_messages(messages)  # type: ignore
        # Simple validation for Ollama
        return isinstance(messages, list) and all(
            isinstance(m, dict) and 'role' in m and 'content' in m 
            for m in messages
        )
    
    def get_available_models(self):
        """Direct access to available models"""
        if hasattr(self.base_client, 'get_available_models'):
            return self.base_client.get_available_models()  # type: ignore
        # Ollama doesn't have this method
        return [self.base_client.default_model]  # type: ignore
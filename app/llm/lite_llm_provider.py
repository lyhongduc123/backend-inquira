from litellm import CustomStreamWrapper, completion
from litellm.files.main import ModelResponse
from typing import Dict, List, Union, Generator, Any, Optional
from .summarizer import Summarizer
from .analyzer import Analyzer
from .reader import Reader


class LiteLLMProvider:
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs
        
        self._summarizer = None
        self._analyzer = None
        self._reader = None
    
    def simple_prompt(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, **kwargs):
        """Simple completion without streaming

        Args:
            messages: List of messages for the conversation
            tools: Optional list of tools in OpenAI format
            **kwargs: Additional arguments for completion

        Returns:
            ModelResponse: completion response
        """
        params = {**self.kwargs, **kwargs}
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        
        return completion(self.model, messages=messages, **params)
        
    def stream_completion(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Generator[Union[tuple[str, Any], Any], Any, Any]:
        """Completion with stream enabled

        Args:
            messages: List of messages for the conversation
            tools: Optional list of tools in OpenAI format
            **kwargs: Additional arguments for completion

        Yields:
            tuple[str, Any] | ModelResponseStream: streamed response chunks
        """
        params = {**self.kwargs, **kwargs}
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        
        for chunk in completion(self.model, messages=messages, stream=True, **params):
            yield chunk
    
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
    
    def get_model(self) -> str:
        """Get the default model used by the base client"""
        return self.model
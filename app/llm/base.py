"""
Base LLM client for shared functionality
"""
from typing import List, Dict, Any, Optional, Generator, Union, Iterable, cast
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available OpenAI model types"""
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_3_5_TURBO = "gpt-3.5-turbo"

class BaseLLMClient:
    """Base client for OpenAI API interactions"""
    
    def __init__(self, api_key: str, default_model: str = ModelType.GPT_4O_MINI.value):
        self.client = OpenAI(api_key=api_key)
        self.default_model = default_model
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], Generator[ChatCompletionChunk, None, None]]:
        """
        Generate chat completion using OpenAI API
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (defaults to instance default)
            temperature: Randomness in response (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            stream: Whether to stream the response
            **kwargs: Additional parameters for OpenAI API
        
        Returns:
            Response from OpenAI API
        """
        try:
            response = self.client.chat.completions.create(
                model=model or self.default_model,
                messages=cast(Iterable[ChatCompletionMessageParam], messages),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )
            
            if stream:
                return cast(Generator[ChatCompletionChunk, None, None], response)
            
            completion = cast(ChatCompletion, response)
            return {
                "content": completion.choices[0].message.content,
                "model": completion.model,
                "usage": completion.usage,
                "finish_reason": completion.choices[0].finish_reason
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def simple_prompt(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Simple prompt completion with optional system message
        
        Args:
            prompt: User prompt
            system_message: Optional system message to set context
            model: Model to use
            temperature: Randomness in response
            max_tokens: Maximum tokens in response
        
        Returns:
            Generated text response
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if isinstance(response, dict):
            return response["content"]
        else:
            # Handle streaming case (shouldn't happen in simple_prompt)
            raise ValueError("Unexpected streaming response in simple_prompt")
    
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
            temperature: Randomness in response
            **kwargs: Additional parameters
        
        Yields:
            Streamed text chunks
        """
        print(f"[DEBUG] stream_completion called with model={model or self.default_model}")
        stream = self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            stream=True,
            **kwargs
        )
        
        print(f"[DEBUG] Got stream object, type: {type(stream)}")
        
        # OpenAI returns a Stream object, not a Generator
        chunk_num = 0
        try:
            for chunk in stream:
                chunk_num += 1
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0: # type: ignore
                    delta_content = chunk.choices[0].delta.content # type: ignore
                    if delta_content is not None:
                        if chunk_num <= 3:  # Only log first few chunks
                            print(f"[DEBUG] Yielding chunk {chunk_num}: {delta_content[:50] if len(delta_content) > 50 else delta_content}")
                        yield delta_content
            print(f"[DEBUG] Stream completed. Total chunks with content: {chunk_num}")
        except Exception as e:
            print(f"[ERROR] Error during streaming: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation)
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Estimated token count
        """
        # Rough approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """
        Validate message format
        
        Args:
            messages: List of message dictionaries
        
        Returns:
            True if valid, False otherwise
        """
        valid_roles = ["system", "user", "assistant"]
        
        for message in messages:
            if not isinstance(message, dict):
                return False
            if "role" not in message or "content" not in message:
                return False
            if message["role"] not in valid_roles:
                return False
        
        return True
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models
        
        Returns:
            List of available model names
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data if 'gpt' in model.id.lower()]
        except Exception as e:
            logger.warning(f"Failed to fetch models from API: {str(e)}")
            # Return default models if API call fails
            return [model.value for model in ModelType]
    
    def create_message(self, role: str, content: str) -> Dict[str, str]:
        """
        Helper to create a message dict
        
        Args:
            role: Message role (system, user, assistant)
            content: Message content
        
        Returns:
            Message dictionary
        """
        return {"role": role, "content": content}
    
    def build_conversation(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        Build a conversation message list
        
        Args:
            user_message: Current user message
            system_message: Optional system message
            history: Optional conversation history
        
        Returns:
            Complete message list
        """
        messages = []
        
        if system_message:
            messages.append(self.create_message("system", system_message))
        
        if history:
            messages.extend(history)
        
        messages.append(self.create_message("user", user_message))
        
        return messages

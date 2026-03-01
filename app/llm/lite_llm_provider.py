from aiohttp import ClientSession
import httpx
import litellm
from litellm.exceptions import (
    NotFoundError,
    RateLimitError,
    APIError,
    Timeout,
    ServiceUnavailableError,
)
from app.core.config import settings

from litellm.files.main import ModelResponse
from typing import Generator, List, Dict, Any, Optional, Union, Literal, Type
from typing_extensions import TypedDict
from pydantic import BaseModel
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class CompletionParams(TypedDict, total=False):
    model: str
    timeout: Optional[Union[float, str, "httpx.Timeout"]]
    temperature: Optional[float]
    top_p: Optional[float]
    n: Optional[int]
    stream: Optional[bool]
    stop: Optional[Any]
    max_completion_tokens: Optional[int]
    max_tokens: Optional[int]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    user: Optional[str]
    reasoning_effort: Optional[
        Literal["none", "minimal", "low", "medium", "high", "default"]
    ]
    verbosity: Optional[Literal["low", "medium", "high"]]
    response_format: Optional[Union[dict, Type[BaseModel]]]
    tools: Optional[List]
    tool_choice: Optional[Union[str, dict]]
    shared_session: Optional["ClientSession"]


class LiteLLMProvider:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.providers = [
            {
                "model": provider,
                "api_key": self._get_api_key_for_model(provider),
            }
            for provider in settings.LLM_MODEL
        ]
        self.current_provider_index = 0

    def _get_api_key_for_model(self, model_name: str) -> str:
        """Determine API key based on model name prefix"""
        if model_name.startswith("openrouter/"):
            return settings.OPENROUTER_API_KEY
        elif model_name.startswith("gemini/"):
            return settings.GEMINI_API_KEY
        elif model_name.startswith("mistral/"):
            return settings.MISTRALAI_API_KEY
        else:
            return settings.OPENAI_API_KEY

    def simple_prompt(self, messages: List[Dict[str, Any]], **kwargs: CompletionParams):
        """Get a simple completion response from the LLM provider
        
        Args:
            messages (List[Dict[str, Any]]): List of messages for the LLM prompt
            
        Raises:
            Exception: If all providers fail, raises an exception with the last error encountered
            
        Returns:
            The response from the LLM provider
        """
        total_providers = len(self.providers)
        attempts = 0
        last_error = None

        while attempts < total_providers:
            provider = self.providers[self.current_provider_index]

            try:
                params = {**self.kwargs, **kwargs}

                response = litellm.completion(
                    model=provider["model"],
                    api_key=provider["api_key"],
                    messages=messages,
                    drop_params=True,
                    **params,
                )

                return response  # ✅ success → stay on this provider

            except (RateLimitError, APIError, Timeout, ServiceUnavailableError) as e:
                logger.error(
                    f"Error with provider {provider['model']}: {e}. Switching..."
                )

                last_error = e
                self.current_provider_index = (
                    self.current_provider_index + 1
                ) % total_providers
                attempts += 1

        raise Exception(f"All LLM providers failed. Last error: {last_error}")

    def stream_completion(
        self, messages: List[Dict[str, Any]], **kwargs: CompletionParams
    ) -> Generator[Any, None, None]:
        """Stream completion

        Args:
            messages (List[Dict[str, Any]]): List of messages for the LLM prompt

        Raises:
            Exception: If all providers fail, raises an exception with the last error encountered

        Yields:
            Generator([Any, None, None]): The streaming response chunks from the LLM
        """
        total_providers = len(self.providers)
        attempts = 0
        last_error = None

        while attempts < total_providers:
            provider = self.providers[self.current_provider_index]
            params = {**self.kwargs, **kwargs}

            try:
                for chunk in litellm.completion(
                    model=provider["model"],
                    api_key=provider["api_key"],
                    messages=messages,
                    stream=True,
                    drop_params=True,
                    **params,
                ):
                    yield chunk

                return  # ✅ success → stop here

            except (RateLimitError, APIError, Timeout, ServiceUnavailableError) as e:
                logger.error(
                    f"Error with provider {provider['model']}: {e}. Switching..."
                )

                last_error = e
                self.current_provider_index = (
                    self.current_provider_index + 1
                ) % total_providers
                attempts += 1

        raise Exception(f"All LLM providers failed. Last error: {last_error}")

    def get_model(self) -> str:
        """Get the default model used by the base client"""
        return self.providers[0]["model"] if self.providers else "default"

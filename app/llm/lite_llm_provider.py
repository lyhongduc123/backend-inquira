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

from litellm.files.main import ModelResponse
from typing import Generator, List, Dict, Any, Optional, Union, Literal, Type
from typing_extensions import TypedDict
from pydantic import BaseModel
from .summarizer import Summarizer
from .analyzer import Analyzer
from .reader import Reader
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
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs
        print(f"Kwargs: {self.kwargs}")

        self._summarizer = None
        self._analyzer = None
        self._reader = None

    def simple_prompt(self, messages: List[Dict[str, Any]], **kwargs: CompletionParams):
        """Simple completion without streaming

        Args:
            messages: List of messages for the conversation
            tools: Optional list of tools in OpenAI format
            **kwargs: Additional arguments for completion

        Returns:
            ModelResponse: completion response

        Raises:
            Exception: If the LLM call fails after retries
        """
        params = {**self.kwargs, **kwargs}

        return litellm.completion(
            self.model, messages=messages, drop_params=True, **params
        )

    def stream_completion(
        self, messages: List[Dict[str, Any]], **kwargs: CompletionParams
    ) -> Generator[Union[tuple[str, Any], Any], Any, Any]:
        """Completion with stream enabled

        Args:
            messages: List of messages for the conversation
            tools: Optional list of tools in OpenAI format
            **kwargs: Additional arguments for completion

        Yields:
            tuple[str, Any] | ModelResponseStream: streamed response chunks

        Raises:
            Exception: If the LLM streaming call fails
        """
        params = {**self.kwargs, **kwargs}

        for chunk in litellm.completion(
            self.model, messages=messages, stream=True, drop_params=True, **params
        ):
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

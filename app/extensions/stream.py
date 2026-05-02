import json
from typing import Any, AsyncGenerator, Optional
import time
import random

from app.extensions.logger import create_logger

logger = create_logger(__name__)


class StreamEventType:
    """Event types for structured streaming"""

    CHUNK = "chunk"  # Each chunk as LLM generates
    PROGRESS = "progress"  # Progress updates
    METADATA = "metadata"  # Metadata updates (e.g., paper metadata)
    CONVERSATION = "conversation"  # Conversation events (e.g., user/assistant messages)
    DONE = "done"  # End of stream with summary
    ERROR = "error"  # Error occurred
    PING = "ping"  # Keepalive heartbeat


async def stream_event(name: str, data: Any):
    """Stream event as SSE

    Args:
        name (str): event name
        data (Any): data

    Yields:
        str: "event: {name}\\ndata: {data}\\n\\n"
    """
    if isinstance(data, str):
        serialized = data
    else:
        serialized = json.dumps(data, ensure_ascii=False, default=str)

    data_lines = serialized.splitlines() or [serialized]
    payload = "\n".join(f"data: {line}" for line in data_lines)
    yield f"event: {name}\n{payload}\n\n"


async def stream_heartbeat() -> AsyncGenerator[str, None]:
    """
    Stream a heartbeat/keepalive event to prevent connection timeout.

    Frontend receives:
    event: ping
    data: {}

    Use this during long-running operations to keep the connection alive.
    """
    async for evt in stream_event(name=StreamEventType.PING, data={}):
        yield evt


def stream_like_llm(text):
    i = 0
    while i < len(text):
        step = random.randint(1, 4) 
        yield text[i:i+step]
        i += step
        time.sleep(random.uniform(0.02, 0.12))

async def stream_chunk(content: str) -> AsyncGenerator[str, None]:
    """
    Stream a single chunk.

    Frontend receives:
    event: chunk
    data: {"type":"chunk","content":"Hello"}
    """
    async for evt in stream_event(
        name=StreamEventType.CHUNK,
        data=json.dumps({"type": StreamEventType.CHUNK, "content": content}),
    ):
        yield evt


def get_simple_response_content(response: Any) -> str:
    """
    Extract text content from non-streaming LLM response.

    Handles both dict responses and LiteLLM ModelResponse objects.

    Args:
        response: Non-streaming response from LLM

    Returns:
        Extracted text content (empty string if not found)
    """
    if response is None:
        return ""

    # Direct string
    if isinstance(response, str):
        return response

    # Dict with 'content' or 'text' field
    if isinstance(response, dict):
        return response.get("content", response.get("text", ""))

    # LiteLLM ModelResponse object
    if hasattr(response, "choices") and response.choices:
        message = response.choices[0].message
        if hasattr(message, "content") and message.content:
            return message.content

    return ""


def get_simple_response_reasoning(response: Any) -> Optional[str]:
    """
    Extract reasoning content from non-streaming LLM response.

    Works with reasoning-capable models like OpenAI o1, DeepSeek R1.

    Args:
        response: Non-streaming response from LLM

    Returns:
        Reasoning content or None if not present
    """
    if response is None:
        return None

    # LiteLLM ModelResponse object
    if hasattr(response, "choices") and response.choices:
        message = response.choices[0].message

        # Check for reasoning_content field (OpenAI o1)
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            return message.reasoning_content

    # Check provider_specific_fields for reasoning
    if hasattr(response, "provider_specific_fields"):
        reasoning = response.provider_specific_fields.get("reasoning")
        if reasoning:
            return reasoning

    # Dict format
    if isinstance(response, dict):
        return response.get("reasoning_content") or response.get("reasoning")

    return None


def get_stream_response_content(chunk: Any) -> Optional[str]:
    """
    Extract text content from streaming LLM response chunk.

    Handles both string chunks and structured chunk objects from LiteLLM.

    Args:
        chunk: Streaming chunk (can be str, dict, or ModelResponse object)

    Returns:
        Extracted text content or None if no content present
    """
    if chunk is None:
        return None

    # Direct string
    if isinstance(chunk, str):
        return chunk

    # Dict with 'text' or 'content' field
    if isinstance(chunk, dict):
        return chunk.get("text") or chunk.get("content")

    # LiteLLM ModelResponse object
    if hasattr(chunk, "choices") and chunk.choices:
        delta = chunk.choices[0].delta
        if hasattr(delta, "content"):
            return delta.content

    return None


def get_stream_response_reasoning(chunk: Any) -> Optional[str]:
    """
    Extract reasoning content from streaming LLM response chunk.

    Works with reasoning-capable models like OpenAI o1, DeepSeek R1.

    Args:
        chunk: Streaming chunk from LLM

    Returns:
        Reasoning content or None if not present
    """
    if chunk is None:
        return None

    # LiteLLM ModelResponse object
    if hasattr(chunk, "choices") and chunk.choices:
        delta = chunk.choices[0].delta

        # Check for reasoning_content field (OpenAI o1)
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            return delta.reasoning_content

        # Check provider_specific_fields for reasoning
        if hasattr(chunk, "provider_specific_fields"):
            if chunk.provider_specific_fields:
                reasoning = chunk.provider_specific_fields.get("reasoning")
                if reasoning:
                    return reasoning

    # Dict format
    if isinstance(chunk, dict):
        return chunk.get("reasoning_content") or chunk.get("reasoning")

    return None

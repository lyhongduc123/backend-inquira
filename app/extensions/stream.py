import json
from typing import Any, AsyncGenerator, List, Dict, Optional, Union, TYPE_CHECKING

from app.extensions.logger import create_logger

if TYPE_CHECKING:
    from app.papers.schemas import PaperMetadata

logger = create_logger(__name__)


class StreamEventType:
    """Event types for structured streaming"""

    CHUNK = "chunk"  # Each chunk as LLM generates
    PROGRESS = "progress"  # Progress updates
    METADATA = "metadata"  # Metadata updates (e.g., paper metadata)
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
    yield f"event: {name}\ndata: {data}\n\n"


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


# ========================================
# Structured Streaming Events
# ========================================


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

async def stream_done(
    cited_paper_ids: List[str],
    retrieved_paper_ids: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream completion event with summary.

    Frontend receives:
    event: done
    data: {
        "type":"done",
        "cited_paper_ids":["id1","id2"],
        "retrieved_paper_ids":["id1",...,"id20"],
        "metadata":{...}
    }

    Args:
        cited_paper_ids: Papers that were cited (extracted for analytics)
        retrieved_paper_ids: All papers retrieved (e.g., 20 papers)
        metadata: Additional metadata (token count, processing time, etc.)
    """
    done_data = {
        "type": StreamEventType.DONE,
        "cited_paper_ids": cited_paper_ids,
        "retrieved_paper_ids": retrieved_paper_ids,
        "cited_count": len(cited_paper_ids),
        "retrieved_count": len(retrieved_paper_ids),
    }

    if metadata:
        done_data["metadata"] = metadata

    async for evt in stream_event(
        name=StreamEventType.DONE, data=json.dumps(done_data)
    ):
        yield evt


async def stream_error(
    message: str,
    error_type: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream error event.

    Frontend receives:
    event: error
    data: {"type":"error","message":"...","error_type":"..."}

    Args:
        message: Error message
        error_type: Type of error (optional)
        details: Additional error details (optional)
    """
    error_data: Dict[str, Union[str, Dict[str, Any]]] = {
        "type": StreamEventType.ERROR,
        "message": message,
    }

    if error_type:
        error_data["error_type"] = error_type
    if details:
        error_data["details"] = details

    async for evt in stream_event(
        name=StreamEventType.ERROR, data=json.dumps(error_data)
    ):
        yield evt


# ========================================
# Response Content Extraction Helpers
# ========================================


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

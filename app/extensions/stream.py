import json
from typing import Any

from app.extensions.logger import create_logger

logger = create_logger(__name__)

async def stream_event(name: str, data: Any):
    """Stream event as SSE

    Args:
        name (str): event name
        data (Any): data

    Yields:
        str: "event: {name}\\ndata: {json.dumps(data)}\\n\\n"
    """
    if isinstance(data, dict) or isinstance(data, list):
        data = json.dumps(data, default=str)
    elif data is None:
        data = ""
    
    print("SSE outgoing: event=%s, data=%s", name, data)
    yield f"event: {name}\ndata: {data}\n\n"
    
def get_simple_response_content(response: Any) -> str:
    """Get simple completion response as string

    Args:
        prompt (str): user prompt
        system_message (str): system prompt (optional)

    Returns:
        str: completion text
    """
    return response.choices[0].message.content  # type: ignore

def get_stream_response_content(chunk: Any) -> str:
    """Get streamed completion chunk content

    Args:
        chunk (Any): streamed chunk

    Returns:
        str: chunk text
    """
    return chunk.choices[0].delta.get('content', '')  # type: ignore

def extract_delta(new: str, prev: str) -> str:
    if new.startswith(prev):
        return new[len(prev):]
    return new  # fallback when model rewinds or adds prefix

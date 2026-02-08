from typing import Any, Optional
 
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

def get_simple_response_reasoning(response: Any) -> Optional[str]:
    """Get reasoning content from simple response if available

    Args:
        response (Any): LiteLLM response
    Returns:
        Optional[str]: reasoning content or None
    """
    reasoning_content = None
    if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'message'):
                reasoning_content = getattr(choice.message, 'reasoning_content', None)
                
                if not reasoning_content and hasattr(choice, 'provider_specific_fields'):
                    provider_fields = choice.provider_specific_fields
                    if isinstance(provider_fields, dict):
                        reasoning_content = provider_fields.get('reasoning_content') or provider_fields.get('reasoning')
    return reasoning_content

def get_stream_response_reasoning(chunk: Any) -> Optional[str]:
    """Get reasoning content from streamed chunk if available

    Args:
        chunk (Any): streamed chunk
    Returns:
        Optional[str]: reasoning content or None
    """
    # Extract reasoning content if available in the chunk
    if hasattr(chunk, "choices") and len(chunk.choices) > 0:
        choice = chunk.choices[0]

        # Check for reasoning in delta
        if hasattr(choice, "delta") and isinstance(choice.delta, dict):
            reasoning = choice.delta.get("reasoning_content")
            if reasoning:
                return reasoning

        # Check for reasoning in provider_specific_fields
        if hasattr(choice, "provider_specific_fields"):
            provider_fields = choice.provider_specific_fields
            if isinstance(provider_fields, dict):
                reasoning = provider_fields.get(
                    "reasoning_content"
                ) or provider_fields.get("reasoning")
                if reasoning:
                    return reasoning

def extract_delta(new: str, prev: str) -> str:
    if new.startswith(prev):
        return new[len(prev):]
    return new  # fallback when model rewinds or adds prefix
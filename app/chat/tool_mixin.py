"""
Tool-aware chat service mixin for handling LLM tool calls
"""
import json
from typing import AsyncGenerator, List, Dict, Any, Optional
from app.llm.tools import ToolExecutor, ToolCall, tool_registry
from app.extensions.logger import create_logger
from app.extensions.stream import stream_event

logger = create_logger(__name__)


class ToolAwareChatMixin:
    """Mixin to add tool calling capabilities to chat service"""
    
    async def _handle_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        db_session,
        user_id: int
    ) -> AsyncGenerator[str, None]:
        """
        Handle tool calls from LLM
        
        Args:
            tool_calls: List of tool call dictionaries from LLM
            db_session: Database session to pass to tools
            user_id: User ID for context
            
        Yields:
            SSE events for tool execution
        """
        # Create tool executor with context
        executor = ToolExecutor(context={
            "db_session": db_session,
            "user_id": user_id
        })
        
        # Parse tool calls
        parsed_calls = []
        for tc in tool_calls:
            try:
                # Handle different formats from different providers
                if isinstance(tc, dict):
                    if 'function' in tc:
                        # OpenAI format
                        func = tc['function']
                        tool_call = ToolCall(
                            id=tc.get('id', f"call_{len(parsed_calls)}"),
                            name=func['name'],
                            arguments=json.loads(func['arguments']) if isinstance(func['arguments'], str) else func['arguments']
                        )
                    else:
                        # Direct format
                        tool_call = ToolCall(
                            id=tc.get('id', f"call_{len(parsed_calls)}"),
                            name=tc['name'],
                            arguments=tc.get('arguments', {})
                        )
                    parsed_calls.append(tool_call)
            except Exception as e:
                logger.error(f"Error parsing tool call: {e}", exc_info=True)
                continue
        
        if not parsed_calls:
            return
        
        # Emit tool call start events
        for tool_call in parsed_calls:
            async for evt in stream_event(
                name="tool_call",
                data={
                    "event_type": "tool_call_start",
                    "tool_name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "arguments": tool_call.arguments
                }
            ):
                yield evt
        
        # Execute tools
        results = await executor.execute_tools(parsed_calls)
        
        # Emit tool results
        for result in results:
            if result.success:
                async for evt in stream_event(
                    name="tool_call",
                    data={
                        "event_type": "tool_call_end",
                        "tool_name": result.name,
                        "tool_call_id": result.tool_call_id,
                        "result": result.result
                    }
                ):
                    yield evt
            else:
                async for evt in stream_event(
                    name="tool_call",
                    data={
                        "event_type": "tool_call_error",
                        "tool_name": result.name,
                        "tool_call_id": result.tool_call_id,
                        "error": result.error
                    }
                ):
                    yield evt
        
        return results
    
    async def stream_with_tools(
        self,
        messages: List[Dict[str, str]],
        db_session,
        user_id: int,
        max_tool_iterations: int = 3
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM response with tool calling support
        
        Args:
            messages: Conversation messages
            db_session: Database session
            user_id: User ID
            max_tool_iterations: Maximum number of tool call iterations
            
        Yields:
            SSE events for streaming response and tool calls
        """
        # Get available tools
        tools = tool_registry.get_tools_for_openai()
        
        # Track conversation with tool results
        conversation_messages = messages.copy()
        
        iteration = 0
        while iteration < max_tool_iterations:
            iteration += 1
            
            # Stream LLM response
            has_tool_calls = False
            tool_calls = []
            content_chunks = []
            
            # Use LLM provider with tools
            from app.llm import llm_service
            llm_provider = llm_service.llm_provider
            
            try:
                # Stream response
                for chunk in llm_provider.stream_completion(
                    messages=conversation_messages,
                    tools=tools if tools else None
                ):
                    # Handle different chunk formats
                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        
                        # Check for tool calls
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            has_tool_calls = True
                            for tc in delta.tool_calls:
                                # Accumulate tool call data
                                if hasattr(tc, 'id') and tc.id:
                                    tool_calls.append({
                                        'id': tc.id,
                                        'type': 'function',
                                        'function': {
                                            'name': tc.function.name if hasattr(tc.function, 'name') else '',
                                            'arguments': tc.function.arguments if hasattr(tc.function, 'arguments') else ''
                                        }
                                    })
                        
                        # Stream content
                        if hasattr(delta, 'content') and delta.content:
                            content_chunks.append(delta.content)
                            async for evt in stream_event(name="chunk", data={"text": delta.content}):
                                yield evt
                
                # If no tool calls, we're done
                if not has_tool_calls:
                    break
                
                # Handle tool calls
                logger.info(f"LLM requested {len(tool_calls)} tool calls")
                
                # Add assistant message with tool calls to conversation
                conversation_messages.append({
                    "role": "assistant",
                    "content": "".join(content_chunks) if content_chunks else None,
                    "tool_calls": tool_calls
                })
                
                # Execute tools and get results
                results = []
                async for evt in self._handle_tool_calls(tool_calls, db_session, user_id):
                    yield evt
                    # Extract results from events
                    if evt.startswith("event: tool_call"):
                        try:
                            data_line = evt.split('\n')[1]
                            if data_line.startswith("data: "):
                                result_data = json.loads(data_line[6:])
                                if result_data.get("event_type") == "tool_call_end":
                                    results.append(result_data)
                        except:
                            pass
                
                # Add tool results to conversation
                for result in results:
                    conversation_messages.append({
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "name": result["tool_name"],
                        "content": json.dumps(result["result"]) if not isinstance(result["result"], str) else result["result"]
                    })
                
                # Continue loop to get LLM response with tool results
                
            except Exception as e:
                logger.error(f"Error in tool-aware streaming: {e}", exc_info=True)
                async for evt in stream_event(name="error", data={"error": str(e)}):
                    yield evt
                break

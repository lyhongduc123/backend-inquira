"""
Tool executor for handling LLM tool calls
"""
from typing import List, Dict, Any, Optional
from app.llm.tools.schemas import ToolCall, ToolResult
from app.llm.tools.registry import tool_registry
from app.extensions.logger import create_logger
import json

logger = create_logger(__name__)


class ToolExecutor:
    """Executes tool calls from LLM"""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """
        Initialize tool executor
        
        Args:
            context: Additional context to pass to tool handlers (e.g., db_session, user_id)
        """
        self.context = context or {}
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a single tool call
        
        Args:
            tool_call: The tool call to execute
            
        Returns:
            Tool execution result
        """
        logger.info(f"Executing tool: {tool_call.name} with args: {tool_call.arguments}")
        
        # Get the tool handler
        handler = tool_registry.get_handler(tool_call.name)
        
        if not handler:
            error_msg = f"Tool '{tool_call.name}' not found in registry"
            logger.error(error_msg)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=error_msg,
                success=False
            )
        
        try:
            # Execute the handler with arguments and context
            if self.context:
                result = await handler(**tool_call.arguments, **self.context)
            else:
                result = await handler(**tool_call.arguments)
            
            logger.info(f"Tool {tool_call.name} executed successfully")
            
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=result,
                success=True
            )
        
        except Exception as e:
            error_msg = f"Error executing tool {tool_call.name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=error_msg,
                success=False
            )
    
    async def execute_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        Execute multiple tool calls
        
        Args:
            tool_calls: List of tool calls to execute
            
        Returns:
            List of tool results
        """
        results = []
        for tool_call in tool_calls:
            result = await self.execute_tool(tool_call)
            results.append(result)
        
        return results
    
    def format_tool_results_for_llm(self, results: List[ToolResult]) -> List[Dict[str, Any]]:
        """
        Format tool results for sending back to LLM
        
        Args:
            results: List of tool results
            
        Returns:
            List of formatted messages for LLM
        """
        messages = []
        
        for result in results:
            if result.success:
                content = json.dumps(result.result) if not isinstance(result.result, str) else result.result
            else:
                content = f"Error: {result.error}"
            
            messages.append({
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "name": result.name,
                "content": content
            })
        
        return messages

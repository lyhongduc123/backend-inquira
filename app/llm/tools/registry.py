"""
Tool registry for managing available tools
"""
from typing import Dict, Callable, List, Optional, Any
from app.llm.tools.schemas import Tool
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class ToolRegistry:
    """Registry for managing LLM tools"""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._handlers: Dict[str, Callable] = {}
    
    def register(
        self,
        tool: Tool,
        handler: Callable
    ):
        """
        Register a tool with its handler
        
        Args:
            tool: Tool definition
            handler: Function to execute the tool
        """
        if tool.name in self._tools:
            logger.warning(f"Tool {tool.name} already registered, overwriting")
        
        self._tools[tool.name] = tool
        self._handlers[tool.name] = handler
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self._tools.get(name)
    
    def get_handler(self, name: str) -> Optional[Callable]:
        """Get a tool handler by name"""
        return self._handlers.get(name)
    
    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools"""
        return list(self._tools.values())
    
    def get_tools_for_openai(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI format"""
        return [tool.to_openai_format() for tool in self._tools.values()]
    
    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered"""
        return name in self._tools
    
    def unregister(self, name: str):
        """Unregister a tool"""
        if name in self._tools:
            del self._tools[name]
            del self._handlers[name]
            logger.info(f"Unregistered tool: {name}")


# Global tool registry instance
tool_registry = ToolRegistry()

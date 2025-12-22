"""
Schemas for LLM tool calling system
"""
from typing import Dict, Any, List, Optional, Callable, Literal
from pydantic import BaseModel, Field
from enum import Enum


class ParameterType(str, Enum):
    """Parameter types for tool parameters"""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ToolParameter(BaseModel):
    """Schema for a tool parameter"""
    name: str = Field(..., description="Parameter name")
    type: ParameterType = Field(..., description="Parameter type")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=False, description="Whether parameter is required")
    enum: Optional[List[str]] = Field(None, description="Allowed values for enum types")
    items: Optional[Dict[str, Any]] = Field(None, description="Item schema for array types")
    properties: Optional[Dict[str, Any]] = Field(None, description="Properties for object types")


class Tool(BaseModel):
    """Schema for a tool definition"""
    name: str = Field(..., description="Tool name (snake_case)")
    description: str = Field(..., description="Tool description for LLM to understand when to use it")
    parameters: List[ToolParameter] = Field(default_factory=list, description="Tool parameters")
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        properties = {}
        required = []
        
        for param in self.parameters:
            param_schema = {
                "type": param.type.value,
                "description": param.description
            }
            
            if param.enum:
                param_schema["enum"] = param.enum
            if param.items:
                param_schema["items"] = param.items
            if param.properties:
                param_schema["properties"] = param.properties
                
            properties[param.name] = param_schema
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


class ToolCall(BaseModel):
    """Schema for a tool call from LLM"""
    id: str = Field(..., description="Tool call ID")
    name: str = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(..., description="Tool arguments")


class ToolResult(BaseModel):
    """Schema for a tool execution result"""
    tool_call_id: str = Field(..., description="ID of the tool call")
    name: str = Field(..., description="Tool name")
    result: Any = Field(..., description="Tool execution result")
    error: Optional[str] = Field(None, description="Error message if tool execution failed")
    success: bool = Field(True, description="Whether tool execution succeeded")


class ToolEvent(BaseModel):
    """Schema for tool execution events (for streaming)"""
    event_type: Literal["tool_call_start", "tool_call_end", "tool_call_error"] = Field(
        ..., description="Type of tool event"
    )
    tool_name: str = Field(..., description="Name of the tool")
    tool_call_id: str = Field(..., description="ID of the tool call")
    arguments: Optional[Dict[str, Any]] = Field(None, description="Tool arguments")
    result: Optional[Any] = Field(None, description="Tool result")
    error: Optional[str] = Field(None, description="Error message")
    timestamp: Optional[str] = Field(None, description="Event timestamp")


from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class S2RelevanceResponse(BaseModel):
    """Schema for Semantic Scholar API response."""

    total: int = Field(..., description="Total number of results")
    offset: int = Field(..., description="Offset of the current page")
    next: Optional[int] = Field(None, description="Offset for the next page")
    data: List[Dict[str, Any]] = Field(..., description="List of paper results")
    
class S2AuthorPapersResponse(BaseModel):
    """Schema for Semantic Scholar author papers response."""
    
    offset: int = Field(..., description="Offset of the current page")
    next: Optional[int] = Field(None, description="Offset for the next page")
    data: List[Dict[str, Any]] = Field(..., description="List of papers by the author")    
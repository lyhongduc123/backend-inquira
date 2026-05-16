"""
User settings schemas for API
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from app.core.model import CamelModel


class UserSettingsUpdate(CamelModel):
    """Request to update user settings"""
    language: Optional[str] = Field(None, min_length=2, max_length=10, description="Language code for AI responses (e.g., 'en', 'vi', 'fr')")
    preferences: Optional[Dict[str, Any]] = Field(None, description="Additional preferences")


class UserSettingsResponse(CamelModel):
    """User settings response"""
    id: int
    user_id: int
    language: str
    preferences: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

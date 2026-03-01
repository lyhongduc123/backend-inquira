"""
OAuth authentication schemas
"""
from pydantic import BaseModel, EmailStr
from app.core.model import CamelModel
from typing import Optional
from datetime import datetime


class Token(CamelModel):
    """Schema for JWT token response (refresh token sent as httpOnly cookie)"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until access token expires


class TokenData(BaseModel):
    """Schema for token payload data"""
    user_id: int | None = None
    email: str | None = None


class RefreshTokenRequest(CamelModel):
    """Schema for refresh token request (optional body, primary source is cookie)"""
    refresh_token: Optional[str] = None


class UserResponse(CamelModel):
    """Schema for user data in responses"""
    id: int
    email: str
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    provider: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class OAuthCallbackResponse(CamelModel):
    """Schema for OAuth callback response (refresh token sent as httpOnly cookie)"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

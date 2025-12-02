"""
OAuth authentication schemas
"""
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


class Token(BaseModel):
    """Schema for JWT token response with refresh token"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until access token expires


class TokenData(BaseModel):
    """Schema for token payload data"""
    user_id: int | None = None
    email: str | None = None


class RefreshTokenRequest(BaseModel):
    """Schema for refresh token request"""
    refresh_token: str


class UserResponse(BaseModel):
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


class OAuthCallbackResponse(BaseModel):
    """Schema for OAuth callback response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

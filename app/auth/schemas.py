"""
OAuth authentication schemas
"""
from enum import Enum
from pydantic import BaseModel, EmailStr, Field
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


class EmailOtpMode(str, Enum):
    """Supported OTP authentication intents."""
    LOGIN = "login"
    SIGNUP = "signup"


class EmailOtpRequest(CamelModel):
    """Request OTP for email login or signup."""
    email: EmailStr
    mode: EmailOtpMode = EmailOtpMode.LOGIN
    name: Optional[str] = None


class EmailOtpRequestResponse(CamelModel):
    """Response after requesting OTP code."""
    message: str
    expires_in: int
    resend_after: int


class EmailOtpVerifyRequest(CamelModel):
    """Verify OTP for email login or signup."""
    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6, pattern=r"^\d{6}$")
    mode: EmailOtpMode = EmailOtpMode.LOGIN
    name: Optional[str] = None


class EmailOtpVerifyResponse(CamelModel):
    """Token + user response for successful email OTP authentication."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class EmailOtpPreVerifyResponse(CamelModel):
    """Response after pre-validating OTP before signup profile completion."""
    message: str

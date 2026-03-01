"""
OAuth authentication router with secure httpOnly cookie support for refresh tokens
"""

import secrets
from fastapi import APIRouter, Depends, Query, status, Request, Response, Cookie
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta
from pydantic import BaseModel
from typing import Optional

from app.auth.schemas import (
    Token,
    RefreshTokenRequest,
    UserResponse,
    OAuthCallbackResponse,
)
from app.auth.oauth_service import (
    GoogleOAuthProvider,
    GitHubOAuthProvider,
    get_or_create_user,
    create_refresh_token,
    verify_refresh_token,
    revoke_refresh_token,
    revoke_all_user_tokens,
)
from app.auth.service import create_access_token, get_user_by_id
from app.auth.dependencies import get_current_user
from app.db.database import get_db_session
from app.models.users import DBUser
from app.core.config import settings
from app.core.exceptions import BadRequestException, UnauthorizedException, InternalServerException
from app.extensions.logger import create_logger

router = APIRouter()
logger = create_logger(__name__)

# Store OAuth states temporarily (in production, use Redis or similar)
oauth_states: dict[str, dict[str, str]] = {}

# Cookie configuration constants
REFRESH_TOKEN_COOKIE_NAME = "refresh_token"
REFRESH_TOKEN_MAX_AGE = settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60  # seconds


def set_refresh_token_cookie(response: Response, refresh_token: str) -> None:
    """
    Set refresh token as secure httpOnly cookie
    
    Security features:
    - httpOnly: Prevents JavaScript access (XSS protection)
    - secure: Only sent over HTTPS in production
    - samesite: CSRF protection
    - max_age: Auto-expires after refresh token lifetime
    """
    response.set_cookie(
        key=REFRESH_TOKEN_COOKIE_NAME,
        value=refresh_token,
        httponly=True,  # Cannot be accessed by JavaScript
        secure=settings.COOKIE_SECURE,  # HTTPS only in production
        samesite=settings.COOKIE_SAMESITE,  # CSRF protection
        max_age=REFRESH_TOKEN_MAX_AGE,
        domain=settings.COOKIE_DOMAIN,
        path="/",
    )


def clear_refresh_token_cookie(response: Response) -> None:
    """Clear refresh token cookie on logout"""
    response.delete_cookie(
        key=REFRESH_TOKEN_COOKIE_NAME,
        domain=settings.COOKIE_DOMAIN,
        path="/",
    )


def get_refresh_token_from_cookie_or_body(
    request: RefreshTokenRequest,
    refresh_token_cookie: Optional[str] = Cookie(None, alias=REFRESH_TOKEN_COOKIE_NAME)
) -> str:
    """
    Get refresh token from cookie (preferred) or request body (fallback)
    
    Priority:
    1. Cookie (secure, httpOnly)
    2. Request body (backward compatibility, less secure)
    """
    token = refresh_token_cookie or request.refresh_token
    if not token:
        raise UnauthorizedException("No refresh token provided")
    return token


@router.get("/google")
async def google_login():
    """
    Initiate Google OAuth flow

    Returns redirect URL to Google's OAuth consent screen
    """
    provider = GoogleOAuthProvider(
        client_id=settings.OAUTH_GOOGLE_CLIENT_ID,
        client_secret=settings.OAUTH_GOOGLE_CLIENT_SECRET,
        redirect_uri=settings.OAUTH_GOOGLE_REDIRECT_URI,
    )

    state = secrets.token_urlsafe(32)
    oauth_states[state] = {"provider": "google"}

    auth_url = provider.get_authorization_url(state)
    return RedirectResponse(url=auth_url)


@router.get("/google/callback")
async def google_callback(
    code: str | None = Query(None),
    state: str = Query(...),
    error: str | None = Query(None),
    db: AsyncSession = Depends(get_db_session),
) -> RedirectResponse:
    """
    Google OAuth callback endpoint

    Exchanges authorization code for user info and creates/updates user
    """
    # Verify state
    if state not in oauth_states:
        raise BadRequestException("Invalid state parameter")

    oauth_states.pop(state)

    if error:
        redirect_url = f"{settings.FRONTEND_URL}/auth/error?error={error}"
        return RedirectResponse(url=redirect_url)

    provider = GoogleOAuthProvider(
        client_id=settings.OAUTH_GOOGLE_CLIENT_ID,
        client_secret=settings.OAUTH_GOOGLE_CLIENT_SECRET,
        redirect_uri=settings.OAUTH_GOOGLE_REDIRECT_URI,
    )

    try:
        user_data = await provider.get_user_info(code)  # type: ignore
        user = await get_or_create_user(db, user_data)

        # Create tokens
        access_token = create_access_token(data={"sub": user.id, "email": user.email})
        refresh_token = await create_refresh_token(db, user.id)

        # Set refresh token as secure httpOnly cookie
        response = RedirectResponse(
            url=f"{settings.FRONTEND_URL}/auth/callback?access_token={access_token}"
        )
        set_refresh_token_cookie(response, refresh_token)
        return response

    except Exception as e:
        logger.error(f"Google OAuth authentication failed: {e}", exc_info=True)
        raise InternalServerException(f"OAuth authentication failed: {str(e)}")


@router.get("/github")
async def github_login():
    """
    Initiate GitHub OAuth flow

    Returns redirect URL to GitHub's OAuth consent screen
    """
    provider = GitHubOAuthProvider(
        client_id=settings.OAUTH_GITHUB_CLIENT_ID,
        client_secret=settings.OAUTH_GITHUB_CLIENT_SECRET,
        redirect_uri=settings.OAUTH_GITHUB_REDIRECT_URI,
    )

    state = secrets.token_urlsafe(32)
    oauth_states[state] = {"provider": "github"}

    auth_url = provider.get_authorization_url(state)
    return RedirectResponse(url=auth_url)


@router.get("/github/callback")
async def github_callback(
    code: str | None = Query(None),
    state: str = Query(...),
    error: str | None = Query(None),
    db: AsyncSession = Depends(get_db_session),
) -> RedirectResponse:
    """
    GitHub OAuth callback endpoint

    Exchanges authorization code for user info and creates/updates user
    """
    # Verify state
    if state not in oauth_states:
        raise BadRequestException("Invalid state parameter")

    oauth_states.pop(state)

    if error:
        redirect_url = f"{settings.FRONTEND_URL}/auth/error?error={error}"
        return RedirectResponse(url=redirect_url)

    provider = GitHubOAuthProvider(
        client_id=settings.OAUTH_GITHUB_CLIENT_ID,
        client_secret=settings.OAUTH_GITHUB_CLIENT_SECRET,
        redirect_uri=settings.OAUTH_GITHUB_REDIRECT_URI,
    )

    try:
        user_data = await provider.get_user_info(code)  # type: ignore
        user = await get_or_create_user(db, user_data)

        # Create tokens
        access_token = create_access_token(data={"sub": user.id, "email": user.email})
        refresh_token = await create_refresh_token(db, user.id)

        # Set refresh token as secure httpOnly cookie
        response = RedirectResponse(
            url=f"{settings.FRONTEND_URL}/auth/callback?access_token={access_token}"
        )
        set_refresh_token_cookie(response, refresh_token)
        return response

    except Exception as e:
        logger.error(f"GitHub OAuth authentication failed: {e}", exc_info=True)
        raise InternalServerException(f"OAuth authentication failed: {str(e)}")


@router.post("/refresh", response_model=Token)
async def refresh_access_token(
    http_request: Request,
    response: Response,
    request: RefreshTokenRequest = RefreshTokenRequest(),
    refresh_token_cookie: Optional[str] = Cookie(None, alias=REFRESH_TOKEN_COOKIE_NAME),
    db: AsyncSession = Depends(get_db_session)
) -> Token:
    """
    Refresh access token using refresh token from secure httpOnly cookie
    
    Security improvements:
    - Reads refresh token from httpOnly cookie (preferred) or body (fallback)
    - Rotates refresh token on each use (one-time use tokens)
    - Sets new refresh token as httpOnly cookie
    - Returns only access token in response body

    - **refresh_token**: Valid refresh token (cookie preferred, body for backward compatibility)

    Returns new access token; new refresh token set as cookie
    """
    # Get refresh token from cookie or body
    refresh_token = get_refresh_token_from_cookie_or_body(request, refresh_token_cookie)
    
    user_id = await verify_refresh_token(db, refresh_token)

    if not user_id:
        raise UnauthorizedException("Invalid or expired refresh token")

    user = await get_user_by_id(db, user_id)
    if not user or not user.is_active:
        raise UnauthorizedException("User not found or inactive")

    # Revoke old refresh token (token rotation for security)
    await revoke_refresh_token(db, refresh_token)

    # Create new tokens
    access_token = create_access_token(data={"sub": user.id, "email": user.email})
    new_refresh_token = await create_refresh_token(db, user.id)

    # Set new refresh token as httpOnly cookie
    set_refresh_token_cookie(response, new_refresh_token)

    token_response = Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
    
    return token_response


class LogoutResponse(BaseModel):
    """Response for logout operations"""
    message: str


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    http_request: Request,
    response: Response,
    request: RefreshTokenRequest = RefreshTokenRequest(),
    refresh_token_cookie: Optional[str] = Cookie(None, alias=REFRESH_TOKEN_COOKIE_NAME),
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user),
) -> LogoutResponse:
    """
    Logout user by revoking refresh token and clearing cookie

    Security improvements:
    - Reads refresh token from httpOnly cookie (preferred) or body (fallback)
    - Revokes refresh token in database
    - Clears httpOnly cookie

    - **refresh_token**: Refresh token to revoke (cookie preferred)

    Requires valid JWT token in Authorization header
    """
    # Get refresh token from cookie or body
    refresh_token = get_refresh_token_from_cookie_or_body(request, refresh_token_cookie)
    
    # Revoke refresh token
    await revoke_refresh_token(db, refresh_token)
    
    # Clear refresh token cookie
    clear_refresh_token_cookie(response)
    
    return LogoutResponse(message="Successfully logged out")


class LogoutAllResponse(BaseModel):
    """Response for logout all operations"""
    message: str
    devices_count: int


@router.post("/logout-all", response_model=LogoutAllResponse)
async def logout_all(
    http_request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user),
) -> LogoutAllResponse:
    """
    Logout from all devices by revoking all refresh tokens and clearing cookie

    Security improvements:
    - Revokes all refresh tokens for user
    - Clears httpOnly cookie

    Requires valid JWT token in Authorization header
    """
    count = await revoke_all_user_tokens(db, current_user.id)
    
    # Clear refresh token cookie
    clear_refresh_token_cookie(response)
    
    return LogoutAllResponse(
        message=f"Successfully logged out from {count} device(s)",
        devices_count=count
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    http_request: Request,
    current_user: DBUser = Depends(get_current_user)
) -> UserResponse:
    """
    Get current authenticated user's information

    Requires valid JWT token in Authorization header
    """
    user_response = UserResponse(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        avatar_url=current_user.avatar_url,
        provider=current_user.provider,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
    )
    
    return user_response

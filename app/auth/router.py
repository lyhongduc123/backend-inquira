"""
OAuth authentication router
"""

import secrets
from fastapi import APIRouter, Depends, Query, status, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta
from pydantic import BaseModel

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
from app.core.responses import ApiResponse, success_response
from app.core.exceptions import BadRequestException, UnauthorizedException, InternalServerException
from app.extensions.logger import create_logger

router = APIRouter()
logger = create_logger(__name__)

# Store OAuth states temporarily (in production, use Redis or similar)
oauth_states: dict[str, dict[str, str]] = {}


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

        # Redirect to frontend with tokens
        redirect_url = f"{settings.FRONTEND_URL}/auth/callback?access_token={access_token}&refresh_token={refresh_token}"
        return RedirectResponse(url=redirect_url)

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

        # Redirect to frontend with tokens
        redirect_url = f"{settings.FRONTEND_URL}/auth/callback?access_token={access_token}&refresh_token={refresh_token}"
        return RedirectResponse(url=redirect_url)

    except Exception as e:
        logger.error(f"GitHub OAuth authentication failed: {e}", exc_info=True)
        raise InternalServerException(f"OAuth authentication failed: {str(e)}")


@router.post("/refresh", response_model=ApiResponse[Token])
async def refresh_access_token(
    http_request: Request,
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db_session)
) -> ApiResponse[Token]:
    """
    Refresh access token using refresh token

    - **refresh_token**: Valid refresh token

    Returns new access token and refresh token
    """
    user_id = await verify_refresh_token(db, request.refresh_token)

    if not user_id:
        raise UnauthorizedException("Invalid or expired refresh token")

    user = await get_user_by_id(db, user_id)
    if not user or not user.is_active:
        raise UnauthorizedException("User not found or inactive")

    # Revoke old refresh token
    await revoke_refresh_token(db, request.refresh_token)

    # Create new tokens
    access_token = create_access_token(data={"sub": user.id, "email": user.email})
    new_refresh_token = await create_refresh_token(db, user.id)

    token_response = Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
    
    request_id = getattr(http_request.state, 'request_id', None)
    return success_response(token_response, request_id=request_id)


class LogoutResponse(BaseModel):
    """Response for logout operations"""
    message: str


@router.post("/logout", response_model=ApiResponse[LogoutResponse])
async def logout(
    http_request: Request,
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user),
) -> ApiResponse[LogoutResponse]:
    """
    Logout user by revoking refresh token

    - **refresh_token**: Refresh token to revoke

    Requires valid JWT token in Authorization header
    """
    await revoke_refresh_token(db, request.refresh_token)
    
    request_id = getattr(http_request.state, 'request_id', None)
    return success_response(
        LogoutResponse(message="Successfully logged out"),
        request_id=request_id
    )


class LogoutAllResponse(BaseModel):
    """Response for logout all operations"""
    message: str
    devices_count: int


@router.post("/logout-all", response_model=ApiResponse[LogoutAllResponse])
async def logout_all(
    http_request: Request,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user),
) -> ApiResponse[LogoutAllResponse]:
    """
    Logout from all devices by revoking all refresh tokens

    Requires valid JWT token in Authorization header
    """
    count = await revoke_all_user_tokens(db, current_user.id)
    
    request_id = getattr(http_request.state, 'request_id', None)
    return success_response(
        LogoutAllResponse(
            message=f"Successfully logged out from {count} device(s)",
            devices_count=count
        ),
        request_id=request_id
    )


@router.get("/me", response_model=ApiResponse[UserResponse])
async def get_current_user_info(
    http_request: Request,
    current_user: DBUser = Depends(get_current_user)
) -> ApiResponse[UserResponse]:
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
    
    request_id = getattr(http_request.state, 'request_id', None)
    return success_response(user_response, request_id=request_id)

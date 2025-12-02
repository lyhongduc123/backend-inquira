"""
OAuth authentication router
"""
import secrets
from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta

from app.auth.schemas import (
    Token,
    RefreshTokenRequest,
    UserResponse,
    OAuthCallbackResponse
)
from app.auth.oauth_service import (
    GoogleOAuthProvider,
    GitHubOAuthProvider,
    get_or_create_user,
    create_refresh_token,
    verify_refresh_token,
    revoke_refresh_token,
    revoke_all_user_tokens
)
from app.auth.service import create_access_token, get_user_by_id
from app.auth.dependencies import get_current_user
from app.db.database import get_db_session
from app.models.users import DBUser
from app.core.config import settings

router = APIRouter()

# Store OAuth states temporarily (in production, use Redis or similar)
oauth_states = {}


@router.get("/google")
async def google_login():
    """
    Initiate Google OAuth flow
    
    Returns redirect URL to Google's OAuth consent screen
    """
    provider = GoogleOAuthProvider(
        client_id=settings.GOOGLE_CLIENT_ID,
        client_secret=settings.GOOGLE_CLIENT_SECRET,
        redirect_uri=settings.GOOGLE_REDIRECT_URI
    )
    
    state = secrets.token_urlsafe(32)
    oauth_states[state] = {"provider": "google"}
    
    auth_url = provider.get_authorization_url(state)
    return {"authorization_url": auth_url}


@router.get("/google/callback")
async def google_callback(
    code: str = Query(...),
    state: str = Query(...),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Google OAuth callback endpoint
    
    Exchanges authorization code for user info and creates/updates user
    """
    # Verify state
    if state not in oauth_states:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid state parameter"
        )
    
    oauth_states.pop(state)
    
    provider = GoogleOAuthProvider(
        client_id=settings.GOOGLE_CLIENT_ID,
        client_secret=settings.GOOGLE_CLIENT_SECRET,
        redirect_uri=settings.GOOGLE_REDIRECT_URI
    )
    
    try:
        user_data = await provider.get_user_info(code)
        user = await get_or_create_user(db, user_data)
        
        # Create tokens
        access_token = create_access_token(
            data={"sub": user.id, "email": user.email}
        )
        refresh_token = await create_refresh_token(db, user.id)
        
        # Redirect to frontend with tokens
        redirect_url = f"{settings.FRONTEND_URL}/auth/callback?access_token={access_token}&refresh_token={refresh_token}"
        return RedirectResponse(url=redirect_url)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OAuth authentication failed: {str(e)}"
        )


@router.get("/github")
async def github_login():
    """
    Initiate GitHub OAuth flow
    
    Returns redirect URL to GitHub's OAuth consent screen
    """
    provider = GitHubOAuthProvider(
        client_id=settings.GITHUB_CLIENT_ID,
        client_secret=settings.GITHUB_CLIENT_SECRET,
        redirect_uri=settings.GITHUB_REDIRECT_URI
    )
    
    state = secrets.token_urlsafe(32)
    oauth_states[state] = {"provider": "github"}
    
    auth_url = provider.get_authorization_url(state)
    return {"authorization_url": auth_url}


@router.get("/github/callback")
async def github_callback(
    code: str = Query(...),
    state: str = Query(...),
    db: AsyncSession = Depends(get_db_session)
):
    """
    GitHub OAuth callback endpoint
    
    Exchanges authorization code for user info and creates/updates user
    """
    # Verify state
    if state not in oauth_states:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid state parameter"
        )
    
    oauth_states.pop(state)
    
    provider = GitHubOAuthProvider(
        client_id=settings.GITHUB_CLIENT_ID,
        client_secret=settings.GITHUB_CLIENT_SECRET,
        redirect_uri=settings.GITHUB_REDIRECT_URI
    )
    
    try:
        user_data = await provider.get_user_info(code)
        user = await get_or_create_user(db, user_data)
        
        # Create tokens
        access_token = create_access_token(
            data={"sub": user.id, "email": user.email}
        )
        refresh_token = await create_refresh_token(db, user.id)
        
        # Redirect to frontend with tokens
        redirect_url = f"{settings.FRONTEND_URL}/auth/callback?access_token={access_token}&refresh_token={refresh_token}"
        return RedirectResponse(url=redirect_url)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OAuth authentication failed: {str(e)}"
        )


@router.post("/refresh", response_model=Token)
async def refresh_access_token(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Refresh access token using refresh token
    
    - **refresh_token**: Valid refresh token
    
    Returns new access token and refresh token
    """
    user_id = await verify_refresh_token(db, request.refresh_token)
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    user = await get_user_by_id(db, user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Revoke old refresh token
    await revoke_refresh_token(db, request.refresh_token)
    
    # Create new tokens
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email}
    )
    new_refresh_token = await create_refresh_token(db, user.id)
    
    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/logout")
async def logout(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
):
    """
    Logout user by revoking refresh token
    
    - **refresh_token**: Refresh token to revoke
    
    Requires valid JWT token in Authorization header
    """
    await revoke_refresh_token(db, request.refresh_token)
    return {"message": "Successfully logged out"}


@router.post("/logout-all")
async def logout_all(
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
):
    """
    Logout from all devices by revoking all refresh tokens
    
    Requires valid JWT token in Authorization header
    """
    count = await revoke_all_user_tokens(db, current_user.id)
    return {"message": f"Successfully logged out from {count} device(s)"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: DBUser = Depends(get_current_user)
):
    """
    Get current authenticated user's information
    
    Requires valid JWT token in Authorization header
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        avatar_url=current_user.avatar_url,
        provider=current_user.provider,
        is_active=current_user.is_active,
        created_at=current_user.created_at
    )

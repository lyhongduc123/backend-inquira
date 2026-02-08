"""
OAuth service for Google and GitHub authentication
"""
import httpx
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import secrets

from app.models.users import DBUser
from app.models.refresh_tokens import DBRefreshToken
from app.core.config import settings
from app.auth.service import create_access_token
from app.extensions.logger import create_logger
from urllib.parse import urlencode

logger = create_logger(__name__)


class OAuthProvider:
    """Base OAuth provider class"""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
    
    async def get_user_info(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for user info"""
        raise NotImplementedError


class GoogleOAuthProvider(OAuthProvider):
    """Google OAuth provider"""
    
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
    
    def get_authorization_url(self, state: str) -> str:
        """Get Google OAuth authorization URL"""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "access_type": "offline",
            "prompt": "consent"
        }
        query = urlencode(params)
        return f"https://accounts.google.com/o/oauth2/v2/auth?{query}"
    
    async def get_user_info(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for user info from Google"""
        async with httpx.AsyncClient() as client:
            # Exchange code for access token
            token_response = await client.post(
                self.TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": self.redirect_uri
                }
            )
            token_response.raise_for_status()
            token_data = token_response.json()
            access_token = token_data["access_token"]
            
            # Get user info
            userinfo_response = await client.get(
                self.USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"}
            )
            userinfo_response.raise_for_status()
            user_info = userinfo_response.json()
            
            return {
                "provider_id": user_info["id"],
                "email": user_info["email"],
                "name": user_info.get("name"),
                "avatar_url": user_info.get("picture"),
                "provider": "google"
            }


class GitHubOAuthProvider(OAuthProvider):
    """GitHub OAuth provider"""
    
    TOKEN_URL = "https://github.com/login/oauth/access_token"
    USERINFO_URL = "https://api.github.com/user"
    EMAIL_URL = "https://api.github.com/user/emails"
    
    def get_authorization_url(self, state: str) -> str:
        """Get GitHub OAuth authorization URL"""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "read:user user:email",
            "state": state
        }
        query = urlencode(params)
        return f"https://github.com/login/oauth/authorize?{query}"
    
    async def get_user_info(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for user info from GitHub"""
        async with httpx.AsyncClient() as client:
            # Exchange code for access token
            token_response = await client.post(
                self.TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "redirect_uri": self.redirect_uri
                },
                headers={"Accept": "application/json"}
            )
            token_response.raise_for_status()
            token_data = token_response.json()
            access_token = token_data["access_token"]
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json"
            }
            
            # Get user info
            userinfo_response = await client.get(self.USERINFO_URL, headers=headers)
            userinfo_response.raise_for_status()
            user_info = userinfo_response.json()
            
            # Get primary email
            email_response = await client.get(self.EMAIL_URL, headers=headers)
            email_response.raise_for_status()
            emails = email_response.json()
            primary_email = next((e["email"] for e in emails if e["primary"]), emails[0]["email"] if emails else None)
            
            return {
                "provider_id": str(user_info["id"]),
                "email": primary_email or user_info.get("email"),
                "name": user_info.get("name") or user_info.get("login"),
                "avatar_url": user_info.get("avatar_url"),
                "provider": "github"
            }


async def get_or_create_user(db: AsyncSession, user_data: Dict[str, Any]) -> DBUser:
    """Get existing user or create new one from OAuth data"""
    # Try to find user by provider_id first
    result = await db.execute(
        select(DBUser).where(
            DBUser.provider == user_data["provider"],
            DBUser.provider_id == user_data["provider_id"]
        )
    )
    user = result.scalar_one_or_none()
    
    if user:
        # Update user info
        user.email = user_data["email"]
        user.name = user_data.get("name")
        user.avatar_url = user_data.get("avatar_url")
        user.updated_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(user)
        return user
    
    # Check if email exists with different provider
    result = await db.execute(
        select(DBUser).where(DBUser.email == user_data["email"])
    )
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        logger.warning(f"Email {user_data['email']} exists with different provider")
    
    # Create new user
    new_user = DBUser(
        email=user_data["email"],
        name=user_data.get("name"),
        avatar_url=user_data.get("avatar_url"),
        provider=user_data["provider"],
        provider_id=user_data["provider_id"],
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    return new_user


async def create_refresh_token(db: AsyncSession, user_id: int) -> str:
    """Create a new refresh token for a user"""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    
    db_token = DBRefreshToken(
        user_id=user_id,
        token=token,
        expires_at=expires_at,
        created_at=datetime.now(timezone.utc),
        revoked=False
    )
    
    db.add(db_token)
    await db.commit()
    
    return token


async def verify_refresh_token(db: AsyncSession, token: str) -> Optional[int]:
    """Verify refresh token and return user_id if valid"""
    result = await db.execute(
        select(DBRefreshToken).where(
            DBRefreshToken.token == token,
            DBRefreshToken.revoked == False
        )
    )
    db_token = result.scalar_one_or_none()
    
    if not db_token:
        return None
    
    # Ensure expires_at is timezone-aware
    if db_token.expires_at.tzinfo is None:
        db_token.expires_at = db_token.expires_at.replace(tzinfo=timezone.utc)
        
    # Check if token is expired
    if db_token.expires_at < datetime.now(timezone.utc):
        return None
    
    return db_token.user_id


async def revoke_refresh_token(db: AsyncSession, token: str) -> bool:
    """Revoke a refresh token"""
    result = await db.execute(
        select(DBRefreshToken).where(DBRefreshToken.token == token)
    )
    db_token = result.scalar_one_or_none()
    
    if not db_token:
        return False
    
    db_token.revoked = True
    await db.commit()
    
    return True


async def revoke_all_user_tokens(db: AsyncSession, user_id: int) -> int:
    """Revoke all refresh tokens for a user"""
    result = await db.execute(
        select(DBRefreshToken).where(
            DBRefreshToken.user_id == user_id,
            DBRefreshToken.revoked == False
        )
    )
    tokens = result.scalars().all()
    
    count = 0
    for token in tokens:
        token.revoked = True
        count += 1
    
    await db.commit()
    return count

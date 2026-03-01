"""
Authentication dependencies for FastAPI route protection
"""
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from app.auth.service import decode_access_token, get_user_by_id
from app.models.users import DBUser
from app.db.database import get_db_session
from app.core.exceptions import UnauthorizedException, ForbiddenException
from app.extensions.logger import create_logger

logger = create_logger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db_session)
) -> DBUser:
    """
    Dependency to get the current authenticated user from JWT token
    
    Args:
        credentials: HTTP Authorization header with Bearer token
        db: Database session
    
    Returns:
        DBUser instance of authenticated user
    
    Raises:
        UnauthorizedException: If token is invalid or user not found
        ForbiddenException: If user is inactive
    """
    token = credentials.credentials
    logger.debug(f"Validating token: {token[:20]}...")
    token_data = decode_access_token(token)
    logger.debug(f"Decoded token data for user_id: {token_data.user_id if token_data else None}")
    
    if token_data is None or token_data.user_id is None:
        raise UnauthorizedException("Could not validate credentials")
    
    user = await get_user_by_id(db, user_id=token_data.user_id)
    
    if user is None:
        raise UnauthorizedException("User not found")
    
    if not user.is_active:
        raise ForbiddenException("User account is inactive")
    
    return user


async def get_current_active_user(
    current_user: DBUser = Depends(get_current_user)
) -> DBUser:
    """
    Dependency to get current active user (additional validation layer)
    
    Args:
        current_user: Current user from get_current_user dependency
    
    Returns:
        DBUser instance if user is active
    
    Raises:
        ForbiddenException: If user is inactive
    """
    if not current_user.is_active:
        raise ForbiddenException("User account is inactive")
    return current_user


async def get_admin_user(
    current_user: DBUser = Depends(get_current_user)
) -> DBUser:
    """
    Dependency to ensure current user has admin privileges.
    
    TODO: Add is_admin field to DBUser model.
    For now, this checks is_active as a placeholder.
    
    Args:
        current_user: Current user from get_current_user dependency
    
    Returns:
        DBUser instance if user is admin
    
    Raises:
        ForbiddenException: If user is not admin
    """
    # TODO: Replace with actual admin check once is_admin field exists
    # if not getattr(current_user, 'is_admin', False):
    #     raise ForbiddenException("Admin privileges required")
    
    # Temporary: Allow all active users (replace this)
    if not current_user.is_active:
        raise ForbiddenException("Admin privileges required")
    
    logger.warning(f"Admin endpoint accessed by user {current_user.id} - TODO: Add proper admin role check")
    return current_user

async def get_fake_user() -> DBUser:
    return DBUser(
        id="dev-user",
        email="dev@example.com",
        is_active=True,
    )

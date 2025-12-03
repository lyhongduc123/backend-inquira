"""
JWT token management service
"""
from datetime import datetime, timedelta, timezone
from typing import Optional, Any
from jose import JWTError, jwt, ExpiredSignatureError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.users import DBUser
from app.core.config import settings
from app.auth.schemas import TokenData


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token
    
    Args:
        data: Payload data to encode in the token
        expires_delta: Optional custom expiration time
    
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": int(expire.timestamp())})
    
    to_encode_str = {k: str(v) for k, v in to_encode.items() if k != "exp"}
    to_encode_str["exp"] = to_encode["exp"]
    
    encoded_jwt = jwt.encode(to_encode_str, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[TokenData]:
    """
    Decode and validate a JWT access token
    
    Args:
        token: JWT token string
    
    Returns:
        TokenData if valid, None if invalid
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id_any: Any = payload.get("sub")
        email_any: Any = payload.get("email")
        
        # If the subject is missing, token is invalid
        if user_id_any is None:
            return None
        
        # Safely convert the subject to an integer (reject if not convertible)
        try:
            user_id: int = int(user_id_any)
        except (TypeError, ValueError):
            return None
        
        email: Optional[str] = str(email_any) if email_any is not None else None
        
        return TokenData(user_id=user_id, email=email)
    except ExpiredSignatureError:
        print("Token has expired")
        return None
    except JWTError as e:
        print("JWT decoding error:", e)
        return None


async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[DBUser]:
    """Get user by ID"""
    result = await db.execute(
        select(DBUser).where(DBUser.id == user_id)
    )
    return result.scalar_one_or_none()

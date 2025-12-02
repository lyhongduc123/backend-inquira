"""
Authentication module
"""
from app.auth.router import router
from app.auth.dependencies import get_current_user, get_current_active_user

__all__ = ["router", "get_current_user", "get_current_active_user"]

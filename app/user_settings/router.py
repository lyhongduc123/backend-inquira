"""
User settings router for API endpoints
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import TYPE_CHECKING
from app.db.database import get_db_session
from app.auth.dependencies import get_current_user
from app.models.users import DBUser
from app.core.dependencies import get_container
from app.user_settings.service import UserSettingsService
from app.user_settings.schemas import UserSettingsUpdate, UserSettingsResponse

if TYPE_CHECKING:
    from app.core.container import ServiceContainer


router = APIRouter()


@router.get("/", response_model=UserSettingsResponse)
async def get_user_settings(
    current_user: DBUser = Depends(get_current_user),
    container: "ServiceContainer" = Depends(get_container)
) -> UserSettingsResponse:
    """Get current user settings"""
    settings = await container.user_settings_service.get_settings(current_user.id)
    return settings


@router.patch("/", response_model=UserSettingsResponse)
async def update_user_settings(
    request: UserSettingsUpdate,
    current_user: DBUser = Depends(get_current_user),
    container: "ServiceContainer" = Depends(get_container)
) -> UserSettingsResponse:
    """Update user settings"""
    settings = await container.user_settings_service.update_settings(
        user_id=current_user.id,
        language=request.language,
        preferences=request.preferences
    )
    return settings

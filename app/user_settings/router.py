"""
User settings router for API endpoints
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db_session
from app.auth.dependencies import get_current_user
from app.models.users import DBUser
from app.user_settings.service import UserSettingsService
from app.user_settings.schemas import UserSettingsUpdate, UserSettingsResponse


router = APIRouter()


@router.get("/", response_model=UserSettingsResponse)
async def get_user_settings(
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> UserSettingsResponse:
    """Get current user settings"""
    service = UserSettingsService(db)
    settings = await service.get_settings(current_user.id)
    return settings


@router.patch("/", response_model=UserSettingsResponse)
async def update_user_settings(
    request: UserSettingsUpdate,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> UserSettingsResponse:
    """Update user settings"""
    service = UserSettingsService(db)
    settings = await service.update_settings(
        user_id=current_user.id,
        language=request.language,
        preferences=request.preferences
    )
    return settings

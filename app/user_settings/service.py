"""
User settings service for business logic
"""
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.user_settings.repository import UserSettingsRepository
from app.user_settings.schemas import UserSettingsResponse


class UserSettingsService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = UserSettingsRepository(db)

    async def get_settings(self, user_id: int) -> UserSettingsResponse:
        """Get user settings (creates with defaults if doesn't exist)"""
        settings = await self.repo.get_or_create(user_id)
        return UserSettingsResponse(
            id=settings.id,
            user_id=settings.user_id,
            language=settings.language,
            preferences=settings.preferences,
            created_at=settings.created_at,
            updated_at=settings.updated_at
        )

    async def update_settings(
        self, 
        user_id: int, 
        language: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> UserSettingsResponse:
        """Update user settings"""
        settings = await self.repo.update(user_id, language, preferences)
        return UserSettingsResponse(
            id=settings.id,
            user_id=settings.user_id,
            language=settings.language,
            preferences=settings.preferences,
            created_at=settings.created_at,
            updated_at=settings.updated_at
        )

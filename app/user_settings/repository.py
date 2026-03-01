"""
User settings repository for database operations
"""
from typing import Optional, Dict, Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user_settings import DBUserSettings


class UserSettingsRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_user(self, user_id: int) -> Optional[DBUserSettings]:
        """Get settings for a specific user"""
        result = await self.db.execute(
            select(DBUserSettings).where(DBUserSettings.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def create(
        self, 
        user_id: int, 
        language: str = "en", 
        preferences: Optional[Dict[str, Any]] = None
    ) -> DBUserSettings:
        """Create user settings"""
        settings = DBUserSettings(
            user_id=user_id,
            language=language,
            preferences=preferences or {}
        )
        self.db.add(settings)
        await self.db.commit()
        await self.db.refresh(settings)
        return settings

    async def update(
        self,
        user_id: int,
        language: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> DBUserSettings:
        """Update user settings"""
        settings = await self.get_by_user(user_id)
        
        if not settings:
            # Create if doesn't exist
            return await self.create(user_id, language or "en", preferences)
        
        if language is not None:
            settings.language = language
        
        if preferences is not None:
            # Merge preferences instead of replacing
            settings.preferences = {**settings.preferences, **preferences}
        
        await self.db.commit()
        await self.db.refresh(settings)
        return settings

    async def get_or_create(self, user_id: int) -> DBUserSettings:
        """Get or create user settings with defaults"""
        settings = await self.get_by_user(user_id)
        if not settings:
            settings = await self.create(user_id)
        return settings

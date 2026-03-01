"""
User settings model for storing user preferences
"""
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import DateTime, Integer, String, ForeignKey, func
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from typing import TYPE_CHECKING
from app.models.base import DatabaseBase as Base

if TYPE_CHECKING:
    from app.models.users import DBUser


class DBUserSettings(Base):
    __tablename__ = "user_settings"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, nullable=False, index=True)
    
    # Language preference for AI responses
    language: Mapped[str] = mapped_column(String, default="en", nullable=False)
    
    # Additional settings stored as JSONB for flexibility
    preferences: Mapped[dict] = mapped_column(JSONB, default={}, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["DBUser"] = relationship("DBUser", back_populates="settings", uselist=False)


from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Boolean, String, Integer, DateTime, func
from datetime import datetime
from typing import Optional, TYPE_CHECKING, List
from app.models.base import DatabaseBase as Base

if TYPE_CHECKING:
    from app.models.bookmarks import DBBookmark
    from app.models.user_settings import DBUserSettings

class DBUser(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    
    # OAuth provider fields
    provider: Mapped[str] = mapped_column(String, nullable=False)  # 'google' or 'github'
    provider_id: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    # Relationships
    bookmarks: Mapped[List["DBBookmark"]] = relationship("DBBookmark", back_populates="user", cascade="all, delete-orphan")
    settings: Mapped[Optional["DBUserSettings"]] = relationship("DBUserSettings", back_populates="user", uselist=False, cascade="all, delete-orphan")

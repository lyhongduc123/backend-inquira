"""
Bookmarks model for saving papers
"""
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Boolean, DateTime, Integer, String, ForeignKey, func, Text
from datetime import datetime
from typing import TYPE_CHECKING, Optional
from app.models.base import DatabaseBase as Base

if TYPE_CHECKING:
    from app.models.papers import DBPaper
    from app.models.users import DBUser


class DBBookmark(Base):
    __tablename__ = "bookmarks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    paper_id: Mapped[str] = mapped_column(String, ForeignKey("papers.paper_id"), nullable=False, index=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["DBUser"] = relationship("DBUser", back_populates="bookmarks")
    paper: Mapped["DBPaper"] = relationship("DBPaper", back_populates="bookmarks")

from datetime import timezone
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Boolean, DateTime, Integer, String, func
from app.models.base import DatabaseBase as Base

class DBAnswer(Base):
    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    query_id: Mapped[int] = mapped_column(index=True, nullable=False)
    answer_text: Mapped[str] = mapped_column(nullable=False)
    is_approved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Boolean, Integer, String, DateTime, Enum, ForeignKey, func
from app.models.base import DatabaseBase as Base

class DBQuery(Base):
    __tablename__ = "queries"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(nullable=False, index=True)
    query_text: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(Enum("pending", "processing", "completed", "failed", name="query_status"), default="pending")
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), nullable=True, index=True)

    messages: Mapped[list] = relationship("DBMessage", back_populates="queries")
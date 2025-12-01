from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Boolean, DateTime, Integer, String, Enum, ForeignKey, func
from app.models.base import DatabaseBase as Base
from app.models.conversations import DBConversation
from .message_papers import DBMessagePaper

if TYPE_CHECKING:
    from app.models.papers import DBPaper  # type: ignore

class DBMessage(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(nullable=False, index=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.conversation_id"), index=True)
    role: Mapped[str] = mapped_column(Enum("user", "assistant", name="message_role"), nullable=False)
    content: Mapped[str] = mapped_column(nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(Enum("pending", "sent", "failed", name="message_status"), default="pending")
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    queries: Mapped[list] = relationship("DBQuery", back_populates="messages")
    conversation: Mapped["DBConversation"] = relationship("DBConversation", back_populates="messages")
    message_papers: Mapped[List["DBMessagePaper"]] = relationship(
        "DBMessagePaper",
        back_populates="message",
        cascade="all, delete-orphan",
    )
    papers: Mapped[List["DBPaper"]] = relationship(
        "DBPaper",
        secondary=DBMessagePaper.__tablename__,
        back_populates="messages",
    )

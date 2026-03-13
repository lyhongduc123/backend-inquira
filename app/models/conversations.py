from sqlalchemy import Boolean, DateTime, Integer, String, Enum, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from typing import TYPE_CHECKING, Literal
from app.models.base import DatabaseBase as Base

if TYPE_CHECKING:
    from app.models.messages import DBMessage
    from app.models.papers import DBPaper

ConversationType = Literal["multi_paper_rag", "single_paper_detail"]

class DBConversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    conversation_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    
    # Conversation type and primary paper for single-paper conversations
    conversation_type: Mapped[str] = mapped_column(
        Enum("multi_paper_rag", "single_paper_detail", name="conversation_type_enum"),
        default="multi_paper_rag",
        nullable=False,
        index=True,
        comment="Type of conversation: multi_paper_rag or single_paper_detail"
    )
    primary_paper_id: Mapped[str | None] = mapped_column(
        String(100),
        ForeignKey("papers.paper_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Primary paper for single-paper detail conversations"
    )
    
    # Metadata for conversation summary and other contextual info
    conversation_metadata: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        default={},
        comment="Metadata including conversation summary, last summarized date, etc."
    )
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    messages: Mapped[list["DBMessage"]] = relationship('DBMessage', back_populates='conversation', lazy='dynamic')
    primary_paper: Mapped["DBPaper | None"] = relationship('DBPaper', foreign_keys=[primary_paper_id])
from sqlalchemy import Boolean, Column, Integer, String, DateTime, Enum, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.models.base import DatabaseBase as Base
from .message_papers import DBMessagePaper


class DBMessage(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    conversation_id = Column(ForeignKey("conversations.conversation_id"), index=True)
    role = Column(Enum("user", "assistant", name="message_role"), nullable=False)
    content = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    status = Column(
        Enum("pending", "sent", "failed", name="message_status"), default="pending"
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    
    # Relationships
    queries = relationship("DBQuery", back_populates="messages")
    conversation = relationship("DBConversation", back_populates="messages")
    message_papers = relationship(
        "DBMessagePaper",
        back_populates="message",
        cascade="all, delete-orphan",
    )
    papers = relationship(
        "DBPaper",
        secondary=DBMessagePaper.__tablename__,
        back_populates="messages",
    )

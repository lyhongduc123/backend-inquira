from sqlalchemy import Column, Integer, ForeignKey, UniqueConstraint
from app.models.base import DatabaseBase as Base
from sqlalchemy.orm import relationship


class DBMessagePaper(Base):
    __tablename__ = "message_papers"

    id = Column(Integer, primary_key=True)
    message_id = Column(
        Integer, ForeignKey("messages.id", ondelete="CASCADE"), index=True
    )
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), index=True)

    message = relationship("DBMessage", back_populates="message_papers")
    paper = relationship("DBPaper", back_populates="message_papers")

    __table_args__ = (
        UniqueConstraint("message_id", "paper_id", name="_message_paper_uc"),
    )

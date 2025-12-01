
from __future__ import annotations

from typing import TYPE_CHECKING
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, ForeignKey, UniqueConstraint
from app.models.base import DatabaseBase as Base

if TYPE_CHECKING:
    from app.models.message import DBMessage  # type: ignore
    from app.models.paper import DBPaper  # type: ignore


class DBMessagePaper(Base):
    __tablename__ = "message_papers"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(Integer, ForeignKey("messages.id", ondelete="CASCADE"), index=True)
    paper_id: Mapped[int] = mapped_column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), index=True)

    message: Mapped["DBMessage"] = relationship("DBMessage", back_populates="message_papers")
    paper: Mapped["DBPaper"] = relationship("DBPaper", back_populates="message_papers")

    __table_args__ = (
        UniqueConstraint("message_id", "paper_id", name="_message_paper_uc"),
    )

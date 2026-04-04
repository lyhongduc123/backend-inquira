"""Shared benchmark corpus model used by evaluation/benchmark services."""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import DatabaseBase


class DBBenchmarkPaper(DatabaseBase):
    """Isolated benchmark paper table."""

    __tablename__ = "beir_test_papers"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    paper_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str] = mapped_column(Text, nullable=True)
    embedding: Mapped[Vector] = mapped_column(Vector(768), nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    citation_count: Mapped[int] = mapped_column(Integer, default=0)
    reference_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

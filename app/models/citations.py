from __future__ import annotations
from typing import TYPE_CHECKING
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Boolean, DateTime, Integer, String, Text, Float, ForeignKey, func, Index
from sqlalchemy.dialects.postgresql import JSONB
from app.models.base import DatabaseBase as Base

if TYPE_CHECKING:
    from app.models.papers import DBPaper


class DBCitation(Base):
    """
    Structured citation tracking for citation network analysis.
    Enables detection of self-citation, cross-field citation, and citation quality.
    """
    __tablename__ = "citations"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    
    # Citation relationship
    citing_paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    cited_paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    
    # Citation context
    citation_context: Mapped[str] = mapped_column(Text, nullable=True)  # Sentence/paragraph where citation appears
    section: Mapped[str] = mapped_column(String(100), nullable=True)  # Introduction, Methods, Discussion, etc.
    
    # Citation type/intent
    intent: Mapped[str] = mapped_column(String(50), nullable=True)  # background, method, result, supporting, contrasting
    is_self_citation: Mapped[bool] = mapped_column(Boolean, default=False, index=True)  # Same author(s)
    is_same_institution: Mapped[bool] = mapped_column(Boolean, default=False)  # Same institution
    
    # Citation quality signals
    is_influential: Mapped[bool] = mapped_column(Boolean, default=False)  # Semantic Scholar influential citation
    mention_count: Mapped[int] = mapped_column(Integer, default=1)  # How many times cited in the paper
    
    # Temporal info
    citation_year: Mapped[int] = mapped_column(Integer, nullable=True, index=True)
    years_since_publication: Mapped[int] = mapped_column(Integer, nullable=True)  # Recency
    
    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    citing_paper: Mapped["DBPaper"] = relationship(
        "DBPaper",
        foreign_keys=[citing_paper_id],
        back_populates="citations_made"
    )
    cited_paper: Mapped["DBPaper"] = relationship(
        "DBPaper",
        foreign_keys=[cited_paper_id],
        back_populates="citations_received"
    )
    
    __table_args__ = (
        Index('idx_citation_pair', 'citing_paper_id', 'cited_paper_id', unique=True),
        Index('idx_citation_analysis', 'is_self_citation', 'is_influential'),
        Index('idx_citation_temporal', 'citation_year', 'years_since_publication'),
    )

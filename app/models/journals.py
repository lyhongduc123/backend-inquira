from __future__ import annotations
from typing import TYPE_CHECKING
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Boolean, DateTime, Integer, String, Text, Float, func, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects import postgresql
from app.models.base import DatabaseBase as Base


class DBJournal(Base):
    """
    SCImago Journal & Country Rank (SJR) data for venue prestige scoring.
    Stores journal metrics for academic legitimacy validation and ranking.
    """
    __tablename__ = "journals"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    
    # Identification
    source_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)  # SJR unique ID
    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    title_normalized: Mapped[str] = mapped_column(String(500), nullable=True, index=True)  # Lowercase, no punctuation
    type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # journal, conference series, book series
    issn: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=True)  # Multiple ISSNs
    issn_text: Mapped[str] = mapped_column(Text, nullable=True)  # Original comma-separated format for backward compatibility
    
    # Publishing info
    publisher: Mapped[str] = mapped_column(String(500), nullable=True, index=True)
    country: Mapped[str] = mapped_column(Text, nullable=True, index=True)
    region: Mapped[str] = mapped_column(Text, nullable=True)
    coverage: Mapped[str] = mapped_column(Text, nullable=True)  # Year range e.g., "1950-2025"
    
    # Open Access
    is_open_access: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_open_access_diamond: Mapped[bool] = mapped_column(Boolean, default=False)  # Free for authors and readers
    
    # SJR Metrics (primary ranking indicators)
    sjr_score: Mapped[float] = mapped_column(Float, nullable=True, index=True)  # SJR indicator value
    sjr_best_quartile: Mapped[str] = mapped_column(String(10), nullable=True, index=True)  # Q1, Q2, Q3, Q4
    h_index: Mapped[int] = mapped_column(Integer, nullable=True, index=True)
    
    # Citation metrics
    total_docs_current_year: Mapped[int] = mapped_column(Integer, nullable=True)  # Documents published in the year
    total_docs_3years: Mapped[int] = mapped_column(Integer, nullable=True)  # Last 3 years
    total_refs: Mapped[int] = mapped_column(Integer, nullable=True)  # Total references
    total_cites_3years: Mapped[int] = mapped_column(Integer, nullable=True)  # Citations received in 3 years
    citable_docs_3years: Mapped[int] = mapped_column(Integer, nullable=True)
    
    # Impact metrics
    cites_per_doc_2years: Mapped[float] = mapped_column(Float, nullable=True, index=True)  # Impact factor equivalent
    refs_per_doc: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Diversity & Social Impact
    percent_female: Mapped[float] = mapped_column(Float, nullable=True)  # % female authors
    overton_count: Mapped[int] = mapped_column(Integer, nullable=True)  # Policy document citations
    sdg_count: Mapped[int] = mapped_column(Integer, nullable=True)  # Sustainable Development Goals coverage
    
    # Research categories
    categories: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=True)  # Specific categories with quartiles
    areas: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=True)  # Broad research areas
    
    # Data year (since we have 2022, 2023, 2024)
    data_year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    
    # Global rank
    rank: Mapped[int] = mapped_column(Integer, nullable=True, index=True)
    
    # Computed fields for matching
    search_terms: Mapped[str] = mapped_column(Text, nullable=True)  # Concatenated searchable text
    
    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_journal_sjr_ranking', 'sjr_score', 'sjr_best_quartile'),
        Index('idx_journal_impact', 'cites_per_doc_2years', 'h_index'),
        Index('idx_journal_title_search', 'title_normalized'),
        Index('idx_journal_year_source', 'data_year', 'source_id'),
        Index('idx_journal_issn_gin', 'issn', postgresql_using='gin'),  # GIN index for array searching
        UniqueConstraint('source_id', 'data_year', name='uq_journal_source_year'),
    )

    def __repr__(self) -> str:
        return f"<DBJournal(title='{self.title}', sjr={self.sjr_score}, quartile={self.sjr_best_quartile})>"

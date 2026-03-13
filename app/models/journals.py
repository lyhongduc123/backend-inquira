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
    source_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    title_normalized: Mapped[str] = mapped_column(String(500), nullable=True, index=True) 
    type: Mapped[str] = mapped_column(String(50), nullable=False, index=True) 
    issn: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=True)  
    issn_text: Mapped[str] = mapped_column(Text, nullable=True) 
    
    # Publishing info
    publisher: Mapped[str] = mapped_column(String(500), nullable=True, index=True)
    country: Mapped[str] = mapped_column(Text, nullable=True, index=True)
    region: Mapped[str] = mapped_column(Text, nullable=True)
    coverage: Mapped[str] = mapped_column(Text, nullable=True)  # Year range e.g., "1950-2025"
    
    # Open Access
    is_open_access: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_open_access_diamond: Mapped[bool] = mapped_column(Boolean, default=False) 

    sjr_score: Mapped[float] = mapped_column(Float, nullable=True, index=True) 
    sjr_best_quartile: Mapped[str] = mapped_column(String(10), nullable=True, index=True)  
    h_index: Mapped[int] = mapped_column(Integer, nullable=True, index=True)

    total_docs_current_year: Mapped[int] = mapped_column(Integer, nullable=True) 
    total_docs_3years: Mapped[int] = mapped_column(Integer, nullable=True)  
    total_refs: Mapped[int] = mapped_column(Integer, nullable=True) 
    total_cites_3years: Mapped[int] = mapped_column(Integer, nullable=True)  
    citable_docs_3years: Mapped[int] = mapped_column(Integer, nullable=True)
    
    cites_per_doc_2years: Mapped[float] = mapped_column(Float, nullable=True, index=True)  
    refs_per_doc: Mapped[float] = mapped_column(Float, nullable=True)
    percent_female: Mapped[float] = mapped_column(Float, nullable=True)  
    overton_count: Mapped[int] = mapped_column(Integer, nullable=True)  
    sdg_count: Mapped[int] = mapped_column(Integer, nullable=True) 
    categories: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=True) 
    areas: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=True)  
    data_year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=True, index=True)
    search_terms: Mapped[str] = mapped_column(Text, nullable=True) 
    
    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_journal_sjr_ranking', 'sjr_score', 'sjr_best_quartile'),
        Index('idx_journal_impact', 'cites_per_doc_2years', 'h_index'),
        Index('idx_journal_title_search', 'title_normalized'),
        Index('idx_journal_year_source', 'data_year', 'source_id'),
        Index('idx_journal_issn_gin', 'issn', postgresql_using='gin'), 
        UniqueConstraint('source_id', 'data_year', name='uq_journal_source_year'),
    )

    def __repr__(self) -> str:
        return f"<DBJournal(title='{self.title}', sjr={self.sjr_score}, quartile={self.sjr_best_quartile})>"

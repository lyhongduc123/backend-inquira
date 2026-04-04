from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
    Boolean,
    DateTime,
    Integer,
    String,
    Text,
    Float,
    Date,
    ForeignKey,
    func,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB
from app.models.base import DatabaseBase as Base

if TYPE_CHECKING:
    from app.models.papers import DBPaper
    from app.models.institutions import DBInstitution


class DBAuthor(Base):
    """
    First-class author entity for trust and reputation tracking.
    Supports author disambiguation, career trajectory analysis, and network effects.
    """

    __tablename__ = "authors"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    author_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="Primary author ID (Semantic Scholar preferred)",
    )
    openalex_id: Mapped[str] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        comment="OpenAlex author ID (always stored separately)",
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=True)

    orcid: Mapped[str] = mapped_column(
        String(50), nullable=True, index=True
    )
    external_ids: Mapped[dict] = mapped_column(
        JSONB, nullable=True
    )  # OpenAlex, Semantic Scholar, etc.

    h_index: Mapped[int] = mapped_column(Integer, nullable=True, default=None)
    i10_index: Mapped[int] = mapped_column(Integer, nullable=True, default=None)
    openalex_counts_by_year: Mapped[dict] = mapped_column(
        JSONB,
        nullable=True,
        comment="Cached OpenAlex counts_by_year payload keyed by year",
    )
    g_index: Mapped[int] = mapped_column(Integer, nullable=True, default=None, comment="g-index: largest number g where top g papers have at least g² citations")
    total_citations: Mapped[int] = mapped_column(Integer, nullable=True, default=None, index=True)
    total_papers: Mapped[int] = mapped_column(Integer, nullable=True, default=None)
    retracted_papers_count: Mapped[int] = mapped_column(Integer, nullable=True, default=0)
    has_retracted_papers: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    url: Mapped[str] = mapped_column(Text, nullable=True)
    
    first_publication_year: Mapped[int] = mapped_column(Integer, nullable=True)
    last_known_institution_id: Mapped[int] = mapped_column(
        ForeignKey("institutions.id"), nullable=True
    )
    retraction_rate: Mapped[float] = mapped_column(Float, nullable=True, index=True)
    field_weighted_citation_impact: Mapped[float] = mapped_column(
        Float, nullable=True
    )  # This might be not computable
    is_corresponding_author_frequently: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  
    average_author_position: Mapped[float] = mapped_column(
        Float, nullable=True
    ) 
    self_citation_rate: Mapped[float] = mapped_column(
        Float, nullable=True
    ) 
    homepage_url: Mapped[str] = mapped_column(Text, nullable=True)
    is_processed: Mapped[bool] = mapped_column(
        Boolean, default=False, index=True, comment="Whether author has been processed for conflict detection"
    )
    is_conflict: Mapped[bool] = mapped_column(
        Boolean, default=False, index=True, comment="Whether author profile has significant data conflicts (>50% citation difference between sources)"
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    last_paper_indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    papers: Mapped[list["DBAuthorPaper"]] = relationship(
        "DBAuthorPaper", back_populates="author", cascade="all, delete-orphan"
    )
    author_institutions: Mapped[list["DBAuthorInstitution"]] = relationship(
        "DBAuthorInstitution", back_populates="author", cascade="all, delete-orphan"
    )
    last_known_institution: Mapped["DBInstitution"] = relationship(
        "DBInstitution", foreign_keys=[last_known_institution_id]
    )

    __table_args__ = (
        Index("idx_author_reputation", "retraction_rate", "total_citations"),
        Index("idx_author_verified", "verified"),
    )


class DBAuthorPaper(Base):
    """
    Association table for many-to-many relationship between authors and papers.
    Includes author contribution metadata for trust signals.
    """

    __tablename__ = "author_papers"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    author_id: Mapped[int] = mapped_column(
        ForeignKey("authors.id"), nullable=False, index=True
    )
    paper_id: Mapped[int] = mapped_column(
        ForeignKey("papers.id"), nullable=False, index=True
    )

    # Author position/role metadata
    author_position: Mapped[int] = mapped_column(
        Integer, nullable=True
    ) 
    is_corresponding: Mapped[bool] = mapped_column(Boolean, default=False)
    institution_id: Mapped[int] = mapped_column(
        ForeignKey("institutions.id"), nullable=True, index=True
    )
    institution_raw: Mapped[str] = mapped_column(
        Text, nullable=True
    )  
    author_string: Mapped[str] = mapped_column(String(255), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    author: Mapped["DBAuthor"] = relationship(
        "DBAuthor", back_populates="papers"
    )
    paper: Mapped["DBPaper"] = relationship("DBPaper", back_populates="authors")
    institution: Mapped["DBInstitution"] = relationship(
        "DBInstitution", back_populates="author_papers"
    )

    __table_args__ = (
        Index("idx_author_paper_unique", "author_id", "paper_id", unique=True),
        Index("idx_corresponding_authors", "is_corresponding", "author_id"),
    )


class DBAuthorInstitution(Base):
    """
    Temporal tracking of author-institution affiliations.
    Enables career trajectory and institutional diversity analysis.
    """

    __tablename__ = "author_institutions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    author_id: Mapped[int] = mapped_column(
        ForeignKey("authors.id"), nullable=False, index=True
    )
    institution_id: Mapped[int] = mapped_column(
        ForeignKey("institutions.id"), nullable=False, index=True
    )

    # Temporal information
    start_year: Mapped[int] = mapped_column(Integer, nullable=True)
    end_year: Mapped[int] = mapped_column(Integer, nullable=True)  # NULL = current
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)

    # Evidence
    paper_count: Mapped[int] = mapped_column(
        Integer, default=1
    )  # How many papers link this affiliation

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    author: Mapped["DBAuthor"] = relationship(
        "DBAuthor", back_populates="author_institutions"
    )
    institution: Mapped["DBInstitution"] = relationship(
        "DBInstitution", back_populates="author_institutions"
    )

    __table_args__ = (
        Index("idx_author_institution_temporal", "author_id", "is_current"),
    )

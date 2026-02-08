from __future__ import annotations
from typing import TYPE_CHECKING
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Boolean, DateTime, Integer, String, Text, Float, Date, ForeignKey, func, Index
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
        String(100), unique=True, nullable=False, index=True,
        comment="Primary author ID (Semantic Scholar preferred, OpenAlex fallback)"
    )
    openalex_id: Mapped[str] = mapped_column(
        String(100), nullable=True, index=True,
        comment="OpenAlex author ID (always stored separately)"
    )
    
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=True)

    orcid: Mapped[str] = mapped_column(String(50), nullable=True, unique=True, index=True)
    external_ids: Mapped[dict] = mapped_column(JSONB, nullable=True)  # OpenAlex, Semantic Scholar, etc.
    
    h_index: Mapped[int] = mapped_column(Integer, default=0)
    i10_index: Mapped[int] = mapped_column(Integer, default=0)
    total_citations: Mapped[int] = mapped_column(Integer, default=0, index=True)
    total_papers: Mapped[int] = mapped_column(Integer, default=0)
    

    verified: Mapped[bool] = mapped_column(Boolean, default=False)  # ORCID-verified or manually verified
    retracted_papers_count: Mapped[int] = mapped_column(Integer, default=0, index=True)
    
    # Career trajectory
    first_publication_year: Mapped[int] = mapped_column(Integer, nullable=True)
    last_known_institution_id: Mapped[int] = mapped_column(ForeignKey("institutions.id"), nullable=True)
    
    # Reputation scores (computed)
    reputation_score: Mapped[float] = mapped_column(Float, nullable=True, index=True)  # 0-100 composite score
    field_weighted_citation_impact: Mapped[float] = mapped_column(Float, nullable=True)  # Average FWCI across papers
    collaboration_diversity_score: Mapped[float] = mapped_column(Float, nullable=True)  # Network breadth
    
    # Academic seniority indicators
    is_corresponding_author_frequently: Mapped[bool] = mapped_column(Boolean, default=False)  # >50% of papers
    average_author_position: Mapped[float] = mapped_column(Float, nullable=True)  # 1.0 = always first, 0.0 = always last
    
    # Red flags
    has_retracted_papers: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    self_citation_rate: Mapped[float] = mapped_column(Float, nullable=True)  # % of citations from own papers
    
    # Homepage & social
    homepage_url: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_paper_indexed_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    paper_authors: Mapped[list["DBAuthorPaper"]] = relationship(
        "DBAuthorPaper", 
        back_populates="author",
        cascade="all, delete-orphan"
    )
    author_institutions: Mapped[list["DBAuthorInstitution"]] = relationship(
        "DBAuthorInstitution",
        back_populates="author",
        cascade="all, delete-orphan"
    )
    last_known_institution: Mapped["DBInstitution"] = relationship(
        "DBInstitution",
        foreign_keys=[last_known_institution_id]
    )
    
    __table_args__ = (
        Index('idx_author_reputation', 'reputation_score', 'total_citations'),
        Index('idx_author_trust', 'verified', 'has_retracted_papers'),
    )


class DBAuthorPaper(Base):
    """
    Association table for many-to-many relationship between authors and papers.
    Includes author contribution metadata for trust signals.
    """
    __tablename__ = "author_papers"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    author_id: Mapped[int] = mapped_column(ForeignKey("authors.id"), nullable=False, index=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    
    # Author position/role metadata
    author_position: Mapped[int] = mapped_column(Integer, nullable=True)  # 1 = first author, etc.
    is_corresponding: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Affiliation at time of paper
    institution_id: Mapped[int] = mapped_column(ForeignKey("institutions.id"), nullable=True, index=True)
    institution_raw: Mapped[str] = mapped_column(Text, nullable=True)  # Raw affiliation string
    
    # Author string as appeared in paper (for disambiguation tracking)
    author_string: Mapped[str] = mapped_column(String(255), nullable=True)
    
    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    author: Mapped["DBAuthor"] = relationship("DBAuthor", back_populates="paper_authors")
    paper: Mapped["DBPaper"] = relationship("DBPaper", back_populates="paper_authors")
    institution: Mapped["DBInstitution"] = relationship("DBInstitution", back_populates="author_papers")
    
    __table_args__ = (
        Index('idx_author_paper_unique', 'author_id', 'paper_id', unique=True),
        Index('idx_corresponding_authors', 'is_corresponding', 'author_id'),
    )


class DBAuthorInstitution(Base):
    """
    Temporal tracking of author-institution affiliations.
    Enables career trajectory and institutional diversity analysis.
    """
    __tablename__ = "author_institutions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    author_id: Mapped[int] = mapped_column(ForeignKey("authors.id"), nullable=False, index=True)
    institution_id: Mapped[int] = mapped_column(ForeignKey("institutions.id"), nullable=False, index=True)
    
    # Temporal information
    start_year: Mapped[int] = mapped_column(Integer, nullable=True)
    end_year: Mapped[int] = mapped_column(Integer, nullable=True)  # NULL = current
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Evidence
    paper_count: Mapped[int] = mapped_column(Integer, default=1)  # How many papers link this affiliation
    confidence: Mapped[float] = mapped_column(Float, default=1.0)  # Confidence in this affiliation
    
    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    author: Mapped["DBAuthor"] = relationship("DBAuthor", back_populates="author_institutions")
    institution: Mapped["DBInstitution"] = relationship("DBInstitution", back_populates="author_institutions")
    
    __table_args__ = (
        Index('idx_author_institution_temporal', 'author_id', 'is_current'),
    )

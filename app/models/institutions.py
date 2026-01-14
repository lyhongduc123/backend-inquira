from __future__ import annotations
from typing import TYPE_CHECKING
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Boolean, DateTime, Integer, String, Text, Float, ForeignKey, func, Index
from sqlalchemy.dialects.postgresql import JSONB
from app.models.base import DatabaseBase as Base

if TYPE_CHECKING:
    from app.models.authors import DBAuthorPaper, DBAuthorInstitution


class DBInstitution(Base):
    """
    First-class institution entity for institutional trust and reputation.
    Enables lab/department-level trust signals and geographic diversity tracking.
    """
    __tablename__ = "institutions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    institution_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    
    # Basic info
    name: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(500), nullable=True)
    
    # Identification
    ror_id: Mapped[str] = mapped_column(String(100), nullable=True, unique=True, index=True)  # ROR ID
    external_ids: Mapped[dict] = mapped_column(JSONB, nullable=True)  # OpenAlex, GRID, etc.
    
    # Geographic info
    country_code: Mapped[str] = mapped_column(String(5), nullable=True, index=True)
    country: Mapped[str] = mapped_column(String(100), nullable=True)
    city: Mapped[str] = mapped_column(String(100), nullable=True)
    region: Mapped[str] = mapped_column(String(100), nullable=True)  # State/province
    
    # Institution type
    type: Mapped[str] = mapped_column(String(50), nullable=True, index=True)  # Education, Healthcare, Government, etc.
    
    # Reputation metrics
    total_papers: Mapped[int] = mapped_column(Integer, default=0)
    total_citations: Mapped[int] = mapped_column(Integer, default=0, index=True)
    h_index: Mapped[int] = mapped_column(Integer, default=0)
    
    # Trust signals
    reputation_score: Mapped[float] = mapped_column(Float, nullable=True, index=True)  # 0-100 composite score
    avg_paper_quality: Mapped[float] = mapped_column(Float, nullable=True)  # Average FWCI of papers
    retraction_rate: Mapped[float] = mapped_column(Float, default=0.0)  # % of papers retracted
    
    # Research profile
    top_concepts: Mapped[list] = mapped_column(JSONB, nullable=True)  # Top research areas
    
    # Homepage
    homepage_url: Mapped[str] = mapped_column(Text, nullable=True)
    wikipedia_url: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    author_papers: Mapped[list["DBAuthorPaper"]] = relationship(
        "DBAuthorPaper",
        back_populates="institution"
    )
    author_institutions: Mapped[list["DBAuthorInstitution"]] = relationship(
        "DBAuthorInstitution",
        back_populates="institution",
        cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        Index('idx_institution_reputation', 'reputation_score', 'total_citations'),
        Index('idx_institution_location', 'country_code', 'type'),
    )

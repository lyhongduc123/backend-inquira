"""
Model for cached author-to-author relationships.
Stores computed relationships to avoid expensive API calls.
"""
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import DateTime, Integer, String, ForeignKey, func, Index
from app.models.base import DatabaseBase as Base


class DBAuthorRelationship(Base):
    """
    Cached author-to-author relationships.
    
    Relationship types:
    - 'collaboration': Co-authored papers together
    - 'citing': related_author has cited author's papers
    - 'referenced': author has referenced related_author's papers
    """
    __tablename__ = "author_relationships"
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    
    # Relationship definition
    author_id: Mapped[int] = mapped_column(
        ForeignKey("authors.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Primary author"
    )
    related_author_id: Mapped[int] = mapped_column(
        ForeignKey("authors.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Related author"
    )
    relationship_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="collaboration, citing, or referenced"
    )
    
    # Relationship strength
    relationship_count: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
        comment="Number of papers with this relationship"
    )
    
    # Cache metadata
    last_computed_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="When this relationship was last computed"
    )
    
    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
    
    __table_args__ = (
        Index('idx_author_rel_lookup', 'author_id', 'relationship_type', 'relationship_count'),
        Index('idx_author_rel_reverse', 'related_author_id', 'relationship_type'),
        Index('idx_author_rel_unique', 'author_id', 'related_author_id', 'relationship_type', unique=True),
    )

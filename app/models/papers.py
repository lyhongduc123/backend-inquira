from __future__ import annotations
from datetime import date
from typing import TYPE_CHECKING, List
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
    Boolean,
    DateTime,
    Integer,
    String,
    Text,
    Float,
    ARRAY,
    Date,
    ForeignKey,
    func,
    Index,
    literal_column,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from app.models.base import DatabaseBase as Base

if TYPE_CHECKING:
    from app.models.authors import DBAuthorPaper
    from app.models.citations import DBCitation
    from app.models.messages import DBMessage
    from app.models.message_papers import DBMessagePaper
    from app.models.journals import DBJournal
    from app.models.bookmarks import DBBookmark
    from app.models.conferences import DBConference


class DBPaper(Base):
    """
    Academic paper entity with trust scoring and citation analysis.

    Aggregates data from multiple sources (OpenAlex, Semantic Scholar, CrossRef)
    with computed trust metrics, citation networks, and semantic embeddings.
    """

    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    paper_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique identifier (Semantic Scholar preferred, OpenAlex/DOI fallback)",
    )

    title: Mapped[str] = mapped_column(Text, nullable=False, comment="Full paper title")

    abstract: Mapped[str] = mapped_column(
        Text, nullable=True, comment="Paper abstract/summary"
    )
    embedding: Mapped[Vector] = mapped_column(
        Vector(768),
        nullable=True,
        comment="768-dim vector embedding of abstract and title",
    )
    publication_date: Mapped[date] = mapped_column(
        Date, nullable=True, comment="Publication date"
    )
    venue: Mapped[str] = mapped_column(
        String, nullable=True, comment="Publication venue/conference"
    )
    issn: Mapped[list[str]] = mapped_column(
        ARRAY(String(8)),
        nullable=True,
        index=True,
        comment="Primary ISSN for journal identification",
    )
    issn_l: Mapped[str] = mapped_column(
        String(8), nullable=True, index=True, comment="Linking ISSN (OpenAlex standard)"
    )

    url: Mapped[str] = mapped_column(
        Text, nullable=True, comment="Primary URL to paper"
    )
    pdf_url: Mapped[str] = mapped_column(
        Text, nullable=True, comment="Direct PDF download URL"
    )
    is_open_access: Mapped[bool] = mapped_column(
        Boolean, default=False, index=True, comment="Whether paper is open access"
    )
    open_access_pdf: Mapped[dict] = mapped_column(
        JSONB, nullable=True, comment="Open access PDF metadata (URL, version, license)"
    )
    citation_styles: Mapped[dict] = mapped_column(
        JSONB,
        nullable=True,
    )

    source: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Data source: openalex, semanticscholar",
    )
    external_ids: Mapped[dict] = mapped_column(
        JSONB, nullable=True, comment="External IDs (DOI, PubMed, ArXiv, etc.)"
    )

    tldr: Mapped[str] = mapped_column(
        Text,
        nullable=True,
        comment="TL;DR summary from Semantic Scholar (replaces AI-generated summary)",
    )
    tldr_embedding: Mapped[Vector] = mapped_column(
        Vector(768),
        nullable=True,
        comment="768-dim vector embedding of TLDR (nomic-embed-text)",
    )
    citation_count: Mapped[int] = mapped_column(
        Integer, default=0, index=True, comment="Total number of citations"
    )
    influential_citation_count: Mapped[int] = mapped_column(
        Integer, default=0, comment="Influential citations (Semantic Scholar only)"
    )
    reference_count: Mapped[int] = mapped_column(
        Integer, default=0, comment="Number of references cited by this paper"
    )

    # Semantic Scholar specific fields
    year: Mapped[int] = mapped_column(
        Integer,
        nullable=True,
        index=True,
        comment="Publication year extracted by Semantic Scholar",
    )
    fields_of_study: Mapped[list[str]] = mapped_column(
        ARRAY(String(100)),
        nullable=True,
        comment="Research fields from Semantic Scholar classification",
    )
    publication_types: Mapped[list[str]] = mapped_column(
        ARRAY(String(100)),
        nullable=True,
        comment="Publication types from Semantic Scholar (e.g., JournalArticle, Review)",
    )
    s2_fields_of_study: Mapped[dict] = mapped_column(
        JSONB,
        nullable=True,
        comment="Enriched S2 fields with source attribution [{category, source}]",
    )
    paper_tags: Mapped[list] = mapped_column(
        JSONB,
        nullable=True,
        comment="Computed zero-shot tags with confidence scores",
    )

    topics: Mapped[list] = mapped_column(
        JSONB, nullable=True, comment="OpenAlex research topics with relevance scores"
    )
    keywords: Mapped[list] = mapped_column(
        JSONB, nullable=True, comment="Extracted keywords with scores"
    )
    concepts: Mapped[list] = mapped_column(
        JSONB,
        nullable=True,
        comment="OpenAlex concepts with scores and hierarchy levels",
    )
    mesh_terms: Mapped[list] = mapped_column(
        JSONB, nullable=True, comment="MeSH terms for biomedical papers"
    )

    citation_percentile: Mapped[dict] = mapped_column(
        JSONB, nullable=True, comment="Citation percentile rankings by field and year"
    )
    fwci: Mapped[float] = mapped_column(
        Float,
        nullable=True,
        index=True,
        comment="Field-Weighted Citation Impact (OpenAlex)",
    )

    author_trust_score: Mapped[float] = mapped_column(
        Float, nullable=True, index=True, comment="Average trust score of all authors"
    )

    journal_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("journals.id"),
        nullable=True,
        index=True,
        comment="Foreign key to journals table (SJR data)",
    )
    conference_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("conferences.id"),
        nullable=True,
        index=True,
        comment="Foreign key to conferences table (CORE ranking data)",
    )

    is_retracted: Mapped[bool] = mapped_column(
        Boolean, default=False, index=True, comment="Whether paper has been retracted"
    )
    language: Mapped[str] = mapped_column(
        String(10), nullable=True, comment="ISO 639-1 language code"
    )

    corresponding_author_ids: Mapped[list] = mapped_column(
        ARRAY(String), nullable=True, comment="IDs of corresponding authors"
    )
    institutions_distinct_count: Mapped[int] = mapped_column(
        Integer, nullable=True, comment="Number of unique institutions involved"
    )
    countries_distinct_count: Mapped[int] = mapped_column(
        Integer, nullable=True, comment="Number of unique countries involved"
    )

    is_processed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        index=True,
        comment="Whether paper has been fully processed",
    )
    processing_status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        comment="Processing status: pending, processing, completed, failed",
    )
    processing_error: Mapped[str] = mapped_column(
        Text, nullable=True, comment="Error message if processing failed"
    )

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="Record creation timestamp",
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        comment="Last update timestamp",
    )
    last_accessed_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="Last access timestamp for cache management",
    )

    authors: Mapped[list["DBAuthorPaper"]] = relationship(
        "DBAuthorPaper", back_populates="paper", cascade="all, delete-orphan"
    )
    references: Mapped[list["DBCitation"]] = relationship(
        "DBCitation",
        foreign_keys="[DBCitation.citing_paper_id]",
        back_populates="citing_paper",
        cascade="all, delete-orphan",
    )
    citations: Mapped[list["DBCitation"]] = relationship(
        "DBCitation",
        foreign_keys="[DBCitation.cited_paper_id]",
        back_populates="cited_paper",
        cascade="all, delete-orphan",
    )
    message_papers: Mapped[list] = relationship(
        "DBMessagePaper", back_populates="paper", cascade="all, delete-orphan"
    )
    messages: Mapped[list] = relationship(
        "DBMessage",
        secondary="message_papers",
        back_populates="papers",
        overlaps="message_papers",
        viewonly=True,
    )
    chunks: Mapped[list["DBPaperChunk"]] = relationship(
        "DBPaperChunk", back_populates="paper", cascade="all, delete-orphan"
    )
    conference: Mapped["DBConference"] = relationship(
        "DBConference", foreign_keys=[conference_id]
    )
    journal: Mapped["DBJournal"] = relationship("DBJournal", foreign_keys=[journal_id])

    bookmarks: Mapped[List["DBBookmark"]] = relationship(
        "DBBookmark", back_populates="paper", cascade="all, delete-orphan"
    )
    """User bookmarks for this paper."""

    __table_args__ = (
        # GIN index for fields_of_study array searches
        Index(
            "ix_papers_fields_of_study_gin", "fields_of_study", postgresql_using="gin"
        ),
        # Composite filter index for common constrained retrieval
        # (open access + publication year range + citation count range)
        Index(
            "ix_papers_open_access_year_citation",
            "is_open_access",
            "year",
            "citation_count",
        ),
        # HNSW index for semantic retrieval on title+abstract embeddings
        Index(
            "ix_papers_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_where=text("embedding IS NOT NULL"),
        ),
        # HNSW index for semantic retrieval on TLDR embeddings
        Index(
            "ix_papers_tldr_embedding_hnsw",
            "tldr_embedding",
            postgresql_using="hnsw",
            postgresql_ops={"tldr_embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_where=text("tldr_embedding IS NOT NULL"),
        ),
        Index(
            "ix_papers_title_abstract_gin",
            text("to_tsvector('english', title || ' ' || abstract)"),
            postgresql_using="gin",
        ),
    )


class DBPaperChunk(Base):
    """
    Text chunks for semantic search and citation verification.

    Papers are split into chunks with structural metadata and embeddings
    for efficient semantic retrieval and context extraction.
    """

    __tablename__ = "paper_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    chunk_id: Mapped[str] = mapped_column(
        String(150),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique chunk identifier (paper_id + chunk_index)",
    )
    paper_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("papers.paper_id"),
        nullable=False,
        index=True,
        comment="Reference to parent paper",
    )

    text: Mapped[str] = mapped_column(
        Text, nullable=False, comment="Chunk text content"
    )
    token_count: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="Number of tokens in chunk"
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="Sequential chunk number within paper"
    )

    char_start: Mapped[int] = mapped_column(
        Integer, nullable=True, comment="Character offset start position in full text"
    )
    char_end: Mapped[int] = mapped_column(
        Integer, nullable=True, comment="Character offset end position in full text"
    )

    section_title: Mapped[str] = mapped_column(
        Text, nullable=True, comment="Section header/title if applicable"
    )
    page_number: Mapped[int] = mapped_column(
        Integer, nullable=True, comment="Page number in original PDF"
    )

    label: Mapped[str] = mapped_column(
        String(50),
        nullable=True,
        comment="Docling label: section_header, text, caption, etc.",
    )
    level: Mapped[int] = mapped_column(
        Integer, nullable=True, comment="Hierarchy level for section headers"
    )
    docling_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=True,
        comment="Docling extraction metadata (bbox, page_no, etc.)",
    )

    embedding: Mapped[Vector] = mapped_column(
        Vector(768),
        nullable=True,
        comment="768-dim vector embedding (nomic-embed-text)",
    )

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="Chunk creation timestamp",
    )

    paper: Mapped["DBPaper"] = relationship("DBPaper", back_populates="chunks")

    __table_args__ = (
        # HNSW index for fast chunk-level vector similarity search
        Index(
            "ix_paper_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_where=literal_column("embedding IS NOT NULL"),
        ),
    )

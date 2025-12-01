
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Boolean, DateTime, Integer, String, Text, Float, ARRAY, Date, ForeignKey, func
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from app.models.base import DatabaseBase as Base



class DBPaper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    paper_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[dict] = mapped_column(JSONB)
    abstract: Mapped[str] = mapped_column(Text)
    publication_date: Mapped[Date] = mapped_column(Date)
    venue: Mapped[str] = mapped_column(String)
    url: Mapped[str] = mapped_column(Text)
    pdf_url: Mapped[str] = mapped_column(Text)
    is_open_access: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    open_access_pdf: Mapped[dict] = mapped_column(JSONB)

    # Metadata
    source: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    external_id: Mapped[str] = mapped_column(String(200), index=True)

    # Summary
    summary: Mapped[str] = mapped_column(Text)
    summary_embedding: Mapped[Vector] = mapped_column(Vector(768))

    # Relevance scoring
    relevance_score: Mapped[float] = mapped_column(Float)
    citation_count: Mapped[int] = mapped_column(Integer, default=0)
    influential_citation_count: Mapped[int] = mapped_column(Integer)
    reference_count: Mapped[int] = mapped_column(Integer)

    # Status tracking
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    processing_status: Mapped[str] = mapped_column(String(50), default="pending")
    processing_error: Mapped[str] = mapped_column(Text)

    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_accessed_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    chunks: Mapped[list] = relationship(
        "DBPaperChunk", back_populates="paper", cascade="all, delete-orphan"
    )
    citations: Mapped[list] = relationship(
        "DBCitationMap", back_populates="paper", cascade="all, delete-orphan"
    )
    message_papers: Mapped[list] = relationship(
        "DBMessagePaper",
        back_populates="paper",
        cascade="all, delete-orphan"
    )
    messages: Mapped[list] = relationship(
        "DBMessage",
        secondary="message_papers",
        back_populates="papers"
    )



class DBPaperChunk(Base):
    __tablename__ = "paper_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    chunk_id: Mapped[str] = mapped_column(String(150), unique=True, nullable=False, index=True)
    paper_id: Mapped[str] = mapped_column(String(100), ForeignKey("papers.paper_id"), nullable=False, index=True)

    # Chunk content
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Chunk metadata
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section_title: Mapped[str] = mapped_column(Text)
    page_number: Mapped[int] = mapped_column(Integer)

    # Vector embedding (768-dim for Ollama nomic-embed-text)
    embedding: Mapped[Vector] = mapped_column(Vector(768), nullable=True)

    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    paper: Mapped["DBPaper"] = relationship("DBPaper", back_populates="chunks")



class DBCitationMap(Base):
    __tablename__ = "citation_map"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    query_id: Mapped[str] = mapped_column(String(100))
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_id: Mapped[str] = mapped_column(String(150), nullable=False, index=True)
    paper_id: Mapped[str] = mapped_column(String(100), ForeignKey("papers.paper_id"), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float)

    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    paper: Mapped["DBPaper"] = relationship("DBPaper", back_populates="citations")



class DBResearchQuery(Base):
    __tablename__ = "research_queries"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_embedding: Mapped[Vector] = mapped_column(Vector(1536))

    # Retrieved papers
    retrieved_paper_ids: Mapped[list] = mapped_column(ARRAY(String))

    # Results
    answer: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)

    # Metadata
    user_id: Mapped[int] = mapped_column(Integer)
    session_id: Mapped[str] = mapped_column(String(100))

    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

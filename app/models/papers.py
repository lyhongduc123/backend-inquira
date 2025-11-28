from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    ARRAY,
    Date,
    Index,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.models.base import DatabaseBase as Base


class DBPaper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(String(100), unique=True, nullable=False, index=True)
    title = Column(Text, nullable=False)
    authors = Column(JSONB)
    abstract = Column(Text)
    publication_date = Column(Date)
    venue = Column(String)
    url = Column(Text)
    pdf_url = Column(Text)
    is_open_access = Column(Boolean, default=False, index=True)
    open_access_pdf = Column(JSONB)  # {"url": str, "status": str, "license": str}

    # Metadata
    source = Column(String(50), nullable=False, index=True)
    external_id = Column(String(200), index=True)

    # Summary
    summary = Column(Text)
    summary_embedding = Column(Vector(768))

    # Relevance scoring
    relevance_score = Column(Float)
    citation_count = Column(Integer, default=0)
    influential_citation_count = Column(Integer)
    reference_count = Column(Integer)

    # Status tracking
    is_processed = Column(Boolean, default=False, index=True)
    processing_status = Column(String(50), default="pending")
    processing_error = Column(Text)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    last_accessed_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    chunks = relationship(
        "DBPaperChunk", back_populates="paper", cascade="all, delete-orphan"
    )
    citations = relationship(
        "DBCitationMap", back_populates="paper", cascade="all, delete-orphan"
    )
    message_papers = relationship(
        "DBMessagePaper",
        back_populates="paper",
        cascade="all, delete-orphan"
    )
    messages = relationship(
        "DBMessage",
        secondary="message_papers",
        back_populates="papers"
    )


class DBPaperChunk(Base):
    __tablename__ = "paper_chunks"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(String(150), unique=True, nullable=False, index=True)
    paper_id = Column(
        String(100), ForeignKey("papers.paper_id"), nullable=False, index=True
    )

    # Chunk content
    text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)

    # Chunk metadata
    chunk_index = Column(Integer, nullable=False)
    section_title = Column(Text)
    page_number = Column(Integer)

    # Vector embedding (768-dim for Ollama nomic-embed-text)
    embedding = Column(Vector(768), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    paper = relationship("DBPaper", back_populates="chunks")


class DBCitationMap(Base):
    __tablename__ = "citation_map"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(String(100))
    claim_text = Column(Text, nullable=False)
    chunk_id = Column(String(150), nullable=False, index=True)
    paper_id = Column(
        String(100), ForeignKey("papers.paper_id"), nullable=False, index=True
    )
    confidence = Column(Float)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    paper = relationship("DBPaper", back_populates="citations")


class DBResearchQuery(Base):
    __tablename__ = "research_queries"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    query_embedding = Column(Vector(1536))

    # Retrieved papers
    retrieved_paper_ids = Column(ARRAY(String))

    # Results
    answer = Column(Text)
    confidence = Column(Float)

    # Metadata
    user_id = Column(Integer)
    session_id = Column(String(100))

    created_at = Column(DateTime(timezone=True), server_default=func.now())

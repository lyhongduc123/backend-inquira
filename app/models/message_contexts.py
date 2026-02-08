from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Boolean, Integer, String, DateTime, Text, Float, JSON, ForeignKey, func, UniqueConstraint
from app.models.base import DatabaseBase as Base
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.messages import DBMessage
    from app.models.papers import DBPaper, DBPaperChunk


class DBMessageContext(Base):
    """
    Tracks which chunks/papers were used as context for generating a message.
    Stores the prompt snapshot and generation metadata at the time of generation.
    This enables:
    - Reproducibility: See exactly what context was used
    - Citation verification: Check if citations match used chunks
    - Debugging: Inspect prompts and context for each response
    - Analytics: Track which papers/chunks are most useful
    """
    __tablename__ = "message_contexts"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    
    # Foreign keys
    message_id: Mapped[int] = mapped_column(
        Integer, 
        ForeignKey("messages.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True,
        comment="The assistant message that was generated"
    )
    paper_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("papers.paper_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Paper this context chunk belongs to"
    )
    chunk_id: Mapped[str | None] = mapped_column(
        String(150),
        ForeignKey("paper_chunks.chunk_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Reference to chunk (nullable if chunk is deleted)"
    )
    
    # Context snapshot at generation time
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Position in the retrieved chunks list (for ordering)"
    )
    relevance_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Retrieval relevance score (from vector search)"
    )
    
    # Prompt engineering metadata
    full_prompt: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Complete prompt sent to LLM (stored once per message)"
    )
    
    # Full chunk context data as JSON snapshot
    context_metadata: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        comment="""Complete chunk data snapshot at generation time:
        {
            'text': str,
            'token_count': int,
            'section_title': str,
            'page_number': int,
            'label': str,
            'level': int,
            'char_start': int,
            'char_end': int,
            'paper_title': str,
            'paper_year': int,
            'authors': list,
            ... (any other relevant metadata)
        }
        """
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    
    # Relationships
    message: Mapped["DBMessage"] = relationship(
        "DBMessage",
        back_populates="message_contexts"
    )
    paper: Mapped["DBPaper"] = relationship("DBPaper")
    chunk: Mapped["DBPaperChunk | None"] = relationship("DBPaperChunk")
    
    # Note: No unique constraint since chunk_id can be NULL
    # Multiple contexts can have same message_id if chunk was deleted

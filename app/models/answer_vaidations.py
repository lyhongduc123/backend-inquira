from typing import TYPE_CHECKING
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
    Boolean,
    ForeignKey,
    Integer,
    String,
    DateTime,
    Text,
    Float,
    JSON,
    func,
)
from app.models.base import DatabaseBase as Base
from datetime import datetime

if TYPE_CHECKING:
    from app.models.messages import DBMessage


class DBAnswerValidation(Base):
    """
    Stores answer validation checks and detailed analysis.
    Used for inspecting LLM responses: hallucinations, citations, text matching.
    """

    __tablename__ = "answer_validations"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Input ref
    query_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Original user query snapshot (for standalone validation records)",
    )

    enhanced_query: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="LLM-ready query after history/context enhancement",
    )

    context_used: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Exact built context string passed into LLM",
    )

    context_chunks: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Exact chunk payload used to build LLM context",
    )

    # Output
    # Link to the message being validated
    message_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="The assistant message being validated (message.content has the answer)",
    )

    model_name: Mapped[str | None] = mapped_column(
        String(100), nullable=True, server_default="gpt-oss", comment="Model used for validation"
    )

    # Validation results
    has_hallucination: Mapped[bool] = mapped_column(
        Boolean, default=False, comment="Whether answer contains unsupported claims"
    )
    hallucination_count: Mapped[int] = mapped_column(
        Integer, default=0, comment="Number of hallucinated facts detected"
    )
    hallucination_details: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, comment="Specific hallucinated claims"
    )

    non_existent_facts: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, comment="Facts not found in context"
    )
    incorrect_citations: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, comment="Citations with wrong paper IDs or not in context"
    )

    # Scores
    relevance_score: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="How relevant answer is to query (0-1)"
    )
    factual_accuracy_score: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Factual correctness (0-1)"
    )
    citation_accuracy: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Citation accuracy (0-1)"
    )

    # Citation verification
    total_citations: Mapped[int] = mapped_column(Integer, default=0)
    correct_citations: Mapped[int] = mapped_column(Integer, default=0)
    hallucinated_citations: Mapped[int] = mapped_column(Integer, default=0)
    missing_citations: Mapped[int] = mapped_column(Integer, default=0)

    # Metadata
    execution_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), default="pending", comment="pending, completed, failed"
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    message: Mapped["DBMessage"] = relationship(
        "DBMessage", back_populates="validations"
    )

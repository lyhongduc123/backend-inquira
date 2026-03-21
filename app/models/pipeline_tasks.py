"""
Database models for event-driven chat pipeline.
Enables resumable, fault-tolerant chat operations.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, Text, DateTime, Enum, ForeignKey, Index, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import func
from app.models.base import DatabaseBase as Base


class PipelineTaskStatus:
    """Task status enum values"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelinePhase:
    """Pipeline execution phases"""
    INIT = "init"
    SEARCH = "search"
    RANKING = "ranking"
    LLM_GENERATION = "generation"
    VALIDATION = "validation"
    DONE = "done"


class DBPipelineTask(Base):
    """
    Represents a background chat pipeline task.
    
    Enables:
    - Async execution (non-blocking)
    - Progress tracking
    - Resumability (user can disconnect/reconnect)
    - Result caching
    """
    __tablename__ = "pipeline_tasks"
    
    # Primary key
    task_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    
    # Ownership
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    conversation_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    message_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
        comment="Assistant message created after completion"
    )
    
    # Task configuration
    query: Mapped[str] = mapped_column(Text, nullable=False)
    pipeline_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    filters: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    client_message_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Client-side message ID for deduplication"
    )
    
    # Status tracking
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=PipelineTaskStatus.PENDING,
        index=True
    )
    current_phase: Mapped[str | None] = mapped_column(String(50), nullable=True)
    progress_percent: Mapped[int] = mapped_column(Integer, default=0)
    
    # Results (cached for resumability)
    result_papers: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="RAG result papers (ranked)"
    )
    result_chunks: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="RAG result chunks"
    )
    response_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Final LLM response"
    )
    
    # Error handling
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # Relationships
    events: Mapped[List["DBPipelineEvent"]] = relationship(
        "DBPipelineEvent",
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="DBPipelineEvent.sequence_number"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_pipeline_tasks_user_conversation", user_id, conversation_id),
        Index("idx_pipeline_tasks_status_created", status, created_at),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "query": self.query,
            "pipeline_type": self.pipeline_type,
            "status": self.status,
            "current_phase": self.current_phase,
            "progress_percent": self.progress_percent,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class PipelineEventType:
    """Event type enum values"""
    STEP = "step"              # Progress step
    METADATA = "metadata"      # Paper metadata
    CHUNK = "chunk"            # Response text chunk
    REASONING = "reasoning"    # Model reasoning
    ERROR = "error"            # Error event
    DONE = "done"              # Completion signal


class DBPipelineEvent(Base):
    """
    Represents a single event in pipeline execution.
    
    Events are ordered by sequence_number for:
    - Replay capability
    - Resume from last event
    - Event sourcing pattern
    """
    __tablename__ = "pipeline_events"
    
    # Primary key
    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key to task
    task_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("pipeline_tasks.task_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Event data
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    event_data: Mapped[dict] = mapped_column(JSONB, nullable=False)
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Relationships
    task: Mapped["DBPipelineTask"] = relationship(
        "DBPipelineTask",
        back_populates="events"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_pipeline_events_task_sequence", task_id, sequence_number),
        Index("idx_pipeline_events_task_created", task_id, created_at),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for SSE streaming"""
        return {
            "event_id": self.event_id,
            "task_id": self.task_id,
            "event_type": self.event_type,
            "event_data": self.event_data,
            "sequence_number": self.sequence_number,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

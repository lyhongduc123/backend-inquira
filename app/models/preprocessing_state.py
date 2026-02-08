"""
Preprocessing state tracking model.

Tracks the state of ongoing preprocessing jobs for continuity across restarts.
"""
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime, Boolean, func
from app.models.base import DatabaseBase as Base


class DBPreprocessingState(Base):
    """
    Tracks preprocessing job state for continuity.
    
    Allows resuming from where we left off if the process is interrupted.
    """
    __tablename__ = "preprocessing_states"
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    
    # Job identification
    job_id: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True,
        comment="Unique job identifier"
    )
    
    # Progress tracking
    current_index: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False,
        comment="Current index in dataset stream"
    )
    processed_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False,
        comment="Number of papers successfully processed"
    )
    skipped_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False,
        comment="Number of papers skipped (already exists)"
    )
    error_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False,
        comment="Number of errors encountered"
    )
    
    # Configuration
    target_count: Mapped[int] = mapped_column(
        Integer, nullable=False,
        comment="Target number of papers to process"
    )
    
    # Status
    is_completed: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False,
        comment="Whether job is completed"
    )
    is_running: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False,
        comment="Whether job is currently running"
    )
    is_paused: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False,
        comment="Whether job is paused (user requested stop)"
    )
    
    # Real-time status tracking
    status_message: Mapped[str] = mapped_column(
        String(2000), nullable=True,
        comment="Current status message (e.g., 'Downloading file 1/30...')"
    )
    current_file: Mapped[str] = mapped_column(
        String(2000), nullable=True,
        comment="Current file being processed"
    )
    continuation_token: Mapped[str] = mapped_column(
        String(500), nullable=True,
        comment="Pagination continuation token for bulk search API"
    )
    
    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        comment="Job creation timestamp"
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(),
        comment="Last update timestamp"
    )
    completed_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="Job completion timestamp"
    )

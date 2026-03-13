"""
Event store for persisting and replaying pipeline events.
Supports event sourcing pattern for resumable chat.
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from sqlalchemy.orm import selectinload

from app.models.pipeline_tasks import (
    DBPipelineEvent,
    PipelineEventType
)


class PipelineEventStore:
    """
    Manages persistent storage of pipeline events.
    
    Responsibilities:
    - Save events in sequence
    - Retrieve events for replay
    - Support pagination for streaming
    - Maintain sequence ordering
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def save_event(
        self,
        task_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        sequence_number: Optional[int] = None
    ) -> DBPipelineEvent:
        """
        Save a new event to the store.
        
        Args:
            task_id: Task this event belongs to
            event_type: Event type (step, metadata, chunk, reasoning, error, done)
            event_data: Event payload
            sequence_number: Optional explicit sequence number (auto-generated if None)
        
        Returns:
            Created event record
        """
        # Auto-generate sequence number if not provided
        if sequence_number is None:
            sequence_number = await self._get_next_sequence(task_id)
        
        event = DBPipelineEvent(
            task_id=task_id,
            event_type=event_type,
            event_data=event_data,
            sequence_number=sequence_number
        )
        
        self.db.add(event)
        await self.db.commit()
        await self.db.refresh(event)
        
        return event
    
    async def save_events_batch(
        self,
        task_id: str,
        events: List[Dict[str, Any]]
    ) -> List[DBPipelineEvent]:
        """
        Save multiple events in a single transaction.
        
        Args:
            task_id: Task these events belong to
            events: List of event dictionaries with 'event_type' and 'event_data' keys
        
        Returns:
            List of created event records
        """
        start_sequence = await self._get_next_sequence(task_id)
        
        db_events = []
        for i, event_data in enumerate(events):
            event = DBPipelineEvent(
                task_id=task_id,
                event_type=event_data["event_type"],
                event_data=event_data["event_data"],
                sequence_number=start_sequence + i
            )
            db_events.append(event)
            self.db.add(event)
        
        await self.db.commit()
        
        # Refresh all events
        for event in db_events:
            await self.db.refresh(event)
        
        return db_events
    
    async def get_events(
        self,
        task_id: str,
        from_sequence: int = 0,
        limit: Optional[int] = None
    ) -> List[DBPipelineEvent]:
        """
        Get events for a task, optionally from a specific sequence.
        
        Args:
            task_id: Task identifier
            from_sequence: Starting sequence number (inclusive, default 0)
            limit: Maximum number of events to return
        
        Returns:
            List of events ordered by sequence number
        """
        query = (
            select(DBPipelineEvent)
            .where(
                and_(
                    DBPipelineEvent.task_id == task_id,
                    DBPipelineEvent.sequence_number >= from_sequence
                )
            )
            .order_by(DBPipelineEvent.sequence_number)
        )
        
        if limit:
            query = query.limit(limit)
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def get_all_events(self, task_id: str) -> List[DBPipelineEvent]:
        """
        Get all events for a task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            All events ordered by sequence number
        """
        return await self.get_events(task_id, from_sequence=0)
    
    async def get_latest_sequence(self, task_id: str) -> int:
        """
        Get the latest sequence number for a task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Latest sequence number (0 if no events exist)
        """
        result = await self.db.execute(
            select(func.max(DBPipelineEvent.sequence_number))
            .where(DBPipelineEvent.task_id == task_id)
        )
        max_seq = result.scalar()
        return max_seq if max_seq is not None else 0
    
    async def count_events(self, task_id: str) -> int:
        """
        Count total events for a task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Event count
        """
        result = await self.db.execute(
            select(func.count())
            .select_from(DBPipelineEvent)
            .where(DBPipelineEvent.task_id == task_id)
        )
        return result.scalar() or 0
    
    async def has_done_event(self, task_id: str) -> bool:
        """
        Check if task has a 'done' event.
        
        Args:
            task_id: Task identifier
        
        Returns:
            True if done event exists
        """
        result = await self.db.execute(
            select(func.count())
            .select_from(DBPipelineEvent)
            .where(
                and_(
                    DBPipelineEvent.task_id == task_id,
                    DBPipelineEvent.event_type == PipelineEventType.DONE
                )
            )
        )
        return (result.scalar() or 0) > 0
    
    async def get_event_types_summary(self, task_id: str) -> Dict[str, int]:
        """
        Get summary of event types for debugging.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Dictionary mapping event_type -> count
        """
        result = await self.db.execute(
            select(
                DBPipelineEvent.event_type,
                func.count(DBPipelineEvent.event_id)
            )
            .where(DBPipelineEvent.task_id == task_id)
            .group_by(DBPipelineEvent.event_type)
        )
        
        return {row[0]: row[1] for row in result.all()}
    
    async def _get_next_sequence(self, task_id: str) -> int:
        """
        Get next available sequence number for a task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Next sequence number
        """
        latest = await self.get_latest_sequence(task_id)
        return latest + 1
    
    async def delete_task_events(self, task_id: str) -> int:
        """
        Delete all events for a task (cleanup).
        
        Args:
            task_id: Task identifier
        
        Returns:
            Number of events deleted
        """
        count = await self.count_events(task_id)
        
        from sqlalchemy import delete
        
        await self.db.execute(
            delete(DBPipelineEvent).where(
                DBPipelineEvent.task_id == task_id
            )
        )
        await self.db.commit()
        
        return count

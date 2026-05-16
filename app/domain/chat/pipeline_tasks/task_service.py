"""
Service layer for managing pipeline task lifecycle.
Handles task creation, status tracking, and result caching.
"""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, desc
from sqlalchemy.orm import selectinload

from app.models.pipeline_tasks import (
    DBPipelineTask,
    PipelineTaskStatus,
    PipelinePhase
)


class PipelineTaskService:
    """
    Manages pipeline task lifecycle and state.
    
    Responsibilities:
    - Create new tasks
    - Update task status/progress
    - Track task phases
    - Cache results
    - Query task history
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_task(
        self,
        user_id: int,
        conversation_id: str,
        query: str,
        pipeline_type: str,
        filters: Optional[Dict[str, Any]] = None,
        client_message_id: Optional[str] = None
    ) -> DBPipelineTask:
        """
        Create a new pipeline task.
        
        Args:
            user_id: User who initiated the task
            conversation_id: Conversation this task belongs to
            query: User query text
            pipeline_type: Type of pipeline (hybrid, database, custom, etc.)
            filters: Optional search filters (year_min, year_max, etc.)
            client_message_id: Optional client-side message ID for deduplication
        
        Returns:
            Created task record
        """
        task_id = f"task_{uuid.uuid4().hex[:16]}"
        
        task = DBPipelineTask(
            task_id=task_id,
            user_id=user_id,
            conversation_id=conversation_id,
            query=query,
            pipeline_type=pipeline_type,
            filters=filters,
            client_message_id=client_message_id,
            status=PipelineTaskStatus.PENDING,
            progress_percent=0
        )
        
        self.db.add(task)
        await self.db.commit()
        await self.db.refresh(task)
        
        return task
    
    async def get_task(
        self,
        task_id: str,
        load_events: bool = False
    ) -> Optional[DBPipelineTask]:
        """
        Get task by ID.
        
        Args:
            task_id: Task identifier
            load_events: Whether to eagerly load events relationship
        
        Returns:
            Task record or None if not found
        """
        query = select(DBPipelineTask).where(DBPipelineTask.task_id == task_id)
        
        if load_events:
            query = query.options(selectinload(DBPipelineTask.events))
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def update_status(
        self,
        task_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update task status.
        
        Args:
            task_id: Task identifier
            status: New status (pending, running, completed, failed, cancelled)
            error_message: Optional error message for failed tasks
        """
        values: Dict[str, Any] = {"status": status}
        
        if status == PipelineTaskStatus.RUNNING and error_message is None:
            values["started_at"] = datetime.now()
        elif status in (PipelineTaskStatus.COMPLETED, PipelineTaskStatus.FAILED):
            values["completed_at"] = datetime.now()
        
        if error_message:
            values["error_message"] = error_message
        
        await self.db.execute(
            update(DBPipelineTask)
            .where(DBPipelineTask.task_id == task_id)
            .values(**values)
        )
        await self.db.commit()
    
    async def update_progress(
        self,
        task_id: str,
        phase: str,
        progress_percent: int
    ) -> None:
        """
        Update task progress.
        
        Args:
            task_id: Task identifier
            phase: Current pipeline phase (init, search, ranking, llm, etc.)
            progress_percent: Progress percentage (0-100)
        """
        await self.db.execute(
            update(DBPipelineTask)
            .where(DBPipelineTask.task_id == task_id)
            .values(current_phase=phase, progress_percent=progress_percent)
        )
        await self.db.commit()
    
    async def save_results(
        self,
        task_id: str,
        papers: Optional[List[Dict[str, Any]]] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
        response_text: Optional[str] = None
    ) -> None:
        """
        Cache task results.
        
        Args:
            task_id: Task identifier
            papers: Ranked papers from RAG pipeline
            chunks: Retrieved chunks
            response_text: Final LLM response
        """
        values: Dict[str, Any] = {}
        
        if papers is not None:
            values["result_papers"] = papers
        if chunks is not None:
            values["result_chunks"] = chunks
        if response_text is not None:
            values["response_text"] = response_text
        
        if values:
            await self.db.execute(
                update(DBPipelineTask)
                .where(DBPipelineTask.task_id == task_id)
                .values(**values)
            )
            await self.db.commit()

    async def update_conversation_id(
        self,
        task_id: str,
        conversation_id: str,
    ) -> None:
        """Relink task to a resolved conversation id (e.g. cloned conversation)."""

        await self.db.execute(
            update(DBPipelineTask)
            .where(DBPipelineTask.task_id == task_id)
            .values(conversation_id=conversation_id)
        )
        await self.db.commit()
    
    async def complete_task(
        self,
        task_id: str,
        message_id: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Mark task as completed or failed.
        
        Args:
            task_id: Task identifier
            message_id: Created assistant message ID
            error_message: Optional error message if task failed
        """
        status = PipelineTaskStatus.FAILED if error_message else PipelineTaskStatus.COMPLETED
        
        values: Dict[str, Any] = {
            "status": status,
            "completed_at": datetime.now(),
        }

        if not error_message:
            values["progress_percent"] = 100
        
        if message_id:
            values["message_id"] = message_id
        if error_message:
            values["error_message"] = error_message
        
        await self.db.execute(
            update(DBPipelineTask)
            .where(DBPipelineTask.task_id == task_id)
            .values(**values)
        )
        await self.db.commit()
    
    async def get_conversation_tasks(
        self,
        conversation_id: str,
        user_id: int,
        limit: int = 20
    ) -> List[DBPipelineTask]:
        """
        Get recent tasks for a conversation.
        
        Args:
            conversation_id: Conversation identifier
            user_id: User ID (for authorization)
            limit: Maximum number of tasks to return
        
        Returns:
            List of tasks ordered by creation time (newest first)
        """
        result = await self.db.execute(
            select(DBPipelineTask)
            .where(
                and_(
                    DBPipelineTask.conversation_id == conversation_id,
                    DBPipelineTask.user_id == user_id
                )
            )
            .order_by(desc(DBPipelineTask.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def cancel_task(self, task_id: str) -> None:
        """
        Cancel a running task.
        
        Args:
            task_id: Task identifier
        """
        await self.db.execute(
            update(DBPipelineTask)
            .where(DBPipelineTask.task_id == task_id)
            .values(
                status=PipelineTaskStatus.CANCELLED,
                completed_at=datetime.now()
            )
        )
        await self.db.commit()
    
    async def increment_retry_count(self, task_id: str) -> int:
        """
        Increment retry count for a failed task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            New retry count
        """
        task = await self.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        new_count = task.retry_count + 1
        
        await self.db.execute(
            update(DBPipelineTask)
            .where(DBPipelineTask.task_id == task_id)
            .values(retry_count=new_count)
        )
        await self.db.commit()
        
        return new_count

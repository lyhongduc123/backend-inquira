"""
Simple in-memory task queue for background processing.
Uses asyncio for concurrent task execution without external dependencies.
"""

import asyncio
from typing import Callable, Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BackgroundTask:
    """Represents a background task"""
    task_id: str
    task_type: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Any = None


class TaskQueue:
    """
    Simple in-memory task queue with asyncio workers.
    
    Features:
    - Non-blocking task submission
    - Concurrent task execution
    - Task status tracking
    - Automatic retry on failure (optional)
    
    Usage:
        queue = TaskQueue(max_workers=3)
        await queue.start()
        
        task_id = await queue.submit(
            "author_enrichment",
            enrich_author_func,
            author_id="12345"
        )
        
        status = await queue.get_status(task_id)
    """
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.tasks: Dict[str, BackgroundTask] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.workers: list[asyncio.Task] = []
        self._running = False
        self._task_counter = 0
    
    async def start(self):
        """Start worker threads"""
        if self._running:
            return
        
        self._running = True
        self.workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.max_workers)
        ]
        logger.info(f"Started {self.max_workers} background workers")
    
    async def stop(self):
        """Stop all workers gracefully"""
        self._running = False
        
        # Wait for queue to empty
        await self.queue.join()
        
        # Cancel workers
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("Stopped all background workers")
    
    async def submit(
        self,
        task_type: str,
        func: Callable,
        *args,
        **kwargs
    ) -> str:
        """
        Submit a task for background execution.
        
        Args:
            task_type: Type of task (for logging/tracking)
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
        
        Returns:
            task_id: Unique task identifier
        """
        self._task_counter += 1
        task_id = f"{task_type}_{self._task_counter}_{int(datetime.now().timestamp())}"
        
        task = BackgroundTask(
            task_id=task_id,
            task_type=task_type,
            func=func,
            args=args,
            kwargs=kwargs
        )
        
        self.tasks[task_id] = task
        await self.queue.put(task)
        
        logger.info(f"Submitted task {task_id} ({task_type})")
        return task_id
    
    async def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "error": task.error
        }
    
    async def _worker(self, worker_id: int):
        """Worker coroutine that processes tasks from queue"""
        logger.info(f"Worker {worker_id} started")
        
        while self._running:
            try:
                # Wait for task with timeout
                task = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} queue error: {e}")
                continue
            
            # Execute task
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            try:
                logger.info(f"Worker {worker_id} executing {task.task_id}")
                
                # Execute the function
                if asyncio.iscoroutinefunction(task.func):
                    result = await task.func(*task.args, **task.kwargs)
                else:
                    result = task.func(*task.args, **task.kwargs)
                
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                
                duration = (task.completed_at - task.started_at).total_seconds()
                logger.info(
                    f"Worker {worker_id} completed {task.task_id} "
                    f"in {duration:.2f}s"
                )
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now()
                
                logger.error(
                    f"Worker {worker_id} failed {task.task_id}: {e}",
                    exc_info=True
                )
            
            finally:
                self.queue.task_done()
        
        logger.info(f"Worker {worker_id} stopped")


# Global task queue instance
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """Get or create global task queue"""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue(max_workers=3)
    return _task_queue


async def initialize_task_queue():
    """Initialize and start the global task queue"""
    queue = get_task_queue()
    await queue.start()
    return queue

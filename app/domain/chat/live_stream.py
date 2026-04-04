"""In-memory live task stream broker for SSE delivery."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple


@dataclass
class _TaskStreamState:
    """Per-task live stream state."""

    subscribers: Set[asyncio.Queue] = field(default_factory=set)
    buffered_events: Deque[Tuple[str, Dict[str, Any]]] = field(default_factory=lambda: deque(maxlen=2000))
    terminal_event: Optional[Tuple[str, Dict[str, Any]]] = None


class LiveTaskStreamBroker:
    """Pub/sub broker for live SSE events keyed by pipeline task id."""

    def __init__(self) -> None:
        self._states: Dict[str, _TaskStreamState] = {}
        self._lock = asyncio.Lock()

    async def publish(self, task_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Publish a live event to all current subscribers and in-memory buffer."""
        async with self._lock:
            state = self._states.setdefault(task_id, _TaskStreamState())
            state.buffered_events.append((event_type, data))

            if event_type in ("done", "error"):
                state.terminal_event = (event_type, data)

            queues = list(state.subscribers)

        for queue in queues:
            try:
                queue.put_nowait((event_type, data))
            except asyncio.QueueFull:
                # Drop newest event for this slow subscriber; SSE is best-effort live stream.
                continue

    async def subscribe(
        self,
        task_id: str,
    ) -> Tuple[asyncio.Queue, List[Tuple[str, Dict[str, Any]]], Optional[Tuple[str, Dict[str, Any]]]]:
        """Subscribe to task events and get current buffered snapshot."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)

        async with self._lock:
            state = self._states.setdefault(task_id, _TaskStreamState())
            state.subscribers.add(queue)
            buffered = list(state.buffered_events)
            terminal = state.terminal_event

        return queue, buffered, terminal

    async def unsubscribe(self, task_id: str, queue: asyncio.Queue) -> None:
        """Unsubscribe a queue from a task stream."""
        async with self._lock:
            state = self._states.get(task_id)
            if not state:
                return

            state.subscribers.discard(queue)

            # Keep finished task state only while there are active subscribers.
            if not state.subscribers and state.terminal_event is not None:
                self._states.pop(task_id, None)


_live_task_stream_broker: Optional[LiveTaskStreamBroker] = None


def get_live_task_stream_broker() -> LiveTaskStreamBroker:
    """Get singleton live task stream broker."""
    global _live_task_stream_broker
    if _live_task_stream_broker is None:
        _live_task_stream_broker = LiveTaskStreamBroker()
    return _live_task_stream_broker

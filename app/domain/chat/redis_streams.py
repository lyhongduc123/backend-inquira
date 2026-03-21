"""
Redis Streams utilities for low-latency task event delivery.

This module is used by:
- Background worker: publish task events as they are generated
- SSE endpoint: consume task events in near real-time
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import redis.asyncio as redis  # pyright: ignore[reportMissingImports]

from app.core.config import settings
from app.extensions.logger import create_logger

logger = create_logger(__name__)

_STREAM_PREFIX = "chat:task:events:"
_STREAM_MAXLEN = 10000
_redis_client: Optional[redis.Redis] = None


@dataclass(frozen=True)
class RedisTaskEvent:
    """Task event payload read from Redis Stream."""

    stream_id: str
    event_type: str
    sequence: int
    event_data: Dict[str, Any]


def get_task_stream_key(task_id: str) -> str:
    """Build stream key for a pipeline task."""
    return f"{_STREAM_PREFIX}{task_id}"


async def get_redis_client() -> redis.Redis:
    """Get (or lazily create) shared async Redis client."""
    global _redis_client

    if _redis_client is None:
        _redis_client = redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            encoding="utf-8",
        )
    return _redis_client


async def close_redis_client() -> None:
    """Close shared Redis client if initialized."""
    global _redis_client

    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None


async def publish_task_event(
    task_id: str,
    event_type: str,
    sequence: int,
    event_data: Dict[str, Any],
) -> None:
    """Publish a single task event to Redis Stream."""
    client = await get_redis_client()
    key = get_task_stream_key(task_id)

    payload = {
        "event_type": event_type,
        "sequence": str(sequence),
        "event_data": json.dumps(event_data, ensure_ascii=False, default=str),
    }

    await client.xadd(key, payload, maxlen=_STREAM_MAXLEN, approximate=True)


async def read_task_events(
    task_id: str,
    last_stream_id: str = "$",
    count: int = 100,
    block_ms: int = 15000,
) -> List[RedisTaskEvent]:
    """
    Read task events from Redis Stream using XREAD.

    Args:
        task_id: Pipeline task id
        last_stream_id: Last consumed stream id. Use "$" to receive only new events.
        count: Max entries to read per call
        block_ms: Block time in milliseconds

    Returns:
        Parsed task events in arrival order.
    """
    client = await get_redis_client()
    key = get_task_stream_key(task_id)

    response = await client.xread(
        {key: last_stream_id},
        count=count,
        block=block_ms,
    )

    parsed_events: List[RedisTaskEvent] = []
    if not response:
        return parsed_events

    for _, entries in response:
        for stream_id, fields in entries:
            try:
                sequence = int(fields.get("sequence", "0"))
            except (TypeError, ValueError):
                sequence = 0

            raw_event_data = fields.get("event_data", "{}")
            try:
                event_data = json.loads(raw_event_data)
            except (TypeError, json.JSONDecodeError):
                logger.warning(
                    "Invalid event_data JSON in Redis stream for task %s stream_id=%s",
                    task_id,
                    stream_id,
                )
                event_data = {}

            parsed_events.append(
                RedisTaskEvent(
                    stream_id=stream_id,
                    event_type=str(fields.get("event_type", "")),
                    sequence=sequence,
                    event_data=event_data,
                )
            )

    return parsed_events

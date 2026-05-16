import json
import time
from typing import AsyncGenerator, List, Sequence, Dict, Any, Optional
from app.domain.papers.schemas import PaperMetadata
from app.extensions.stream import StreamEventType, stream_event

class EventType:
    SEARCHING = "searching"
    RANKING = "ranking"
    REASONING = "reasoning"
    SEARCHING_EXTERNAL = "searching_external"
    INGESTING = "ingesting"
    
    PAPER_METADATA = "papers_metadata"


class ChatEventEmitter:
    """Handles both event emission (streaming) and collection (for storage)"""
    
    def __init__(self):
        """Initialize with empty event collection."""
        self._collected_events: List[Dict[str, Any]] = []
        self._reasoning_buffer: str = ""  # Buffer to accumulate reasoning chunks
        self._pipeline_type: Optional[str] = None

    def set_pipeline_type(self, pipeline_type: Optional[str]) -> None:
        """Set current pipeline type so streamed/collected progress events can be mapped by frontend."""
        normalized = (pipeline_type or "").strip().lower()
        self._pipeline_type = normalized or None
    
    def get_collected_events(self) -> List[Dict[str, Any]]:
        """Get all collected events for storage."""
        return self._collected_events.copy()
    
    def _collect_event(self, event_type: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Internal method to collect event data for storage."""
        event = {
            "type": event_type,
            "timestamp": int(time.time() * 1000),
        }
        if self._pipeline_type:
            event["pipeline_type"] = self._pipeline_type
        if content:
            event["content"] = content
        if metadata:
            event["metadata"] = metadata
        
        self._collected_events.append(event)
    
    def _append_reasoning(self, content: str) -> None:
        """Append reasoning content to buffer instead of creating separate events."""
        self._reasoning_buffer += content
    
    def _finalize_reasoning(self) -> None:
        """Finalize reasoning by collecting the complete buffer as a single event."""
        if self._reasoning_buffer:
            self._collect_event(
                event_type="reasoning",
                content=self._reasoning_buffer
            )
            self._reasoning_buffer = ""  # Clear buffer after collecting
    async def emit_paper_metadata_events(
        self,
        papers: Sequence[PaperMetadata],
    ) -> AsyncGenerator[str, None]:
        """
        Emit paper metadata events for streaming to frontend.

        Frontend receives:
        event: paper_metadata
        data: {"type":"paper_metadata","papers":[{...},...]}

        Args:
            papers: List of PaperMetadata
        """
        paper_dicts = [paper.model_dump(mode='json', by_alias=True) for paper in papers]
        async for evt in stream_event(
            name=StreamEventType.METADATA,
            data=json.dumps({"type": EventType.PAPER_METADATA, "content": paper_dicts}),
        ):
            yield evt
            
    async def emit_conversation_event(
        self, 
        conversation_id: str, 
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Emit a conversation event for streaming to frontend.
        
        Args:
            conversation_id: UUID of the conversation
            title: Optional updated title
            metadata: Optional additional metadata
        """
        data = {"conversation_id": conversation_id}
        if title:
            data["title"] = title
        if metadata:
            data["metadata"] = json.dumps(metadata)
            
        async for evt in stream_event(
            name=StreamEventType.CONVERSATION,
            data=json.dumps({"type": "conversation", "content": data}),
        ):
            yield evt
            
    async def emit_searching_event(self, query: List[str]) -> AsyncGenerator[str, None]:
        """
        Emit a searching event and collect it for storage.

        Frontend receives:
        event: progress
        data: {"type":"searching","content":"Searching academic databases for ...","metadata":{"query":"..."}}

        Args:
            query: Search query string
        """
        # Collect for storage
        self._collect_event(
            event_type="searching",
            metadata={"queries": query}
        )
        
        # Emit for streaming
        async for evt in stream_event(
            name=StreamEventType.PROGRESS,
            data=json.dumps({
                "type": EventType.SEARCHING,
                "pipeline_type": self._pipeline_type,
                "content": f"Searching relevant works and documents",
                "metadata": {"queries": query},
            }),
        ):
            yield evt
            
    async def emit_ranking_event(
        self,
        total_papers: int,
        chunks: int,
    ) -> AsyncGenerator[str, None]:
        """
        Emit a ranking event and collect it for storage.

        Frontend receives:
        event: progress
        data: {"type":"ranking","content":"Ranking X out of Y retrieved papers...","metadata":{"total":Y,"ranked":X}}

        Args:
            total_papers: Total number of retrieved papers
            chunks: Number of chunks filtered so far
        """
        # Collect for storage
        self._collect_event(
            event_type="ranking",
            metadata={"total_papers": total_papers, "chunks": chunks}
        )
        
        # Emit for streaming
        async for evt in stream_event(
            name=StreamEventType.PROGRESS,
            data=json.dumps({
                "type": EventType.RANKING,
                "pipeline_type": self._pipeline_type,
                "content": f"Filtering {total_papers} retrieved papers by content relevance, quality, authors,...",
                "metadata": {"total_papers": total_papers, "chunks": chunks},
            }),
        ):
            yield evt
            
    async def emit_reasoning_event(self, content: str) -> AsyncGenerator[str, None]:
        """
        Emit a reasoning event and append it to the reasoning buffer.

        Frontend receives:
        event: progress
        data: {"type":"reasoning","content":"Reasoning..."}
        Args:
            content: Reasoning content chunk from LLM
        """
        # Append to reasoning buffer instead of creating separate events
        self._append_reasoning(content)
        
        # Emit for streaming
        async for evt in stream_event(
            name=StreamEventType.PROGRESS,
            data=json.dumps({
                "type": EventType.REASONING,
                "pipeline_type": self._pipeline_type,
                "content": content,
            }),
        ):
            yield evt
            
    async def emit_searching_external_event(self, source: str, query: str) -> AsyncGenerator[str, None]:
        """
        Emit a searching external event.

        Frontend receives:
        event: progress
        data: {"type":"searching_external","content":"Searching external source: ...","metadata":{"source":"...","query":"..."}}

        Args:
            source: Name of the external source being searched
            query: The search query being used
        """
        async for evt in stream_event(
            name=StreamEventType.PROGRESS,
            data=json.dumps({
                "type": EventType.SEARCHING_EXTERNAL,
                "pipeline_type": self._pipeline_type,
                "content": f"Searching external source: {source}",
                "metadata": {"source": source, "query": query},
            }),
        ):
            yield evt
            
    async def emit_ingesting_event(self, count: int) -> AsyncGenerator[str, None]:
        """
        Emit an ingesting event.

        Frontend receives:
        event: progress
        data: {"type":"ingesting","content":"Ingesting X papers from source: ...","metadata":{"source":"...","count":X}}

        Args:
            source: Name of the source being ingested
            count: Number of papers ingested so far
        """
        async for evt in stream_event(
            name=StreamEventType.PROGRESS,
            data=json.dumps({
                "type": EventType.INGESTING,
                "pipeline_type": self._pipeline_type,
                "content": f"Ingesting {count} papers",
                "metadata": {"count": count},
            }),
        ):
            yield evt
            
    async def emit_chunk_event(self, content: str) -> AsyncGenerator[str, None]:
        """
        Emit a chunk event.

        Frontend receives:
        event: chunk
        data: {"type":"chunk","content":"..."}
        Args:
            content: Chunk content
        """
        async for evt in stream_event(
            name=StreamEventType.CHUNK,
            data=json.dumps({
                "type": StreamEventType.CHUNK,
                "content": content,
            }),
        ):
            yield evt

    async def emit_done_event(self) -> AsyncGenerator[str, None]:
        """
        Emit a done event with final summary.

        Frontend receives:
        event: done
        data: {"type":"done"}

        Args:
            summary: Final summary text
        """
        async for evt in stream_event(
            name=StreamEventType.DONE,
            data=json.dumps({"type": StreamEventType.DONE}),
        ):
            yield evt

    async def emit_error_event(
        self,
        message: str,
        error_type: str = "unknown",
    ) -> AsyncGenerator[str, None]:
        """
        Emit an error event.

        Frontend receives:
        event: error
        data: {"type":"error","message":"...","error_type":"..."}

        Args:
            message: Error message
            error_type: Type of error (default "unknown")
        """
        async for evt in stream_event(
            name=StreamEventType.ERROR,
            data=json.dumps({
                "type": StreamEventType.ERROR,
                "content": message,
                "error_type": error_type,
            }),
        ):
            yield evt
import json
from typing import AsyncGenerator, List, Sequence
from app.papers.schemas import PaperMetadata
from app.extensions.stream import StreamEventType, stream_event


class EventType:
    SEARCHING = "searching"
    RANKING = "ranking"
    REASONING = "reasoning"
    
    PAPER_METADATA = "papers_metadata"


class ChatEventEmitter:
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
            data=json.dumps({"type": EventType.PAPER_METADATA, "papers": paper_dicts}),
        ):
            yield evt
            
    async def emit_searching_event(self, query: List[str]) -> AsyncGenerator[str, None]:
        """
        Emit a searching event.

        Frontend receives:
        event: progress
        data: {"type":"searching","content":"Searching academic databases for ...","metadata":{"query":"..."}}

        Args:
            query: Search query string
        """
        async for evt in stream_event(
            name=StreamEventType.PROGRESS,
            data=json.dumps({
                "type": EventType.SEARCHING,
                "content": f"Searching academic databases...",
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
        Emit a ranking event.

        Frontend receives:
        event: progress
        data: {"type":"ranking","content":"Ranking X out of Y retrieved papers...","metadata":{"total":Y,"ranked":X}}

        Args:
            total_papers: Total number of retrieved papers
            chunks: Number of chunks filtered so far
        """
        async for evt in stream_event(
            name=StreamEventType.PROGRESS,
            data=json.dumps({
                "type": EventType.RANKING,
                "content": f"Filtering {total_papers} retrieved papers by content relevance, quality, authors,...",
                "metadata": {"total_papers": total_papers, "chunks": chunks},
            }),
        ):
            yield evt
            
    async def emit_reasoning_event(self, content: str) -> AsyncGenerator[str, None]:
        """
        Emit a reasoning event.

        Frontend receives:
        event: progress
        data: {"type":"reasoning","content":"Reasoning..."}
        Args:
        """
        async for evt in stream_event(
            name=StreamEventType.PROGRESS,
            data=json.dumps({
                "type": EventType.REASONING,
                "content": content,
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
                "message": message,
                "error_type": error_type,
            }),
        ):
            yield evt


EventEmitter = ChatEventEmitter()

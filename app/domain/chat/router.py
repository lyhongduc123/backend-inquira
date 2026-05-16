"""
Chat router for handling chatbot interactions
"""
import asyncio
import time
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from typing import Container, Optional, TYPE_CHECKING
from sqlalchemy.ext.asyncio import AsyncSession
from .schemas import (
    ChatMessageRequest, 
    FeedbackRequest,
    FeedbackResponse,
    PaperDetailChatRequest,
    ChatSubmitRequest,
    ChatSubmitResponse,
    PipelineTaskResponse
)
from app.core.db.database import get_db_session
from app.extensions.stream import stream_event
from app.extensions.logger import create_logger
from app.auth.dependencies import get_current_user, get_current_user_or_anonymous
from app.models.users import DBUser
from app.core.responses import ApiResponse, success_response
from app.core.exceptions import InternalServerException, NotFoundException, ForbiddenException
from app.core.dependencies import get_container
from app.domain.chat.live_stream import get_live_task_stream_broker
from app.domain.chat.conversation_setup import resolve_conversation_for_user

if TYPE_CHECKING:
    from app.core.container import ServiceContainer


router = APIRouter()
logger = create_logger(__name__)


@router.post("/stream")
async def stream_message(
    http_request: Request,
    request: ChatMessageRequest,
    current_user: DBUser = Depends(get_current_user_or_anonymous),
    container: "ServiceContainer" = Depends(get_container)
) -> StreamingResponse:
    """
    Stream chat message response in real-time with citation tracking
    
    Returns Server-Sent Events (SSE) stream with:
    1. event: conversation - Conversation metadata (JSON)
    2. event: metadata - Paper metadata for all retrieved papers (JSON array)
    3. event: token - Each token as generated (JSON: {type, content})
    4. event: done - Completion with cited vs retrieved summary (JSON)
    
    The new streaming architecture provides:
    - Real-time token-by-token streaming
    - Paper metadata sent once at the start
    - Frontend validates citations against metadata
    - Separation of cited (4 papers) vs retrieved (20 papers)
    
    Frontend should:
    - Parse 'metadata' to get all available papers and cache them
    - Accumulate 'token' events to build response
    - Validate (cite:paper_id) markers against metadata during rendering
    - Use 'done' to organize papers into References (cited) and Related (not cited)
    
    - **query**: User's message/question
    - **conversation_id**: Optional ID of existing conversation
    """
    try:
        request_id = getattr(http_request.state, 'request_id', None)
        logger.info(
            f"Stream endpoint called by user {current_user.id}",
            extra={"user_id": current_user.id, "query_preview": request.query[:50], "request_id": request_id}
        )
        return StreamingResponse(
            container.chat_service.stream_research_pipeline(
                request=request,
                user_id=current_user.id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        logger.error(f"Stream endpoint error: {e}", exc_info=True)
        raise InternalServerException(f"Failed to stream message: {str(e)}")


@router.post("/agent", response_model=ApiResponse[ChatSubmitResponse])
async def submit_agent_chat_message(
    request: ChatSubmitRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user),
    container: "ServiceContainer" = Depends(get_container)
):
    """
    Agent Mode: Submit a chat message for async background processing using the specialized ChatAgentService.
    Identical interface to /submit. Returns a task_id to connect to /stream/{task_id}.
    """
    try:
        setup_result = await resolve_conversation_for_user(
            conversation_service=container.conversation_service,
            user_id=current_user.id,
            conversation_id=request.conversation_id,
            new_conversation_title=request.query[:100],
        )
        if not setup_result:
            raise NotFoundException(f"Conversation {request.conversation_id} not found")

        conversation_id = setup_result.conversation_id

        request_filters = (
            request.filters.model_dump(exclude_none=True)
            if request.filters
            else None
        )
        
        # Create pipeline task
        task = await container.pipeline_task_service.create_task(
            user_id=current_user.id,
            conversation_id=conversation_id,
            query=request.query,
            pipeline_type="agent",
            filters=request_filters,
            client_message_id=request.client_message_id
        )
        
        import asyncio
        asyncio.create_task(
            container.chat_agent_service.stream_agent_workflow(
                task_id=task.task_id,
                user_id=current_user.id,
                conversation_id=conversation_id,
                query=request.query,
                filters=request_filters or {}
            )
        )
        
        logger.info(f"Agent Chat task {task.task_id} submitted for user {current_user.id}")
        
        return success_response(
            data=ChatSubmitResponse(
                task_id=task.task_id,
                conversation_id=conversation_id,
                status="pending",
                message="Agent Task submitted successfully"
            )
        )
    
    except NotFoundException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to submit agent chat task: {e}", exc_info=True)
        raise InternalServerException(f"Failed to submit agent chat task: {str(e)}")


@router.get("/stream/{task_id}")
async def stream_task_events(
    task_id: str,
    from_sequence: int = 0,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user),
    container: "ServiceContainer" = Depends(get_container)
) -> StreamingResponse:
    """
    Stream live LLM completion events for a task.

    Notes:
    - Only completion streaming events are emitted via SSE (`chunk`, `done`, `error`)
    - No DB event sequence replay is used
    - `from_sequence` is ignored for backward compatibility
    
    Args:
        task_id: Task identifier
        from_sequence: Deprecated and ignored
    
    Returns:
        SSE stream of events
    """
    try:
        # Verify task exists and user has permission
        task = await container.pipeline_task_service.get_task(task_id)
        if not task:
            raise NotFoundException(f"Task {task_id} not found")
        
        if task.user_id != current_user.id:
            raise ForbiddenException("Access denied")
        
        async def event_generator():
            """Generate live SSE events from in-memory broker (no DB sequence replay)."""
            broker = get_live_task_stream_broker()
            queue, buffered_events, terminal_event = await broker.subscribe(task_id)
            last_emit_ts = time.monotonic()

            try:
                # Deliver buffered live events (if any)
                for event_type, event_data in buffered_events:
                    async for sse_chunk in stream_event(name=event_type, data=event_data):
                        yield sse_chunk
                    last_emit_ts = time.monotonic()

                # If terminal event already available, end immediately
                if terminal_event is not None:
                    event_type, event_data = terminal_event
                    async for sse_chunk in stream_event(name=event_type, data=event_data):
                        yield sse_chunk
                    return

                while True:
                    try:
                        event_type, event_data = await asyncio.wait_for(queue.get(), timeout=3.0)
                        async for sse_chunk in stream_event(name=event_type, data=event_data):
                            yield sse_chunk
                        last_emit_ts = time.monotonic()

                        if event_type in ("done", "error"):
                            return
                    except asyncio.TimeoutError:
                        updated_task = await container.pipeline_task_service.get_task(task_id)
                        if updated_task and updated_task.status in ("completed", "failed", "cancelled"):
                            if updated_task.status == "completed":
                                # Fallback for late subscribers: emit cached final response once.
                                if updated_task.response_text:
                                    async for sse_chunk in stream_event(
                                        name="chunk",
                                        data={"type": "chunk", "content": updated_task.response_text},
                                    ):
                                        yield sse_chunk

                                async for sse_chunk in stream_event(
                                    name="done",
                                    data={"type": "done", "status": "success", "message_id": updated_task.message_id},
                                ):
                                    yield sse_chunk
                            else:
                                error_msg = updated_task.error_message or "Task failed"
                                async for sse_chunk in stream_event(
                                    name="error",
                                    data={"message": error_msg},
                                ):
                                    yield sse_chunk
                            return

                        if time.monotonic() - last_emit_ts >= 3.0:
                            async for sse_chunk in stream_event(
                                name="ping",
                                data={"ts": int(time.time() * 1000)},
                            ):
                                yield sse_chunk
                            last_emit_ts = time.monotonic()
            finally:
                await broker.unsubscribe(task_id, queue)
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    except (NotFoundException, ForbiddenException) as e:
        raise e
    except Exception as e:
        logger.error(f"Stream task events error: {e}", exc_info=True)
        raise InternalServerException(f"Failed to stream task events: {str(e)}")


@router.get("/tasks/{task_id}", response_model=ApiResponse[PipelineTaskResponse])
async def get_task_status(
    task_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user),
    container: "ServiceContainer" = Depends(get_container)
):
    """
    Get task status and metadata.
    
    Use this endpoint to:
    - Check if task is complete before streaming
    - Get progress percentage
    - Retrieve error messages
    - Get cached results
    
    Returns:
        Task metadata including status, progress, and results
    """

    try:
        task = await container.pipeline_task_service.get_task(task_id)
        if not task:
            raise NotFoundException(f"Task {task_id} not found")
        
        if task.user_id != current_user.id:
            raise ForbiddenException("Access denied")
        
        return success_response(
            data=PipelineTaskResponse(**task.to_dict())
        )
    
    except (NotFoundException, ForbiddenException) as e:
        raise e
    except Exception as e:
        logger.error(f"Get task status error: {e}", exc_info=True)
        raise InternalServerException(f"Failed to get task status: {str(e)}")


@router.post("/test-stream")
@router.get("/test-stream")
async def test_stream(container: "ServiceContainer" = Depends(get_container)) -> StreamingResponse:
    """
    Test endpoint for frontend to verify SSE streaming with comprehensive markdown examples.
    
    Returns a complete showcase of markdown formatting including:
    - Headers (H1-H6)
    - Bold, italic, strikethrough
    - Lists (ordered, unordered, nested)
    - Code blocks with syntax highlighting
    - Inline code
    - Blockquotes
    - Links and images
    - Tables
    - Math equations (KaTeX)
    - Citations with paper metadata
    """
    
    return StreamingResponse(
        container.chat_service.test_streaming_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
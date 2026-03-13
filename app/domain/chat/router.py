"""
Chat router for handling chatbot interactions
"""
import asyncio
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from typing import Optional, TYPE_CHECKING
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
from .services import ChatService
from app.db.database import get_db_session
from app.extensions.stream import stream_event
from app.extensions.logger import create_logger
from app.auth.dependencies import get_current_user
from app.models.users import DBUser
from app.core.responses import ApiResponse, success_response
from app.core.exceptions import InternalServerException, NotFoundException, ForbiddenException
from app.core.dependencies import get_container

if TYPE_CHECKING:
    from app.core.container import ServiceContainer


router = APIRouter()
logger = create_logger(__name__)


@router.post("/stream")
async def stream_message(
    http_request: Request,
    request: ChatMessageRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user),
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
            f"Stream endpoint with citations called by user {current_user.id}",
            extra={"user_id": current_user.id, "query_preview": request.query[:50], "request_id": request_id}
        )
        return StreamingResponse(
            container.chat_service.stream_message_with_citations(
                request=request,
                user_id=current_user.id,
                db_session=db
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Stream endpoint error: {e}", exc_info=True)
        raise InternalServerException(f"Failed to stream message: {str(e)}")

@router.post("/stream/paper/{paper_id}")
async def stream_paper_detail(
    http_request: Request,
    paper_id: str,
    request: PaperDetailChatRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user),
    container: "ServiceContainer" = Depends(get_container)
) -> StreamingResponse:
    """
    Chat about a specific paper with full-text context.
    
    This endpoint enables deep-dive conversations about a single paper:
    - Retrieves full PDF/TEI content if available
    - Uses paper's chunks for precise context
    - Maintains conversation history specific to this paper
    - Auto-creates conversation on first message
    
    Returns Server-Sent Events (SSE) stream with:
    1. event: conversation - Conversation metadata (with conversation_type and primary_paper_id)
    2. event: paper - Full paper metadata
    3. event: chunk - Response text chunks
    4. event: done - Completion signal
    
    - **paper_id**: The paper's unique identifier
    - **query**: User's question about the paper
    - **conversation_id**: Optional ID of existing conversation (null = create new)
    """
    from app.domain.chat.paper_detail_service import PaperDetailChatService
    
    paper_chat_service = PaperDetailChatService(db_session=db)
    
    try:
        request_id = getattr(http_request.state, 'request_id', None)
        logger.info(
            f"Paper detail chat called by user {current_user.id} for paper {paper_id}",
            extra={"user_id": current_user.id, "paper_id": paper_id, "request_id": request_id}
        )
        
        # Verify paper exists
        paper = await container.paper_service.get_paper(paper_id)
        if not paper:
            raise NotFoundException(f"Paper {paper_id} not found")
        
        # Get or create conversation for this paper
        if request.conversation_id:
            conversation = await container.conversation_service.get_conversation(
                conversation_id=request.conversation_id,
                user_id=current_user.id
            )
            if not conversation:
                raise NotFoundException(f"Conversation {request.conversation_id} not found")
        else:
            # Auto-create conversation
            conversation = await container.conversation_service.get_or_create_paper_conversation(
                user_id=current_user.id,
                paper_id=paper_id,
                paper_title=paper.title
            )
        
        # Stream paper detail chat
        return StreamingResponse(
            paper_chat_service.stream_chat(
                paper_id=paper_id,
                query=request.query,
                conversation_id=conversation.conversation_id,
                user_id=current_user.id,
                model=request.model
            ),
            media_type="text/event-stream"
        )
    except NotFoundException as e:
        raise e
    except Exception as e:
        logger.error(f"Paper detail chat error: {e}", exc_info=True)
        raise InternalServerException(f"Failed to stream paper chat: {str(e)}")


# ==================== EVENT-DRIVEN ENDPOINTS (v2) ====================

@router.post("/submit", response_model=ApiResponse[ChatSubmitResponse])
async def submit_chat_message(
    request: ChatSubmitRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user),
    container: "ServiceContainer" = Depends(get_container)
):
    """
    Submit a chat message for async background processing (Event-Driven Architecture v2).
    
    This endpoint immediately returns a task_id without blocking.
    The pipeline executes in the background and clients stream events via /stream/{task_id}.
    
    Benefits:
    - User can reload page without cancelling pipeline
    - Supports reconnection and resume
    - Non-blocking API
    
    Workflow:
    1. POST /submit -> Get task_id
    2. GET /stream/{task_id} -> Stream events (reconnectable)
    3. GET /tasks/{task_id} -> Check status
    
    Returns:
        task_id: Unique identifier for tracking
        conversation_id: Conversation this belongs to
        status: "pending" initially
    """

    try:
        # Get or create conversation
        if request.conversation_id:
            conversation = await container.conversation_service.get_conversation(
                conversation_id=request.conversation_id,
                user_id=current_user.id
            )
            if not conversation:
                raise NotFoundException(f"Conversation {request.conversation_id} not found")
            conversation_id = request.conversation_id
        else:
            conversation = await container.conversation_service.create_conversation(
                user_id=current_user.id,
                title=request.query[:100]  # Use query preview as title
            )
            conversation_id = conversation.conversation_id
        
        # Create pipeline task
        task = await container.pipeline_task_service.create_task(
            user_id=current_user.id,
            conversation_id=conversation_id,
            query=request.query,
            pipeline_type=request.pipeline,
            filters=request.filters,
            client_message_id=request.client_message_id
        )
        
        # Submit task to background worker
        from app.workers.task_queue import get_task_queue
        await get_task_queue().submit_chat_task(
            task_id=task.task_id,
            user_id=current_user.id,
            conversation_id=conversation_id,
            query=request.query,
            pipeline_type=request.pipeline,
            filters=request.filters or {}
        )
        
        logger.info(f"Chat task {task.task_id} submitted for user {current_user.id}")
        
        return success_response(
            data=ChatSubmitResponse(
                task_id=task.task_id,
                conversation_id=conversation_id,
                status="pending",
                message="Task submitted successfully"
            )
        )
    
    except NotFoundException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to submit chat task: {e}", exc_info=True)
        raise InternalServerException(f"Failed to submit chat task: {str(e)}")


@router.get("/stream/{task_id}")
async def stream_task_events(
    task_id: str,
    from_sequence: int = 0,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user),
    container: "ServiceContainer" = Depends(get_container)
) -> StreamingResponse:
    """
    Stream pipeline events for a task (resumable, reconnectable).
    
    This endpoint supports resume-from-sequence for reconnection:
    - Client tracks last received sequence number
    - On reconnect, pass from_sequence to resume from that point
    - Server streams all events >= from_sequence
    
    Events:
    - step: Progress update (phase, progress_percent)
    - metadata: Paper metadata
    - chunk: Response text chunk
    - reasoning: Model reasoning
    - error: Error occurred
    - done: Pipeline completed
    
    Args:
        task_id: Task identifier
        from_sequence: Resume from this sequence number (default 0)
    
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
            """Generate SSE events from database"""
            last_sequence = from_sequence - 1
            
            while True:
                # Fetch new events
                events = await container.pipeline_event_store.get_events(
                    task_id=task_id,
                    from_sequence=last_sequence + 1,
                    limit=50
                )
                
                # Stream events
                for event in events:
                    event_data = {
                        "event_type": event.event_type,
                        "sequence": event.sequence_number,
                        **event.event_data
                    }
                    async for sse_chunk in stream_event(name=event.event_type, data=event_data):
                        yield sse_chunk
                    
                    last_sequence = event.sequence_number
                
                # Check if task is done
                if events and events[-1].event_type == "done":
                    logger.info(f"Task {task_id} streaming completed")
                    break
                
                # If no new events, check task status
                if not events:
                    updated_task = await container.pipeline_task_service.get_task(task_id)
                    if updated_task and updated_task.status in ("completed", "failed", "cancelled"):
                        # Task finished but no done event yet, create one
                        if updated_task.status == "completed":
                            async for sse_chunk in stream_event(name="done", data={"status": "success"}):
                                yield sse_chunk
                        else:
                            error_msg = updated_task.error_message or "Task failed"
                            async for sse_chunk in stream_event(name="error", data={"message": error_msg}):
                                yield sse_chunk
                        break
                    
                    # Wait before polling again
                    await asyncio.sleep(0.5)
        
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
async def test_stream():
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
    async def generate_test_stream():
        # Event 1: Conversation ID
        async for evt in stream_event(name="conversation", data={"conversation_id": "test-12345"}):
            yield evt
        await asyncio.sleep(0.1)
        
        # Event 2: Mock paper sources
        mock_papers = [
            {
                "paper_id": "test-paper-1",
                "title": "Attention Is All You Need",
                "authors": [{"name": "Vaswani et al."}],
                "publication_date": "2017-06-12",
                "venue": "NeurIPS",
                "url": "https://arxiv.org/abs/1706.03762",
                "citation_count": 50000,
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks."
            },
            {
                "paper_id": "test-paper-2",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "authors": [{"name": "Devlin et al."}],
                "publication_date": "2018-10-11",
                "venue": "NAACL",
                "url": "https://arxiv.org/abs/1810.04805",
                "citation_count": 40000,
                "abstract": "We introduce a new language representation model called BERT."
            }
        ]
        async for evt in stream_event(name="sources", data=mock_papers):
            yield evt
        await asyncio.sleep(0.1)
        
        # Event 3: Thought process
        async for evt in stream_event(name="thought", data="Analyzing research papers and generating comprehensive markdown response..."):
            yield evt
        await asyncio.sleep(0.1)
        
        # Event 4: Stream markdown content in chunks
        markdown_content = """# Comprehensive Markdown Test

This is a **complete showcase** of markdown formatting capabilities for the frontend.

## Headers

### H3 Header
#### H4 Header
##### H5 Header
###### H6 Header

## Text Formatting

This text includes **bold text**, *italic text*, ***bold and italic***, ~~strikethrough~~, and `inline code`.

## Lists

### Unordered List
- First item
- Second item
  - Nested item 1
  - Nested item 2
    - Deep nested item
- Third item

### Ordered List
1. First step
2. Second step
   1. Sub-step A
   2. Sub-step B
3. Third step

## Code Blocks

Here's a Python code example:

```python
def hello_world():
    \"\"\"A simple function demonstrating code syntax highlighting\"\"\"
    print("Hello, World!")
    return True

# Using list comprehension
squares = [x**2 for x in range(10)]
```

JavaScript example:

```javascript
const fetchData = async (url) => {
  const response = await fetch(url);
  return response.json();
};
```

## Blockquotes

> This is a blockquote.
> 
> It can span multiple lines and is often used for citations or important notes.
> 
> > Nested blockquotes are also supported.

## Links and Citations

You can read more about [Transformers on Wikipedia](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)).

According to recent research [1], attention mechanisms have revolutionized NLP.

## Tables

| Model | Parameters | Year | Performance |
|-------|-----------|------|-------------|
| GPT-2 | 1.5B | 2019 | Good |
| GPT-3 | 175B | 2020 | Excellent |
| GPT-4 | Unknown | 2023 | Outstanding |

## Math Equations

Inline math: The equation $E = mc^2$ is Einstein's famous formula.

Block equation:

$$
\\frac{\\partial L}{\\partial w} = \\frac{1}{n} \\sum_{i=1}^{n} (h_w(x_i) - y_i) \\cdot x_i
$$

Transformer attention mechanism:

$$
\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V
$$

## Task Lists

- [x] Implement streaming endpoint
- [x] Add markdown examples
- [ ] Test on frontend
- [ ] Add more edge cases

## Horizontal Rule

---

## Special Characters

Escaping special characters: \\* \\_ \\` \\[ \\]

## Citations and References

The transformer architecture [1] introduced multi-head attention, which was later refined in BERT [2]. Both papers demonstrate significant improvements over previous state-of-the-art models.

### References

[1] Vaswani et al. (2017). "Attention Is All You Need." *NeurIPS*.  
[2] Devlin et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers." *NAACL*.

---

**Note**: This test covers most common markdown elements. Your frontend should handle all of these gracefully!
"""
        
        # Split into chunks and stream with realistic delays
        chunk_size = 50  # characters per chunk
        for i in range(0, len(markdown_content), chunk_size):
            chunk = markdown_content[i:i + chunk_size]
            async for evt in stream_event(name="chunk", data=chunk):
                yield evt
            await asyncio.sleep(0.05)  # Simulate realistic streaming delay
        
        # Event 5: Done
        async for evt in stream_event(name="done", data=""):
            yield evt
    
    return StreamingResponse(
        generate_test_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
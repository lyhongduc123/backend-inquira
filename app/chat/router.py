"""
Chat router for handling chatbot interactions
"""
import asyncio
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.chat.schemas import (
    ChatMessageRequest, 
    FeedbackRequest,
    FeedbackResponse
)
from app.llm.schemas import CitationBasedResponse
from app.chat.services import ChatService
from app.db.database import get_db_session
from app.extensions.stream import stream_event
from app.extensions.logger import create_logger
from app.auth.dependencies import get_current_user
from app.models.users import DBUser
from app.core.responses import ApiResponse, success_response
from app.core.exceptions import InternalServerException

router = APIRouter()
logger = create_logger(__name__)


@router.post("/stream")
async def stream_message(
    http_request: Request,
    request: ChatMessageRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> StreamingResponse:
    """
    Stream chat message response in real-time with paper metadata
    
    Returns Server-Sent Events (SSE) stream with:
    1. event: sources - Paper metadata (JSON array)
    2. event: chunk - Response text chunks
    3. event: done - Completion signal
    
    Frontend should:
    - Parse 'sources' event to display citations
    - Accumulate 'chunk' events to build the response
    - Stop listening on 'done' event
    
    - **query**: User's message/question
    - **conversation_id**: Optional ID of existing conversation
    """
    chat_service = ChatService(db_session=db)
    try:
        request_id = getattr(http_request.state, 'request_id', None)
        logger.info(
            f"Stream endpoint called by user {current_user.id}",
            extra={"user_id": current_user.id, "query_preview": request.query[:50], "request_id": request_id}
        )
        return StreamingResponse(
            chat_service.stream_message(
                request=request,
                user_id=current_user.id,
                db_session=db
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Stream endpoint error: {e}", exc_info=True)
        raise InternalServerException(f"Failed to stream message: {str(e)}")


@router.post("/feedback", response_model=ApiResponse[FeedbackResponse])
async def submit_feedback(
    http_request: Request,
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> ApiResponse[FeedbackResponse]:
    """
    Submit feedback/rating for a chat message
    
    - **message_id**: ID of the message being rated
    - **rating**: Rating from 1-5
    - **comment**: Optional feedback comment
    """
    try:
        chat_service = ChatService(db_session=db)
        success = await chat_service.save_feedback(
            message_id=request.message_id,
            rating=request.rating,
            comment=request.comment
        )
        
        feedback_response = FeedbackResponse(
            success=success,
            message="Feedback submitted successfully" if success else "Failed to submit feedback"
        )
        
        request_id = getattr(http_request.state, 'request_id', None)
        return success_response(feedback_response, request_id=request_id)
    except Exception as e:
        logger.error(f"Feedback submission error: {e}", exc_info=True)
        raise InternalServerException(f"Failed to submit feedback: {str(e)}")


@router.post("/stream-with-tools")
async def stream_message_with_tools(
    http_request: Request,
    request: ChatMessageRequest,
    db: AsyncSession = Depends(get_db_session),
    current_user: DBUser = Depends(get_current_user)
) -> StreamingResponse:
    """
    Stream chat message response with AI tool calling support
    
    This endpoint enables the AI to autonomously use tools when needed:
    - **compare_papers**: Compare research papers when user asks for comparisons
    - **opinion_meter**: Analyze opinion distribution when user asks about consensus
    - **citation_analysis**: Analyze paper impact when user asks about influence
    - **research_trends**: Analyze trends when user asks about temporal patterns
    
    Returns Server-Sent Events (SSE) stream with:
    1. event: conversation - Conversation metadata
    2. event: phase - Processing phase updates
    3. event: thought - AI reasoning and analysis
    4. event: sources - Paper metadata
    5. event: tool_call - Tool execution events (start, end, error)
    6. event: chunk - Response text chunks
    7. event: done - Completion signal
    
    Tool events include:
    - tool_call_start: When AI decides to use a tool
    - tool_call_end: When tool execution completes with results
    - tool_call_error: When tool execution fails
    
    - **query**: User's message/question
    - **conversation_id**: Optional ID of existing conversation
    """
    chat_service = ChatService(db_session=db)
    try:
        request_id = getattr(http_request.state, 'request_id', None)
        logger.info(
            f"Tool-enabled stream called by user {current_user.id}",
            extra={"user_id": current_user.id, "query_preview": request.query[:50], "request_id": request_id}
        )
        return StreamingResponse(
            chat_service.stream_message_with_tools(
                request=request,
                user_id=current_user.id,
                db_session=db,
                enable_tools=True
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Tool-enabled stream error: {e}", exc_info=True)
        raise InternalServerException(f"Failed to stream with tools: {str(e)}")


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
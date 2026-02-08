"""
Test router for simulating streaming events
"""
import asyncio
from typing import AsyncGenerator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.extensions.citation_extractor import CitationExtractor
from app.extensions.stream import (
    stream_event,
    stream_token,
    stream_paper_metadata,
    stream_done,
)

router = APIRouter(prefix="/test", tags=["test"])


class TestQueryRequest(BaseModel):
    """Test query request"""
    query: str


def create_sse_message(name: str, data: dict) -> str:
    """Create a Server-Sent Event message"""
    import json
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


class MockPaper:
    """Mock paper object for metadata streaming"""
    def __init__(self, paper_id, title, authors, year, citations, doi, venue):
        self.paper_id = paper_id
        self.title = title
        self.authors = authors
        self.year = year
        self.citation_count = citations
        self.doi = doi
        self.venue = venue
        self.url = f"https://openalex.org/{paper_id}"
        self.relevance_score = 0.85
        self.publication_date = None


async def simulate_query_stream(query: str) -> AsyncGenerator[str, None]:
    """
    Simulate a realistic query stream with all event types
    Mimics the actual backend streaming behavior
    """
    
    # 1. Conversation created
    yield create_sse_message("conversation", {"conversation_id": "test-conv-123"})
    await asyncio.sleep(0.1)
    
    # 2. Phase: Query Understanding
    yield create_sse_message("phase", {
        "phase": "query_understanding",
        "message": "Analyzing your question to identify key concepts",
        "progress": 10,
        "current_step": 1,
        "total_steps": 5
    })
    await asyncio.sleep(0.3)
    
    yield create_sse_message("thought", {
        "type": "Analysis",
        "content": "Identified main topic: marine biodiversity and conservation technology",
        "metadata": {}
    })
    await asyncio.sleep(0.2)
    
    yield create_sse_message("thought", {
        "type": "Planning",
        "content": "Will search for papers on marine species conservation and AI applications in ecology",
        "metadata": {}
    })
    await asyncio.sleep(0.3)
    
    # 3. Phase: Search
    yield create_sse_message("phase", {
        "phase": "search",
        "message": "Searching academic databases for relevant papers",
        "progress": 30,
        "current_step": 2,
        "total_steps": 5
    })
    await asyncio.sleep(0.4)
    
    yield create_sse_message("thought", {
        "type": "Search",
        "content": "Querying OpenAlex for papers published in last 5 years",
        "metadata": {"source": "openalex"}
    })
    await asyncio.sleep(0.3)
    
    yield create_sse_message("thought", {
        "type": "Results",
        "content": "Found 150 potentially relevant papers",
        "metadata": {"count": 150}
    })
    await asyncio.sleep(0.3)
    
    # 4. Phase: Retrieval
    yield create_sse_message("phase", {
        "phase": "retrieval",
        "message": "Filtering and ranking papers by relevance",
        "progress": 50,
        "current_step": 3,
        "total_steps": 5
    })
    await asyncio.sleep(0.4)
    
    yield create_sse_message("thought", {
        "type": "Filtering",
        "content": "Applying relevance filters and citation thresholds",
        "metadata": {}
    })
    await asyncio.sleep(0.2)
    
    yield create_sse_message("thought", {
        "type": "Ranking",
        "content": "Selected top 25 papers based on semantic similarity",
        "metadata": {"selected": 25}
    })
    await asyncio.sleep(0.3)
    
    # 5. Phase: Analysis
    yield create_sse_message("phase", {
        "phase": "analysis",
        "message": "Analyzing paper contents and extracting insights",
        "progress": 70,
        "current_step": 4,
        "total_steps": 5
    })
    await asyncio.sleep(0.4)
    
    yield create_sse_message("thought", {
        "type": "Reading",
        "content": "Extracting key findings from selected papers",
        "metadata": {}
    })
    await asyncio.sleep(0.3)
    
    yield create_sse_message("analysis", {
        "type": "stats",
        "stats": {
            "papers_analyzed": 25,
            "chunks_retrieved": 87,
            "avg_relevance": 0.78,
            "citations": 342
        },
        "message": "Analysis complete"
    })
    await asyncio.sleep(0.3)
    
    yield create_sse_message("thought", {
        "type": "Synthesis",
        "content": "Combining insights from multiple sources",
        "metadata": {}
    })
    await asyncio.sleep(0.3)
    
    # 7. Stream paper metadata (using real stream function) - BEFORE generation
    mock_papers = [
        MockPaper(
            paper_id="W1234567890",
            title="Climate Change Impact on Marine Mammals: A Comprehensive Analysis",
            authors=["Dr. Sarah Johnson", "Prof. Michael Chen"],
            year=2023,
            citations=145,
            doi="10.1234/marine.2023.001",
            venue="Marine Biology Journal"
        ),
        MockPaper(
            paper_id="W0987654321",
            title="Machine Learning Applications in Ecological Conservation",
            authors=["Dr. Alice Thompson", "Dr. Robert Kim"],
            year=2024,
            citations=89,
            doi="10.1234/ecology.2024.002",
            venue="Nature Conservation"
        )
    ]
    async for evt in stream_paper_metadata(mock_papers, db=None):
        yield evt
    await asyncio.sleep(0.2)
    
    # 8. Phase: Generation
    async for evt in stream_event(name="phase", data={
        "phase": "generation",
        "message": "Generating comprehensive answer with citations",
        "progress": 90,
        "current_step": 5,
        "total_steps": 5
    }):
        yield evt
    await asyncio.sleep(0.3)
    
    yield create_sse_message("thought", {
        "type": "writing",
        "content": "Structuring answer with proper citations",
        "metadata": {}
    })
    await asyncio.sleep(0.2)
    
    # 9. Stream answer chunks with citation tracking (mimicking stream_message_with_citations)
    # Initialize citation tracker exactly like the real implementation
    
    # Simulating how LLMs stream text with citations in format (cite:paper_id)
    answer = """Marine biodiversity is facing unprecedented challenges due to climate change and human activities. Recent studies show that approximately 4,000 seal species are currently under threat from habitat loss and ocean warming (cite:W1234567890). This alarming statistic highlights the urgent need for conservation efforts.

Deep learning models have been increasingly applied to ecological monitoring and species conservation (cite:W0987654321). These AI systems can analyze satellite imagery and acoustic data to track animal populations with remarkable accuracy. For instance, neural networks can identify individual seals from aerial photographs with over 95% accuracy (cite:W1234567890).

The integration of machine learning in marine biology has enabled:
- Real-time population monitoring through autonomous systems
- Predictive models for migration patterns
- Early warning systems for ecosystem disruptions (cite:W0987654321)
- Automated classification of underwater species

However, challenges remain in deploying these technologies at scale. The computational cost, need for extensive labeled datasets, and variability in environmental conditions pose significant barriers (cite:W1234567890). Future research must focus on developing more robust and efficient models that can operate in diverse marine environments (cite:W0987654321)."""
    
    # Split into realistic streaming chunks (simulating token-by-token generation)
    # This mimics how OpenAI/LiteLLM streams tokens
    # Use character-by-character streaming to preserve newlines
    current_chunk = ""
    
    for char in answer:
        current_chunk += char
        
        # Stream chunk at word boundaries and newlines (realistic token grouping)
        if char in [" ", "\n", ".", ",", "!", "?", ":", ";"]:
            async for evt in stream_token(current_chunk):
                yield evt
            
            await asyncio.sleep(0.08)  # Realistic streaming delay
            current_chunk = ""
    
    # Stream any remaining text
    if current_chunk:
        async for evt in stream_token(current_chunk):
            yield evt
    
    
    # 10. Complete
    async for evt in stream_event(name="phase", data={
        "phase": "complete",
        "message": "Response generated successfully",
        "progress": 100,
        "current_step": 5,
        "total_steps": 5
    }):
        yield evt
    await asyncio.sleep(0.1)
    
    async for evt in stream_event(name="done", data={}):
        yield evt


@router.post("/stream")
async def test_stream(request: TestQueryRequest):
    """
    Test endpoint that simulates a realistic query stream
    
    Usage:
    ```
    POST /api/test/stream
    {
        "query": "test query"
    }
    ```
    
    This endpoint simulates all the streaming events that occur during
    a real query processing:
    - conversation: Conversation ID
    - phase: Processing phase updates (query_understanding, search, retrieval, analysis, generation, complete)
    - thought: Internal processing thoughts and updates
    - analysis: Statistics about the analysis
    - metadata: Paper metadata
    - chunk: Answer text chunks
    - done: Completion signal
    """
    return StreamingResponse(
        simulate_query_stream(request.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

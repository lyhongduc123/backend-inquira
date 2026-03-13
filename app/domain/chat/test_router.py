"""
Test router for simulating streaming events
"""

import asyncio
from datetime import datetime
from typing import AsyncGenerator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.extensions.stream import stream_event, StreamEventType
from app.extensions.logger import create_logger
from app.domain.papers.schemas import SJRMetadata
from app.domain.authors.schemas import AuthorMetadata
from .event_emitter import EventEmitter, EventType

router = APIRouter(prefix="/test", tags=["test"])
logger = create_logger(__name__)


class TestQueryRequest(BaseModel):
    """Test query request"""

    query: str
    conversation_id: str | None = None


async def simulate_realistic_stream(
    query: str, conversation_id: str | None = None
) -> AsyncGenerator[str, None]:
    """
    Simulate a realistic query stream matching the actual backend streaming behavior.

    Emits events in the same order and format as stream_message_with_citations:
    1. conversation - Conversation metadata
    2. progress (searching) - Searching academic databases
    3. progress (ranking) - Filtering and ranking papers
    4. metadata - Paper metadata for all retrieved papers
    5. progress (reasoning) - Generating response
    6. chunk - Response text chunks
    7. done - Completion signal
    """

    # 2. Emit searching event
    async for evt in EventEmitter.emit_searching_event(
        query=["machine learning", "neural networks", "deep learning"]
    ):
        yield evt
    await asyncio.sleep(0.5)

    # 3. Emit ranking event
    async for evt in EventEmitter.emit_ranking_event(total_papers=45, chunks=128):
        yield evt
    await asyncio.sleep(0.5)

    # 4. Emit paper metadata (simulating retrieved papers)
    mock_papers = []
    from app.domain.papers.schemas import PaperMetadata

    # Create mock papers with realistic data
    mock_papers.append(
        PaperMetadata(
            paper_id="W3177828909",
            title="Retrieval-Augmented Generation for Citation-Grounded Scientific Question Answering",
            abstract=(
                "This paper proposes a transformer-based retrieval-augmented generation "
                "framework for citation-grounded academic question answering."
            ),
            authors=[
                AuthorMetadata(
                    author_id="author_001",
                    name="Dr. Alice Nguyen",
                    h_index=42,
                    verified=True,
                ),
                AuthorMetadata(
                    author_id="author_002",
                    name="Prof. David Kim",
                    h_index=37,
                    verified=True,
                ),
            ],
            year=2024,
            publication_date=datetime(2024, 3, 15),
            venue="International Conference on Machine Learning (ICML)",
            journal="Journal of Artificial Intelligence Research",
            url="https://example.org/papers/rag-qa-2024",
            pdf_url="https://example.org/papers/rag-qa-2024.pdf",
            citation_count=128,
            influential_citation_count=35,
            reference_count=62,
            author_trust_score=0.89,
            institutional_trust_score=0.93,
            fwci=1.47,
            is_open_access=True,
            is_retracted=False,
            topics=[
                {"topic": "Retrieval-Augmented Generation", "score": 0.95},
                {"topic": "Scientific Question Answering", "score": 0.91},
            ],
            keywords=[
                {"keyword": "RAG", "score": 0.98},
                {"keyword": "transformer models", "score": 0.92},
                {"keyword": "citation grounding", "score": 0.89},
            ],
            relevance_score=0.94,
            ranking_scores={
                "semantic_similarity": 0.96,
                "citation_boost": 0.88,
            },
            sjr_data=SJRMetadata(
                title="Journal of Artificial Intelligence Research",
                sjr_score=2.345,
                quartile="Q1",
                h_index=210,
                data_year=2023,
            ),
        )
    )

    mock_papers.append(
        PaperMetadata(
            paper_id="W2963909066",
            title="Evaluating Large Language Models for Evidence-Based Scientific Reasoning",
            abstract=(
                "This study benchmarks large language models on evidence-based reasoning "
                "tasks using curated scientific datasets and citation-aware evaluation metrics."
            ),
            authors=[
                AuthorMetadata(
                    author_id="author_010",
                    name="Dr. Maria Gonzalez",
                    h_index=58,
                    verified=True,
                ),
                AuthorMetadata(
                    author_id="author_011",
                    name="Dr. Ethan Clarke",
                    h_index=44,
                    verified=True,
                ),
            ],
            year=2022,
            publication_date=datetime(2022, 11, 5),
            venue=None,
            journal="Nature Machine Intelligence",
            url="https://example.org/papers/llm-eval-2022",
            pdf_url=None,
            citation_count=542,
            influential_citation_count=120,
            reference_count=75,
            author_trust_score=0.94,
            institutional_trust_score=0.97,
            fwci=2.31,
            is_open_access=False,
            is_retracted=False,
            topics=[
                {"topic": "Large Language Models", "score": 0.97},
                {"topic": "Scientific Evaluation", "score": 0.90},
            ],
            keywords=[
                {"keyword": "LLM evaluation", "score": 0.95},
                {"keyword": "evidence reasoning", "score": 0.91},
                {"keyword": "benchmarking", "score": 0.88},
            ],
            relevance_score=0.89,
            ranking_scores={
                "semantic_similarity": 0.85,
                "citation_boost": 0.93,
            },
            sjr_data=SJRMetadata(
                title="Nature Machine Intelligence",
                sjr_score=14.221,
                quartile="Q1",
                h_index=365,
                data_year=2023,
            ),
        )
    )

    mock_papers.append(
        PaperMetadata(
            paper_id="W2964069224",
            title="Dynamic Citation-Aware Retrieval for Real-Time Academic Assistants",
            abstract=(
                "We introduce a dynamic retrieval framework that integrates citation graphs "
                "into real-time academic assistant systems to improve factual grounding."
            ),
            authors=[
                AuthorMetadata(
                    author_id="author_020",
                    name="Linh Tran",
                    h_index=18,
                    verified=False,
                ),
                AuthorMetadata(
                    author_id="author_021",
                    name="Arjun Patel",
                    h_index=22,
                    verified=False,
                ),
            ],
            year=2025,
            publication_date=datetime(2025, 1, 20),
            venue="ACL 2025",
            journal=None,
            url="https://example.org/papers/citation-aware-2025",
            pdf_url="https://example.org/papers/citation-aware-2025.pdf",
            citation_count=12,
            influential_citation_count=3,
            reference_count=41,
            author_trust_score=0.71,
            institutional_trust_score=0.79,
            fwci=1.12,
            is_open_access=True,
            is_retracted=False,
            topics=[
                {"topic": "Citation Networks", "score": 0.93},
                {"topic": "Academic Assistants", "score": 0.89},
            ],
            keywords=[
                {"keyword": "citation graph", "score": 0.90},
                {"keyword": "real-time retrieval", "score": 0.87},
                {"keyword": "RAG systems", "score": 0.92},
            ],
            relevance_score=0.96,
            ranking_scores={
                "semantic_similarity": 0.97,
                "recency_boost": 0.91,
            },
            sjr_data=SJRMetadata(
                title="Computational Linguistics Conference Proceedings",
                sjr_score=3.874,
                quartile="Q1",
                h_index=198,
                data_year=2024,
            ),
        )
    )

    # Emit paper metadata
    async for evt in EventEmitter.emit_paper_metadata_events(mock_papers):
        yield evt
    await asyncio.sleep(0.3)

    # 5. Emit reasoning event
    async for evt in EventEmitter.emit_reasoning_event(
        "Synthesizing information from retrieved papers to generate a comprehensive response"
    ):
        yield evt
    await asyncio.sleep(0.3)

    # 6. Stream response chunks (simulating LLM token streaming)
    response_text = f"""Based on the academic literature, here's a comprehensive overview of {query}:

## Transformer Architecture

The Transformer architecture, introduced by Vaswani et al. in their seminal work "Attention Is All You Need" (cite:W3177828909), revolutionized natural language processing by relying entirely on attention mechanisms. This approach eliminated the need for recurrent or convolutional layers, enabling parallel processing and better handling of long-range dependencies.

Key innovations of the Transformer include:

- **Self-attention mechanism**: Allows the model to weigh the importance of different parts of the input sequence
- **Multi-head attention**: Enables the model to attend to information from different representation subspaces
- **Positional encoding**: Provides sequence order information without recurrence

## BERT and Transfer Learning

Building on the Transformer architecture, BERT (Bidirectional Encoder Representations from Transformers) introduced by Devlin et al. (cite:W2964069224) demonstrated the power of pre-training deep bidirectional representations. BERT's key contributions include:

1. **Masked Language Modeling**: Pre-training objective that enables bidirectional context understanding
2. **Next Sentence Prediction**: Helps the model understand relationships between sentences
3. **Fine-tuning for downstream tasks**: Achieved state-of-the-art results on multiple NLP benchmarks

The success of BERT sparked a wave of transformer-based language models that continue to dominate NLP research.

## Deep Learning Fundamentals

The foundation for these advances was laid by earlier work in deep learning, such as ResNet (cite:W2963909066). The residual learning framework introduced skip connections that enabled training of much deeper networks, addressing the vanishing gradient problem. These architectural innovations have been crucial for building the large-scale models we see today.

## Current Impact

These papers have collectively shaped modern AI:
- The Transformer architecture is now the backbone of large language models like GPT and Claude
- BERT-style pre-training has become standard practice in NLP
- ResNet principles inform architecture design across domains

The field continues to evolve rapidly, with researchers building on these foundational works to create more capable and efficient models."""

    # Stream in realistic chunks (word by word with some grouping)
    words = response_text.split()
    current_chunk = ""

    for i, word in enumerate(words):
        current_chunk += word + " "

        # Stream chunks at natural boundaries (every 3-8 words, or at punctuation)
        if (i + 1) % (3 + (i % 6)) == 0 or word.endswith((".", "!", "?", ":", "\n")):
            async for evt in EventEmitter.emit_chunk_event(current_chunk.rstrip()):
                yield evt
            await asyncio.sleep(0.05)  # Realistic streaming delay
            current_chunk = ""

    # Stream any remaining text
    if current_chunk.strip():
        async for evt in EventEmitter.emit_chunk_event(current_chunk.rstrip()):
            yield evt

    await asyncio.sleep(0.2)

    # 7. Emit done event
    async for evt in EventEmitter.emit_done_event():
        yield evt


@router.post("/stream")
async def test_stream(request: TestQueryRequest) -> StreamingResponse:
    """
    Test endpoint that simulates the real streaming chat endpoint.

    This endpoint mimics `/api/v1/chat/stream` behavior for frontend testing without
    requiring database, LLM, or paper retrieval services.

    **Usage:**
    ```bash
    POST /api/v1/chat/test/stream
    {
        "query": "Explain transformer architecture in deep learning",
        "conversation_id": "optional-test-conv-id"
    }
    ```

    **Emitted Events (in order):**
    1. `conversation` - Conversation metadata with ID
    2. `progress` (searching) - Searching academic databases
    3. `progress` (ranking) - Filtering and ranking papers
    4. `metadata` - Paper metadata for retrieved papers
    5. `progress` (reasoning) - Generating response
    6. `chunk` - Response text chunks (streamed token-by-token)
    7. `done` - Completion signal

    **Features:**
    - Realistic timing delays between events
    - Mock paper metadata with real paper information
    - Citation markers in response text (cite:paper_id format)
    - Progress events matching real backend phases
    """
    logger.info(f"Test stream endpoint called with query: {request.query[:50]}...")

    return StreamingResponse(
        simulate_realistic_stream(request.query, request.conversation_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

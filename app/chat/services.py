"""
Chat service that orchestrates retrieval and LLM services for chatbot functionality
"""

import asyncio
import json
import time
from typing import AsyncGenerator, Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from app.llm import get_llm_service
from app.chat.schemas import ChatMessageRequest
from app.extensions.logger import create_logger
from app.extensions import (
    stream_event,
    get_stream_response_content,
    get_stream_response_reasoning,
    stream_done,
)
from app.extensions.prompt_filter import is_gibberish
from app.rag_pipeline.pipeline import Pipeline as RAGPipeline
from app.rag_pipeline.schemas import RAGResult
from app.conversations.service import ConversationService
from app.messages.service import MessageService
from app.chat.tool_mixin import ToolAwareChatMixin

from app.processor.services import transformer
from app.extensions.citation_extractor import CitationExtractor
from .event_emitter import EventEmitter

logger = create_logger(__name__)


class ChatService(ToolAwareChatMixin):
    """Service class for handling chat interactions with tool support"""

    def __init__(
        self,
        db_session: AsyncSession,
        rag_pipeline: Optional[RAGPipeline] = None,
        llm_service=None,
    ):
        """Initialize chat service with dependency injection.

        Args:
            db_session: Required database session
            rag_pipeline: Optional RAG pipeline (created if not provided)
            llm_service: Optional LLM service (singleton if not provided)
        """
        self.db_session = db_session
        self.llm_service = llm_service or get_llm_service()
        self.rag_pipeline = rag_pipeline or RAGPipeline(db_session=db_session)
        self.message_service = MessageService(db_session)

    async def stream_message_with_citations(
        self,
        request: ChatMessageRequest,
        user_id: int,
        db_session=None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat message with citation extraction.

        Args:
            request: Chat message request
            user_id: User ID
            db_session: Database session

        Yields:
            SSE events:
            - conversation: Conversation ID
            - retrieved: All retrieved paper IDs
            - metadata: Paper metadata for client caching
            - token: Each token as generated
            - citation: When a citation is detected (with claim, confidence)
            - done: Completion with cited vs retrieved paper lists
        """
        if not db_session:
            logger.error("Database session required for stream_message_with_citations")
            async for evt in stream_event(
                name="error", data={"error": "Database connection error"}
            ):
                yield evt
            return

        user_id = user_id
        conversation_service = ConversationService(db_session)
        conversation = await conversation_service.get_or_create_conversation(
            user_id=user_id,
            conversation_id=request.conversation_id,
            title=request.query,
        )

        should_add_user_message = True
        if request.is_retry and request.client_message_id:
            existing_message = await self.message_service.check_existing_message(
                conversation_id=conversation.conversation_id,
                client_message_id=request.client_message_id,
            )
            if existing_message:
                logger.info(
                    f"Retry detected - skipping duplicate user message for client_message_id: {request.client_message_id}"
                )
                should_add_user_message = False
    
        msg_id = None
        if should_add_user_message:
            msg_id = await conversation_service.add_message_to_conversation(
                conversation_id=conversation.conversation_id,
                user_id=user_id,
                message_text=request.query,
                role="user",
                client_message_id=request.client_message_id,
            )

        if is_gibberish(request.query):
            async for evt in self._handle_gibberish_input(
                conversation_service,
                conversation.conversation_id,
                user_id,
                request.query,
            ):
                yield evt
            return

        results = None
        progress_events = []  # Collect progress events for message_metadata
        try:
            async for result_or_evt, event_data in self._execute_rag_pipeline(
                request.query, filters=request.filter
            ):
                if isinstance(result_or_evt, str):
                    yield result_or_evt
                    if isinstance(event_data, dict) and event_data.get("type") in ["search_queries", "ranking"]:
                        if event_data["type"] == "search_queries":
                            progress_events.append({
                                "type": "searching",
                                "metadata": {"queries": event_data.get("queries", [])},
                                "timestamp": int(time.time() * 1000),
                            })
                        elif event_data["type"] == "ranking":
                            progress_events.append({
                                "type": "ranking",
                                "metadata": {
                                    "total_papers": event_data.get("total_papers", 0),
                                    "chunks": event_data.get("total_chunks", 0),
                                },
                                "timestamp": int(time.time() * 1000),
                            })
                elif isinstance(result_or_evt, RAGResult):
                    results = result_or_evt
        except Exception as e:
            logger.error(f"Error during RAG pipeline execution: {e}")
            async for evt in EventEmitter.emit_error_event(
                message="An error occurred while retrieving research papers. Please try again.",
                error_type="rag_pipeline_error",
            ):
                yield evt

            return

        if results is None or len(results.papers) == 0:
            async for evt in self._handle_no_results(
                conversation.conversation_id, user_id, conversation_service
            ):
                yield evt
            return

        context, chunk_papers = self._build_context_from_results(results)
        retrieved_paper_ids = [str(p.paper_id) for p in results.papers]
        papers_metadata = [
            transformer.ranked_paper_to_metadata(p) for p in results.papers
        ]
        paper_snapshots = [p.model_dump(mode="json") for p in papers_metadata]
        async for evt in EventEmitter.emit_paper_metadata_events(papers_metadata):
            yield evt

        assistant_response_chunks = []
        reasoning_chunks = []
        async for chunk_text in self.llm_service.stream_citation_based_response(
            query=request.query, context=context
        ):
            text = get_stream_response_content(chunk_text)
            reasoning_chunk = get_stream_response_reasoning(chunk_text)
            if reasoning_chunk and reasoning_chunk not in reasoning_chunks:
                async for evt in EventEmitter.emit_reasoning_event(reasoning_chunk):
                    yield evt
                
                # Store reasoning event with content (actual reasoning, not just a label)
                progress_events.append({
                    "type": "reasoning",
                    "content": reasoning_chunk,
                    "timestamp": int(time.time() * 1000),
                })
                reasoning_chunks.append(reasoning_chunk)
            if text is None:
                continue

            async for evt in EventEmitter.emit_chunk_event(text):
                yield evt

            assistant_response_chunks.append(text)

        if reasoning_chunks:
            full_reasoning = "".join(reasoning_chunks)
            logger.info(f"Collected reasoning content: {len(full_reasoning)} chars")

        # Extract citations from complete response for analytics
        full_response = "".join(assistant_response_chunks)
        cited_paper_ids = CitationExtractor.extract_citations_from_text(full_response)
        final_citations = CitationExtractor.group_citations_by_paper(full_response)

        # Build citation_details for database (for analytics)
        citation_details = {}
        for paper_id, details in final_citations.items():
            citation_details[paper_id] = {
                "claim": details["claim"],
                "confidence": details["confidence"],
                "chunk_ids": [],
                "context": f"Cited {len(details['claims'])} times",
            }

        try:
            await conversation_service.add_message_to_conversation(
                conversation_id=conversation.conversation_id,
                user_id=user_id,
                message_text=full_response,
                role="assistant",
                paper_ids=retrieved_paper_ids,
                paper_snapshots=paper_snapshots,
                progress_events=progress_events,
            )
        except Exception as e:
            logger.error(f"Failed to save messages: {e}")

        async for evt in EventEmitter.emit_done_event():
            yield evt

    async def _handle_gibberish_input(
        self,
        conversation_service: ConversationService,
        conversation_id: str,
        user_id: int,
        query: str,
    ) -> AsyncGenerator[str, None]:
        """Handle gibberish input with helpful introduction message"""
        logger.info(f"Gibberish detected: {query}")

        try:
            await conversation_service.add_message_to_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                message_text=query,
                role="user",
                auto_title=False,
            )
        except Exception as e:
            logger.error(f"Failed to save user message: {e}")

        intro_parts = [
            "Hello! I'm exegent, an academic research assistant.\n\n",
            "I'm here to help you explore and understand academic research papers! I can:\n\n",
            " **Search** through millions of research papers across all disciplines  \n",
            " **Analyze** and summarize complex scientific papers  \n",
            " **Find** relevant citations and evidence for your questions  \n",
            " **Compare** different research findings and methodologies  \n\n",
            "**How to get started:**\n\n",
            "Ask me clear research questions like:\n",
            '- "What are the latest findings on climate change?"\n',
            '- "How does machine learning improve medical diagnosis?"\n',
            '- "What are the ethical implications of AI?"\n\n',
            "**Tips for better results:**\n",
            "- Be specific about what you want to know\n",
            "- Use proper words and complete sentences\n",
            "- Ask about scientific topics, research areas, or academic questions\n\n",
            "Try asking me a research question, and I'll find and analyze relevant papers for you!",
        ]

        intro_message = "".join(intro_parts)
        async for evt in stream_event(name="chunk", data={"text": intro_message}):
            yield evt

        try:
            await conversation_service.add_message_to_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                message_text=intro_message,
                role="assistant",
                auto_title=False,
            )
        except Exception as e:
            logger.error(f"Failed to save assistant message: {e}")

        async for evt in stream_event(name="done", data=None):
            yield evt

    async def _handle_no_results(
        self,
        conversation_id: str,
        user_id: int,
        conversation_service: ConversationService,
    ) -> AsyncGenerator[str, None]:
        """Handle when no papers are found"""
        msg = """I couldn't find any relevant research papers for your question. This could be because:

1. The topic might be too specific or recent
2. There may be no academic papers published on this subject
3. The papers may be behind paywalls or not indexed in the databases I have access to.

Please try asking a different question or rephrase your current one."""

        try:
            await conversation_service.add_message_to_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                message_text=msg,
                role="assistant",
                auto_title=False,
            )
        except Exception as e:
            logger.error(f"Failed to save no results message: {e}")

        async for evt in EventEmitter.emit_chunk_event(msg):
            yield evt
        async for evt in EventEmitter.emit_done_event():
            yield evt

    async def _execute_rag_pipeline(
        self,
        query: str,
        max_subtopics: int = 3,
        per_subtopic_limit: int = 30,
        filters: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[
        tuple[str | RAGResult | None, List[str] | Dict[str, Any]], None
    ]:
        """Execute RAG pipeline and yield events"""
        results: Optional[RAGResult] = None
        subtopics_found: List[str] = []

        async for event in self.rag_pipeline.run(
            query,
            max_subtopics=max_subtopics,
            per_subtopic_limit=per_subtopic_limit,
            filters=filters,
        ):
            if event.type == "step":
                async for evt in stream_event(name="step", data=event.data):
                    yield evt, {"type": "step", "data": event.data}
            elif event.type == "search_queries":
                queries = event.data.get("queries", [])  # type: ignore
                async for evt in EventEmitter.emit_searching_event(queries):
                    yield evt, {"type": "search_queries", "queries": queries}
            elif event.type == "ranking":
                async for evt in EventEmitter.emit_ranking_event(
                    total_papers=event.data.get("total_papers", 0),  # type: ignore
                    chunks=event.data.get("total_chunks", 0),  # type: ignore
                ):
                    yield evt, {
                        "type": "ranking",
                        "total_papers": event.data.get("total_papers", 0), # type: ignore
                        "total_chunks": event.data.get("total_chunks", 0), # type: ignore
                    }
            elif event.type == "result":
                results = event.data  # type: ignore

        yield results, subtopics_found

    def _build_context_from_results(
        self, results: RAGResult
    ) -> tuple[str, Dict[str, Any]]:
        """
        Build LLM-friendly context from RAG results.

        Returns optimized text format instead of JSON to reduce tokens.
        Format: [1] Title | Authors | Year (Citations) | paper_id
                Section: section_name
                Content: chunk_text

        Returns:
            Tuple of (formatted_context_string, paper_id_to_dbpaper_mapping)
        """
        chunk_papers = {}
        context_entries = []

        # Map chunks to papers (extract .paper from RankedPaper)
        for chunk in results.chunks:
            chunk_paper_id = str(chunk.paper_id)
            if chunk_paper_id not in chunk_papers:
                ranked_paper = next(
                    (
                        rp
                        for rp in results.papers
                        if str(rp.paper.paper_id) == chunk_paper_id
                    ),
                    None,
                )
                if ranked_paper:
                    chunk_papers[chunk_paper_id] = ranked_paper.paper

        # Build optimized text context entries
        for idx, chunk in enumerate(results.chunks, 1):
            chunk_paper_id = str(chunk.paper_id)
            paper = chunk_papers.get(chunk_paper_id)

            if not paper:
                continue

            # Extract author names from paper_authors relationship
            author_names = []
            if hasattr(paper, "paper_authors") and paper.paper_authors:
                author_names = [
                    ap.author.name
                    for ap in paper.paper_authors
                    if ap.author and hasattr(ap.author, "name")
                ]
            authors_str = ", ".join(author_names[:3]) if author_names else "Unknown"
            if len(author_names) > 3:
                authors_str += f" et al. ({len(author_names)} total)"

            # Extract year
            year_str = (
                str(paper.publication_date.year) if paper.publication_date else "N/A"
            )

            # Build compact header
            entry = f"[{idx}] {paper.title}\n"
            entry += f"Paper ID: {chunk_paper_id}\n"
            entry += f"Authors: {authors_str} | Year: {year_str} | Citations: {paper.citation_count or 0}\n"

            # Add URLs if available
            if paper.url:
                entry += f"URL: {paper.url}\n"
            if paper.pdf_url:
                entry += f"PDF: {paper.pdf_url}\n"

            # Add section context
            if chunk.section_title:
                entry += f"Section: {chunk.section_title}\n"

            # Add chunk content
            entry += f"Content: {chunk.text}\n"

            context_entries.append(entry)

        # Join with double newline separator
        formatted_context = "\n\n".join(context_entries)

        return formatted_context, chunk_papers



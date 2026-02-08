"""
Chat service that orchestrates retrieval and LLM services for chatbot functionality
"""

import asyncio
import json
from typing import AsyncGenerator, Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from app.llm import get_llm_service
from app.chat.schemas import ChatMessageRequest
from app.extensions.logger import create_logger
from app.extensions import (
    stream_event,
    get_stream_response_content,
    get_stream_response_reasoning,
    stream_token,
    stream_paper_metadata,
    stream_done,
    StreamEventType,
)
from app.extensions.prompt_filter import is_gibberish
from app.rag_pipeline.pipeline import RAGPipeline
from app.rag_pipeline.schemas import RAGResult
from app.conversations.service import ConversationService
from app.chat.tool_mixin import ToolAwareChatMixin

from app.processor.services import transformer
from app.extensions.citation_extractor import CitationExtractor

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
            "- \"What are the latest findings on climate change?\"\n",
            "- \"How does machine learning improve medical diagnosis?\"\n",
            "- \"What are the ethical implications of AI?\"\n\n",
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

    async def _handle_no_results(self) -> AsyncGenerator[str, None]:
        """Handle when no papers are found"""
        async for evt in stream_event(
            name="thought",
            data={
                "type": "retrieval_failure",
                "content": "No relevant papers found in academic databases",
                "metadata": {
                    "reason": "no_results",
                    "suggestion": "try_different_query",
                },
            },
        ):
            yield evt

        error_message = """I couldn't find any relevant research papers for your question. This could be because:

1. The topic might be too specific or recent
2. There may be no academic papers published on this subject
3. The papers may be behind paywalls

Please try asking a different question or rephrase your current one."""

        async for evt in stream_event(name="chunk", data=error_message):
            yield evt
        async for evt in stream_event(name="done", data=None):
            yield evt

    async def _execute_rag_pipeline(
        self,
        query: str,
        max_subtopics: int = 3,
        per_subtopic_limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[tuple[str | RAGResult | None, List[str]], None]:
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
                    yield evt, subtopics_found
            elif event.type == "search_queries":
                # Handle new search_queries event
                queries = event.data.get("queries", [])  # type: ignore
                subtopics_found = queries  # Maintain backward compatibility
                async for evt in stream_event(name="search_queries", data=event.data):
                    yield evt, subtopics_found

                # Emit thought about query optimization
                async for evt in stream_event(
                    name="thought",
                    data={
                        "type": "query_optimization",
                        "content": f"Generated {len(queries)} optimized search queries to retrieve comprehensive and relevant academic papers",
                        "metadata": {
                            "search_queries": queries,
                            "strategy": "multi_query_search",
                        },
                    },
                ):
                    yield evt, subtopics_found
            elif event.type == "subtopics":
                # Backward compatibility for old event type
                subtopics_found = event.data.get("subtopics", [])  # type: ignore
                async for evt in stream_event(name="subtopics", data=event.data):
                    yield evt, subtopics_found

                # Emit thought about subtopic breakdown
                async for evt in stream_event(
                    name="thought",
                    data={
                        "type": "query_decomposition",
                        "content": f"Breaking down into {len(subtopics_found)} focused research areas to ensure comprehensive coverage",
                        "metadata": {
                            "subtopics": subtopics_found,
                            "strategy": "multi-angle_search",
                        },
                    },
                ):
                    yield evt, subtopics_found
            elif event.type == "result":
                results = event.data  # type: ignore
            else:
                async for ent in stream_event(
                    name=event.type, data=event.data  
                ):
                    yield ent, subtopics_found

        yield results, subtopics_found

    def _build_context_from_results(
        self, results: RAGResult
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Build context from RAG results"""
        context = []
        chunk_papers = {}

        # Map chunks to papers
        for chunk in results.chunks:
            chunk_paper_id = str(chunk.paper_id)
            if chunk_paper_id not in chunk_papers:
                paper = next(
                    (p for p in results.papers if str(p.paper_id) == chunk_paper_id),
                    None,
                )
                if paper:
                    chunk_papers[chunk_paper_id] = paper

        # Build context entries
        for chunk in results.chunks:
            chunk_paper_id = str(chunk.paper_id)
            paper = chunk_papers.get(chunk_paper_id)
            if paper:
                context.append(
                    {
                        "title": paper.title,
                        "authors": paper.authors or [],
                        "abstract": paper.abstract,
                        "year": (
                            paper.publication_date.year
                            if paper.publication_date
                            else None
                        ),
                        "url": paper.url,
                        "pdf_url": paper.pdf_url,
                        "citationCount": paper.citation_count or 0,
                        "paper_id": chunk_paper_id,
                        "chunk_text": chunk.text,
                        "section": chunk.section_title or "Unknown Section",
                    }
                )

        return context, chunk_papers

    async def _emit_analysis_events(
        self,
        results: RAGResult,
        context: List[Dict[str, Any]],
        chunk_papers: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Emit analysis and sources events"""
        # Emit context analysis thought
        async for evt in stream_event(
            name="thought",
            data={
                "type": "context_preparation",
                "content": f"Preparing {len(context)} contextual excerpts from papers, prioritizing sections most relevant to your question",
                "metadata": {
                    "context_pieces": len(context),
                    "unique_papers": len(chunk_papers),
                    "sections_identified": len(
                        [c for c in context if c.get("section") != "Unknown Section"]
                    ),
                },
            },
        ):
            yield evt

        # Emit sources
        async for evt in stream_event(
            name="sources", data=transformer.batch_paper_to_dicts(results.papers)
        ):
            yield evt

        # Provide retrieval analysis
        citation_stats = {
            "high_impact": len(
                [p for p in results.papers if (p.citation_count or 0) > 100]
            ),
            "medium_impact": len(
                [p for p in results.papers if 10 < (p.citation_count or 0) <= 100]
            ),
            "recent": len(
                [
                    p
                    for p in results.papers
                    if p.publication_date and p.publication_date.year >= 2020
                ]
            ),
        }

        async for evt in stream_event(
            name="analysis",
            data={
                "type": "source_quality",
                "stats": citation_stats,
                "message": f"Retrieved {citation_stats['high_impact']} highly-cited papers, {citation_stats['recent']} recent publications",
            },
        ):
            yield evt

    async def _save_conversation_messages(
        self,
        conversation_service: ConversationService,
        conversation_id: str,
        user_id: int,
        user_message: str,
        assistant_message: str,
        paper_ids: List[str],
        auto_title_user: bool = True,
    ):
        """Save user and assistant messages to conversation"""
        try:
            await conversation_service.add_message_to_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                message_text=user_message,
                role="user",
                auto_title=auto_title_user,
            )
            logger.info(f"User message added to conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Failed to save user message: {e}")

        try:
            await conversation_service.add_message_to_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                message_text=assistant_message,
                role="assistant",
                auto_title=False,
                paper_ids=paper_ids,
            )
            logger.info(f"Assistant message added to conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Failed to save assistant message: {e}")

    async def stream_message(
        self,
        request: ChatMessageRequest,
        user_id: int,
        db_session=None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat message response with full paper processing and vector search
        Enhanced with detailed thought process and processing phase tracking

        This method performs the complete RAG pipeline with transparent thought exposure:
        1. Get or create conversation
        2. Break down user question into subtopics (with analysis thoughts)
        3. Search and retrieve papers (with retrieval thoughts)
        4. Auto-process papers (with processing thoughts)
        5. Use vector search on chunks (with ranking thoughts)
        6. Stream LLM response with reasoning (with generation thoughts)
        7. Save message and update conversation

        Args:
            request: Chat message request containing user query
            user_id: Optional user ID for personalization
            db_session: Database session for caching and vector search

        Yields:
            Response chunks in Server-Sent Events (SSE) format:
            - event: conversation, data: JSON object with conversation_id
            - event: phase, data: JSON object with phase info {phase, message, progress}
            - event: thought, data: JSON object with LLM reasoning {type, content, metadata}
            - event: step, data: string with current step description
            - event: subtopics, data: JSON array of analyzed subtopics
            - event: sources, data: JSON array of paper metadata
            - event: analysis, data: JSON object with retrieval/processing analysis
            - event: chunk, data: text chunks of the response
            - event: done, data: empty (signals completion)
        """
        if not db_session:
            logger.error("Database session required for stream_message")
            error_message = "Database connection error. Please try again."
            yield f"event: chunk\ndata: {json.dumps(error_message)}\n\n"
            yield "event: done\ndata: \n\n"
            return

        user_id = user_id
        conversation_service = ConversationService(db_session)
        conversation = await conversation_service.get_or_create_conversation(
            user_id=user_id,
            conversation_id=request.conversation_id,
            title=request.query,
        )
        yield f"event: conversation\ndata: {json.dumps({'conversation_id': conversation.id})}\n\n"

        # Check for gibberish input BEFORE emitting any phase events
        if is_gibberish(request.query):
            async for evt in self._handle_gibberish_input(
                conversation_service, conversation.id, user_id, request.query
            ):
                yield evt
            return

        # Phase 1: Searching/Querying
        async for evt in stream_event(
            name="thought",
            data={
                "type": "searching",
                "content": f"Searching academic databases for: '{request.query}'",
                "metadata": {"query": request.query},
            },
        ):
            yield evt

        results: Optional[RAGResult] = None
        async for result_or_evt, _ in self._execute_rag_pipeline(request.query):
            if isinstance(result_or_evt, str):
                yield result_or_evt
            elif isinstance(result_or_evt, RAGResult):
                results = result_or_evt

        if results is None:
            logger.error("RAG pipeline did not return any results")
            error_message = (
                "An error occurred while retrieving research papers. Please try again."
            )
            async for evt in stream_event(name="chunk", data=error_message):
                yield evt
            async for evt in stream_event(name="done", data=None):
                yield evt
            return

        if len(results.papers) == 0:
            async for evt in self._handle_no_results():
                yield evt
            return

        # Phase 2: Ranking/Filtering
        async for evt in stream_event(
            name="thought",
            data={
                "type": "ranking",
                "content": f"Found {len(results.papers)} papers, filtering to {len(results.chunks)} most relevant sections",
                "metadata": {
                    "total_papers": len(results.papers),
                    "total_chunks": len(results.chunks),
                },
            },
        ):
            yield evt

        logger.info(f"Found {len(results.chunks)} relevant chunks via vector search")

        context, chunk_papers = self._build_context_from_results(results)

        # Emit analysis events
        async for evt in self._emit_analysis_events(results, context, chunk_papers):
            yield evt

        # Phase 3: Thinking (LLM reasoning and generation)
        async for evt in stream_event(
            name="thought",
            data={
                "type": "thinking",
                "content": "Synthesizing insights from papers with evidence-based reasoning",
                "metadata": {
                    "context_chunks": len(context),
                    "unique_papers": len(chunk_papers),
                },
            },
        ):
            yield evt

        assistant_response_chunks = []
        async for chunk_text in self.llm_service.stream_citation_based_response(
            query=request.query, context=context
        ):
            text = get_stream_response_content(chunk_text)
            if text is None:
                continue

            newlines = 0
            if text.strip().startswith("#"):
                newlines = 2
            chunk_data = {"text": text, "newlines": newlines}
            assistant_response_chunks.append(text)
            async for chunk_event in stream_event(name="chunk", data=chunk_data):
                yield chunk_event

        full_response = "".join(assistant_response_chunks)
        await self._save_conversation_messages(
            conversation_service,
            conversation.id,
            user_id,
            request.query,
            full_response,
            [str(p.paper_id) for p in results.papers],
        )

        async for done_event in stream_event(name="done", data=None):
            yield done_event

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

        # Emit conversation ID
        yield f"event: conversation\ndata: {json.dumps({'conversation_id': conversation.id})}\n\n"

        # Check for gibberish
        if is_gibberish(request.query):
            async for evt in self._handle_gibberish_input(
                conversation_service, conversation.id, user_id, request.query
            ):
                yield evt
            return

        # Phase 1: Searching
        async for evt in stream_event(
            name="thought",
            data={
                "type": "searching",
                "content": f"Searching academic databases for: '{request.query}'",
                "metadata": {"query": request.query},
            },
        ):
            yield evt

        # Execute RAG pipeline
        results: Optional[RAGResult] = None
        try:
            async for result_or_evt, _ in self._execute_rag_pipeline(
                request.query, filters=request.filter
            ):
                if isinstance(result_or_evt, str):
                    yield result_or_evt
                elif isinstance(result_or_evt, RAGResult):
                    results = result_or_evt
        except Exception as e:
            logger.error(f"Error during RAG pipeline execution: {e}")
            async for evt in stream_event(
                name="error",
                data={"error": "An error occurred while retrieving research papers."},
            ):
                yield evt
            return

        if results is None or len(results.papers) == 0:
            async for evt in self._handle_no_results():
                yield evt
            return

        # Build context
        context, chunk_papers = self._build_context_from_results(results)

        # Stream retrieved paper IDs
        retrieved_paper_ids = [str(p.paper_id) for p in results.papers]

        # Stream paper metadata for client caching (with SJR enrichment)
        async for evt in stream_paper_metadata(results.papers, self.db_session):
            yield evt

        # Phase 2: Ranking
        async for evt in stream_event(
            name="thought",
            data={
                "type": "ranking",
                "content": f"Filtered to {len(context)} most relevant sections from {len(chunk_papers)} papers",
                "metadata": {
                    "context_chunks": len(context),
                    "unique_papers": len(chunk_papers),
                },
            },
        ):
            yield evt

        # Emit analysis events
        async for evt in self._emit_analysis_events(results, context, chunk_papers):
            yield evt

        # Phase 3: Thinking
        async for evt in stream_event(
            name="thought",
            data={
                "type": "thinking",
                "content": "Analyzing papers and generating response with citations",
                "metadata": {"approach": "evidence_based_synthesis"},
            },
        ):
            yield evt

        # Stream LLM response
        assistant_response_chunks = []
        reasoning_chunks = []
        async for chunk_text in self.llm_service.stream_citation_based_response(
            query=request.query, context=context
        ):
            text = get_stream_response_content(chunk_text)
            reasoning_chunk = get_stream_response_reasoning(chunk_text)
            if reasoning_chunk and reasoning_chunk not in reasoning_chunks:
                async for evt in stream_event(
                    name="reasoning", data={"text": reasoning_chunk}
                ):
                    yield evt

                reasoning_chunks.append(reasoning_chunk)
            if text is None:
                continue

            async for evt in stream_token(text):
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
            await self._save_conversation_messages(
                conversation_service,
                conversation.id,
                user_id,
                request.query,
                full_response,
                retrieved_paper_ids,
                auto_title_user=True,
            )

            logger.info(
                f"Message saved: {len(cited_paper_ids)} cited out of {len(retrieved_paper_ids)} retrieved"
            )
        except Exception as e:
            logger.error(f"Failed to save messages: {e}")

        # Stream done event with summary
        async for evt in stream_done(
            cited_paper_ids=cited_paper_ids,
            retrieved_paper_ids=retrieved_paper_ids,
            metadata={
                "total_citations": len(final_citations),
                "response_length": len(full_response),
                "conversation_id": conversation.id,
            },
        ):
            yield evt

    async def save_feedback(
        self, message_id: int, rating: int, comment: Optional[str] = None
    ) -> bool:
        """
        Save user feedback for a chat message

        Args:
            message_id: ID of the message to provide feedback for
            rating: Rating value (1-5 stars)
            comment: Optional text comment with additional feedback

        Returns:
            True if feedback was saved successfully, False otherwise
        """
        # TODO: Implement database persistence for feedback
        # Should save to a Feedback table with message_id, rating, comment, timestamp
        return True

    async def stream_message_with_tools(
        self,
        request: ChatMessageRequest,
        user_id: int,
        db_session=None,
        enable_tools: bool = True,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat message response with tool calling support

        This enhanced method allows the AI to autonomously use tools when needed:
        - compare_papers: Compare multiple papers across different aspects
        - opinion_meter: Analyze opinion distribution on a topic
        - citation_analysis: Analyze citation impact of papers
        - research_trends: Analyze research trends over time

        Args:
            request: Chat message request
            user_id: User ID
            db_session: Database session
            enable_tools: Whether to enable tool calling

        Yields:
            SSE events including tool_call events for tool execution
        """
        if not db_session:
            logger.error("Database session required for stream_message_with_tools")
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
        yield f"event: conversation\ndata: {json.dumps({'conversation_id': conversation.id})}\n\n"

        # Check for gibberish
        if is_gibberish(request.query):
            async for evt in self._handle_gibberish_input(
                conversation_service, conversation.id, user_id, request.query
            ):
                yield evt
            return

        # Phase 1: Searching
        async for evt in stream_event(
            name="thought",
            data={
                "type": "searching",
                "content": f"Searching for papers on: '{request.query}'",
                "metadata": {"query": request.query},
            },
        ):
            yield evt

        results: Optional[RAGResult] = None
        async for result_or_evt, _ in self._execute_rag_pipeline(request.query):
            if isinstance(result_or_evt, str):
                yield result_or_evt
            elif isinstance(result_or_evt, RAGResult):
                results = result_or_evt

        if results is None or len(results.papers) == 0:
            async for evt in self._handle_no_results():
                yield evt
            return

        # Build context
        context, chunk_papers = self._build_context_from_results(results)

        # Phase 2: Ranking
        async for evt in stream_event(
            name="thought",
            data={
                "type": "ranking",
                "content": f"Filtered to {len(context)} relevant sections from {len(chunk_papers)} papers",
                "metadata": {"papers": len(chunk_papers), "chunks": len(context)},
            },
        ):
            yield evt

        # Emit sources
        async for evt in stream_event(
            name="sources", data=transformer.batch_paper_to_dicts(results.papers)
        ):
            yield evt

        # Phase 3: Thinking (with tools)
        async for evt in stream_event(
            name="thought",
            data={
                "type": "thinking",
                "content": "Analyzing papers and determining if tools are needed",
                "metadata": {"tools_enabled": enable_tools},
            },
        ):
            yield evt

        # Build messages for LLM
        system_message = """You are an expert research assistant with access to powerful tools. You can:

1. **compare_papers**: Compare multiple papers when users ask about differences, similarities, or want to see papers side-by-side
2. **opinion_meter**: Analyze opinion distribution when users ask about consensus, controversy, or different viewpoints
3. **citation_analysis**: Analyze paper impact when users ask about influence or citations
4. **research_trends**: Analyze trends when users ask about growth, patterns, or temporal analysis

**When to use tools:**
- User asks for comparison: Use compare_papers
- User asks about opinions/consensus: Use opinion_meter  
- User asks about impact/influence: Use citation_analysis
- User asks about trends/patterns: Use research_trends

**Response style:**
- Use natural, conversational academic language
- Cite sources with inline citations: (cite:paper_id)
- Use Markdown formatting for clarity
- Integrate tool results naturally into your response

**Available papers:** You have access to paper IDs from the retrieved papers. Use these IDs when calling tools."""

        # At this point results is guaranteed to be RAGResult, not None
        paper_list = [{"id": str(p.paper_id), "title": p.title} for p in results.papers]
        user_prompt = f"""Question: {request.query}

Available papers (use these IDs for tools):
{json.dumps(paper_list, indent=2)}

Context from papers:
{json.dumps(context[:5], indent=2)}

Please answer the question. Use tools if needed to provide comprehensive analysis."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]

        # Stream with tool support
        assistant_response_chunks = []

        if enable_tools:
            from app.llm.tools import tool_registry

            tools = tool_registry.get_tools_for_openai()

            # Use tool-aware streaming
            async for evt in self.stream_with_tools(
                messages=messages, db_session=db_session, user_id=user_id
            ):
                yield evt
                # Collect chunks for saving
                if evt.startswith("event: chunk"):
                    try:
                        data_line = evt.split("\n")[1]
                        if data_line.startswith("data: "):
                            chunk_data = json.loads(data_line[6:])
                            if isinstance(chunk_data, dict) and "text" in chunk_data:
                                assistant_response_chunks.append(chunk_data["text"])
                    except:
                        pass
        else:
            # Regular streaming without tools
            async for chunk_text in self.llm_service.stream_citation_based_response(
                query=request.query, context=context
            ):

                text = get_stream_response_content(chunk_text)
                if text:
                    assistant_response_chunks.append(text)
                    async for evt in stream_event(name="chunk", data={"text": text}):
                        yield evt

        # Save conversation
        full_response = "".join(assistant_response_chunks)
        paper_ids = [str(p.paper_id) for p in results.papers]
        await self._save_conversation_messages(
            conversation_service,
            conversation.id,
            user_id,
            request.query,
            full_response,
            paper_ids,
        )

        async for evt in stream_event(name="done", data=None):
            yield evt

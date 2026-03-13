"""
Chat service that orchestrates retrieval and LLM services for chatbot functionality
"""

import asyncio
import json
import time
from typing import AsyncGenerator, Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from app.llm import get_llm_service
from app.domain.chat.schemas import ChatMessageRequest
from app.extensions.logger import create_logger
from app.extensions import (
    stream_event,
    get_stream_response_content,
    get_stream_response_reasoning,
)
from app.extensions.prompt_filter import is_gibberish
from app.rag_pipeline.pipeline import Pipeline as RAGPipeline
from app.rag_pipeline.hybrid_pipeline import HybridPipeline
from app.rag_pipeline.database_pipeline import DatabasePipeline
from app.rag_pipeline.schemas import RAGResult
from app.domain.conversations.service import ConversationService
from app.domain.messages.service import MessageService

from app.extensions.citation_extractor import CitationExtractor
from app.validation.service import validate_answer
from app.validation.schemas import ValidationRequest
from app.domain.conversations.context_manager import ConversationContextManager
from app.domain.conversations.summarization_service import ConversationSummarizationService
from .event_emitter import EventEmitter
from .response_builder import ChatResponseBuilder
from .background_tasks import ChatBackgroundTaskService
from .error_handlers import ChatErrorHandler

logger = create_logger(__name__)


class ChatService():
    """Service class for handling chat interactions with tool support"""

    def __init__(
        self,
        db_session: AsyncSession,
        rag_pipeline: RAGPipeline,
        hybrid_pipeline: HybridPipeline,
        database_pipeline: DatabasePipeline,
        message_service: "MessageService",
        context_manager: "ConversationContextManager",
        summarization_service: "ConversationSummarizationService",
        background_tasks: "ChatBackgroundTaskService",
        llm_service=None,
    ):
        """Initialize chat service with dependency injection.
        
        Args:
            db_session: Database session
            rag_pipeline: Standard RAG pipeline
            hybrid_pipeline: Hybrid RAG pipeline
            database_pipeline: Database-only pipeline
            message_service: Message service for database operations
            context_manager: Conversation context manager
            summarization_service: Conversation summarization service
            response_builder: Chat response builder
            background_tasks: Background task service
            error_handler: Error handler
            llm_service: LLM service (optional, singleton if not provided)
        """
        self.db_session = db_session
        self.llm_service = llm_service or get_llm_service()
        self.rag_pipeline = rag_pipeline
        self.hybrid_pipeline = hybrid_pipeline
        self.database_pipeline = database_pipeline
        self.message_service = message_service
        self.context_manager = context_manager
        self.summarization_service = summarization_service
        self.response_builder = ChatResponseBuilder()
        self.background_tasks = background_tasks
        self.error_handler = ChatErrorHandler()

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

        pipeline_start_time = time.time()
        # Determine pipeline type (backward compatibility with use_hybrid_pipeline)
        if request.use_hybrid_pipeline:
            pipeline_type = "hybrid"
        else:
            pipeline_type = request.pipeline
        
        logger.info(
            f"Pipeline selection: type={pipeline_type}",
            extra={"pipeline_type": pipeline_type}
        )

        conversation_service = ConversationService(db_session)
        
        # Truncate query to fit conversation title max length (200 chars)
        title = request.query[:197] + "..." if len(request.query) > 200 else request.query
        
        conversation = await conversation_service.get_or_create_conversation(
            user_id=user_id,
            conversation_id=request.conversation_id,
            title=title,
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
            async for evt in self.error_handler.handle_gibberish_input(
                conversation_service,
                conversation.conversation_id,
                user_id,
                request.query,
            ):
                yield evt
            return

        # Reset event collection for this request
        EventEmitter.reset_collection()
        logger.debug(f"Request: {request}")
        results = None
        try:
            async for result_or_evt, event_data in self._execute_rag_pipeline(
                request.query,
                filters=request.filters,
                pipeline_type=pipeline_type,
                conversation_id=conversation.conversation_id,
            ):
                if isinstance(result_or_evt, str):
                    yield result_or_evt
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
            async for evt in self.error_handler.handle_no_results(
                conversation.conversation_id, user_id, conversation_service
            ):
                yield evt
            return

        # Get conversation history for context
        conversation_history, history_tokens = (
            await self.context_manager.get_conversation_context(
                conversation_id=conversation.conversation_id,
                db_session=db_session,
                include_current_query=False,  # Exclude current query (it's in request.query)
            )
        )
        logger.info(
            f"Retrieved conversation history: {len(conversation_history)} messages, {history_tokens} tokens"
        )

        context, chunk_papers = self.response_builder.build_context_from_results(results)
        retrieved_paper_ids = self.response_builder.get_retrieved_paper_ids(results)
        paper_snapshots = self.response_builder.extract_metadata_from_results(results)
        from app.domain.papers.schemas import PaperMetadata
        papers_metadata = [
            PaperMetadata.from_ranked_paper(p) for p in results.papers
        ]
        async for evt in EventEmitter.emit_paper_metadata_events(papers_metadata):
            yield evt

        assistant_response_chunks = []
        reasoning_chunks = []
        enhanced_query = self.response_builder.build_enhanced_query(
            request.query, conversation_history
        )

        async for chunk_text in self.llm_service.stream_citation_based_response(
            query=enhanced_query, context=context
        ):
            text = get_stream_response_content(chunk_text)
            reasoning_chunk = get_stream_response_reasoning(chunk_text)
            if reasoning_chunk and reasoning_chunk not in reasoning_chunks:
                async for evt in EventEmitter.emit_reasoning_event(reasoning_chunk):
                    yield evt
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

        # Calculate pipeline completion time
        pipeline_completion_time = int(
            (time.time() - pipeline_start_time) * 1000
        )  # Convert to milliseconds

        # Build citation_details for database (for analytics)
        citation_details = {}
        for paper_id, details in final_citations.items():
            citation_details[paper_id] = {
                "claim": details["claim"],
                "confidence": details["confidence"],
                "chunk_ids": [],
                "context": f"Cited {len(details['claims'])} times",
            }


        # Get all collected progress events from EventEmitter
        # Finalize reasoning buffer before collecting events
        EventEmitter._finalize_reasoning()
        progress_events = EventEmitter.get_collected_events()
        
        message_id = None
        try:
            message_id = await conversation_service.add_message_to_conversation(
                conversation_id=conversation.conversation_id,
                user_id=user_id,
                message_text=full_response,
                role="assistant",
                pipeline_type=pipeline_type,
                completion_time_ms=pipeline_completion_time,
                paper_ids=retrieved_paper_ids,
                paper_snapshots=paper_snapshots,
                progress_events=progress_events,
            )
            logger.info(
                f"Saved assistant message with {len(progress_events)} progress events"
            )
        except Exception as e:
            logger.error(f"Failed to save messages: {e}")

        if message_id:
            try:
                validation_request = ValidationRequest(
                    query=request.query,
                    context=context,
                    generated_answer=full_response,
                    message_id=message_id,
                )
                asyncio.create_task(
                    self.background_tasks.run_validation_with_new_session(validation_request)
                )
                logger.info(f"Validation task created for message {message_id}")
            except Exception as e:
                logger.error(f"Failed to create validation task: {e}")

        try:
            asyncio.create_task(
                self.background_tasks.run_summarization_with_new_session(
                    conversation.conversation_id
                )
            )
        except Exception as e:
            logger.error(f"Failed to create summarization task: {e}")

        async for evt in EventEmitter.emit_done_event():
            yield evt



    async def _execute_rag_pipeline(
        self,
        query: str,
        max_subtopics: int = 3,
        per_subtopic_limit: int = 30,
        filters: Optional[Dict[str, Any]] = None,
        pipeline_type: str = "database",
        conversation_id: Optional[str] = None,
    ) -> AsyncGenerator[tuple[str | RAGResult | None, Dict[str, Any]], None]:
        """Execute RAG pipeline and yield events.

        Args:
            query: User's search query
            max_subtopics: Maximum number of subtopics to generate (standard pipeline only)
            per_subtopic_limit: Limit per subtopic (standard pipeline only)
            filters: Optional filters for retrieval
            pipeline_type: Pipeline to use: 'database' (default, fast), 'hybrid' (S2/OA), or 'standard' (legacy)
            conversation_id: Optional conversation ID for context
        """
        results: Optional[RAGResult] = None
        subtopics_found: List[str] = []

        logger.info(f"Executing {pipeline_type} RAG pipeline for query: {query}")

        if pipeline_type == "database":
            async for event in self.database_pipeline.run_database_search_workflow(  # type: ignore
                query=query,
                top_papers=50,
                top_chunks=40,
                filters=filters,
                conversation_id=conversation_id,
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
                            "total_papers": event.data.get("total_papers", 0),  # type: ignore
                            "total_chunks": event.data.get("total_chunks", 0),  # type: ignore
                        }
                elif event.type == "result":
                    results = event.data  # type: ignore
                    yield None, {"type": "result", "data": results}

        elif pipeline_type == "hybrid":
            async for event in self.hybrid_pipeline.run_hybrid_rag_workflow(  # type: ignore
                query=query,
                max_subtopics=max_subtopics,
                per_subtopic_limit=per_subtopic_limit,
                filters=filters,
                conversation_id=conversation_id,
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
                            "total_papers": event.data.get("total_papers", 0),  # type: ignore
                            "total_chunks": event.data.get("total_chunks", 0),  # type: ignore
                        }
                elif event.type == "result":
                    results = event.data  # type: ignore
        else:
            # Use standard pipeline's run method
            async for event in self.rag_pipeline.run(
                query,
                max_subtopics=max_subtopics,
                per_subtopic_limit=per_subtopic_limit,
                filters=filters,
                conversation_id=conversation_id,
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
                            "total_papers": event.data.get("total_papers", 0),  # type: ignore
                            "total_chunks": event.data.get("total_chunks", 0),  # type: ignore
                        }
                elif event.type == "result":
                    results = event.data  # type: ignore

        yield results, {}


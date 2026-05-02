"""
Chat service that orchestrates retrieval and LLM services for chatbot functionality
"""

import asyncio
import datetime
import time
from typing import AsyncGenerator, Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import db_session_context
from app.core.exceptions import BadRequestException
from app.domain.authors.schemas import AuthorMetadata
from app.domain.conversations.schemas import (
    ConversationUpdate,
    ConversationUpdateInternal,
)
from app.domain.papers.schemas import PaperMetadata
from app.llm import get_llm_service
from app.domain.chat.schemas import ChatMessageRequest
from app.extensions.logger import create_logger
from app.extensions import (
    stream_event,
    get_stream_response_content,
    get_stream_response_reasoning,
)
from app.extensions.prompt_filter import is_gibberish
from app.domain.chat.query_router import route_query
from app.llm.schemas.chat import GeneratedQueryPlanResponse, QuestionBreakdownResponse
from app.rag_pipeline.pipeline import Pipeline as RAGPipeline
from app.rag_pipeline.hybrid_pipeline import HybridPipeline
from app.rag_pipeline.database_pipeline import DatabasePipeline
from app.rag_pipeline.scoped_pipeline import ScopedPipeline
from app.rag_pipeline.schemas import RAGResult, SearchWorkflowConfig
from app.domain.conversations.service import ConversationService
from app.domain.messages.service import MessageService

from app.extensions.citation_extractor import CitationExtractor
from app.validation.schemas import ValidationRequest
from app.domain.conversations.context_manager import ConversationContextManager
from app.domain.conversations.summarization_service import (
    ConversationSummarizationService,
)
from .event_emitter import ChatEventEmitter
from .response_builder import ChatResponseBuilder
from .background_tasks import ChatBackgroundTaskService
from .error_handlers import ChatErrorHandler

logger = create_logger(__name__)


class ChatService:
    """Service class for handling chat interactions with tool support"""

    def __init__(
        self,
        db_session: AsyncSession,
        rag_pipeline: RAGPipeline,
        hybrid_pipeline: HybridPipeline,
        database_pipeline: DatabasePipeline,
        scoped_pipeline: ScopedPipeline,
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
        self.scoped_pipeline = scoped_pipeline
        self.message_service = message_service
        self.context_manager = context_manager
        self.summarization_service = summarization_service
        self.response_builder = ChatResponseBuilder()
        self.background_tasks = background_tasks
        self.error_handler = ChatErrorHandler()

    async def stream_research_pipeline(
        self,
        request: ChatMessageRequest,
        user_id: int,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat message with citation extraction.

        Args:
            request: Chat message request
            user_id: User ID
        Yields:
            SSE events:
            - conversation: Conversation ID
            - retrieved: All retrieved paper IDs
            - metadata: Paper metadata for client caching
            - token: Each token as generated
            - citation: When a citation is detected (with claim, confidence)
            - done: Completion with cited vs retrieved paper lists
        """
        pipeline_start_time = time.time()
        event_emitter = ChatEventEmitter()

        async with db_session_context() as setup_session:
            conversation_service = ConversationService(setup_session)

            conversation, is_new_conversation = await self._setup_conversation(
                conversation_service, request.conversation_id, user_id
            )
            conversation_id = conversation.conversation_id

            # Emit conversation ID immediately to frontend
            async for evt in event_emitter.emit_conversation_event(
                conversation_id=conversation_id
            ):
                yield evt

            pipeline_type, route_decision = self._determine_pipeline(request)
            event_emitter.set_pipeline_type(pipeline_type)

            logger.info(
                f"Pipeline selection: request={request.pipeline}, resolved={pipeline_type}, route_type={route_decision.route_type}",
                extra={"pipeline_type": pipeline_type, "route": route_decision.route_type},
            )

            await self._handle_user_message(
                conversation_service,
                request,
                conversation_id,
                user_id,
                is_new_conversation,
            )

            if is_gibberish(request.query):
                async for evt in self.error_handler.handle_gibberish_input(
                    conversation_service,
                    conversation_id,
                    user_id,
                    request.query,
                    event_emitter,
                ):
                    yield evt
                return

            config, conversation_title = await self._prepare_search_config(
                request,
                conversation_id,
                conversation_service,
                is_new_conversation,
                setup_session,
            )

        if conversation_title:
            async for evt in event_emitter.emit_conversation_event(
                conversation_id=conversation_id, title=conversation_title
            ):
                yield evt

        results = None
        try:
            async for result_or_evt, _ in self._execute_rag_pipeline(
                config=config,
                event_emitter=event_emitter,
                pipeline_type=pipeline_type,
                conv_id=conversation.conversation_id,
                scoped_ids=route_decision.scoped_paper_ids,
            ):
                if isinstance(result_or_evt, str):
                    yield result_or_evt
                elif isinstance(result_or_evt, RAGResult):
                    results = result_or_evt
        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            async for evt in event_emitter.emit_error_event(
                "An error occurred during retrieval.", "rag_error"
            ):
                yield evt
            return

        if not results or not results.papers:
            async for evt in self.error_handler.handle_no_results(
                conversation.conversation_id,
                user_id,
                conversation_service,
                event_emitter,
            ):
                yield evt
            return

        async for evt in self._synthesize_and_save_response(
            request,
            conversation_id,
            user_id,
            event_emitter,
            results,
            pipeline_type,
            pipeline_start_time,
        ):
            yield evt

    async def _execute_rag_pipeline(
        self,
        config: SearchWorkflowConfig,
        event_emitter: ChatEventEmitter,
        pipeline_type: str,
        conv_id: str,
        scoped_ids: List[str],
    ) -> AsyncGenerator[tuple[Any, Dict], None]:
        """Dynamically select and execute the correct pipeline, mapping events cleanly."""
        # 1. Select the correct pipeline iterator
        if pipeline_type == "scoped" or (scoped_ids and len(scoped_ids) > 0):
            iterator = self.scoped_pipeline.run_scoped_search_workflow(
                query=config.query,
                paper_ids=scoped_ids,
                enable_reranking=True,
            )
        elif pipeline_type in ["database", "research"]:
            iterator = self.database_pipeline.run_database_search_workflow(
                config=config
            )
        elif pipeline_type == "hybrid":
            iterator = self.hybrid_pipeline.run_hybrid_rag_workflow(
                query=config.query,
                max_subtopics=2,
                per_subtopic_limit=20,
                filters=config.filters,
                conversation_id=conv_id,
            )
        else:
            logger.warning(
                f"NO VALID PIPELINE SELECTED FOR TYPE: {pipeline_type}, defaulting to database pipeline"
            )
            iterator = self.database_pipeline.run_database_search_workflow(
                config=config
            )

        async for event in iterator:
            if event.type == "step":
                async for evt in stream_event(name="step", data=event.data):
                    yield evt, {"type": "step"}
            elif event.type == "search_queries":
                queries = (
                    event.data.get("queries", [])
                    if isinstance(event.data, dict)
                    else []
                )
                async for evt in event_emitter.emit_searching_event(queries):
                    yield evt, {"type": "search_queries"}
            elif event.type == "ranking":
                async for evt in event_emitter.emit_ranking_event(
                    total_papers=(
                        event.data.get("total_papers", 0)
                        if isinstance(event.data, dict)
                        else 0
                    ),
                    chunks=(
                        event.data.get("total_chunks", 0)
                        if isinstance(event.data, dict)
                        else 0
                    ),
                ):
                    yield evt, {"type": "ranking"}
            elif event.type == "result":
                yield event.data, {"type": "result"}

    async def _prepare_search_config(
        self,
        request: ChatMessageRequest,
        conv_id: str,
        service: ConversationService,
        is_new: bool,
        db_session: AsyncSession,
    ) -> tuple[SearchWorkflowConfig, str | None]:
        """Decompose the query and generate the SearchWorkflowConfig."""
        history = []
        if not is_new:
            history = await self.context_manager.get_conversation_top_history(
                conv_id, db_session, top_n=5
            )

        breakdown = await self._decompose_query(request.query, history)

        conversation = None
        if breakdown and breakdown.clarified_question:
            conversation = await service.update_conversation(
                conv_id, ConversationUpdateInternal(title=breakdown.clarified_question)
            )

        return SearchWorkflowConfig(
            query=request.query,
            search_queries=(
                breakdown.hybrid_queries if breakdown and breakdown.hybrid_queries else [request.query]
            ),
            filters=(
                request.filters.model_dump(exclude_none=True)
                if request.filters
                else None
            ),
            intent=breakdown.intent if breakdown else None,
        ), (conversation.title if conversation else None)

    async def _setup_conversation(
        self, service: ConversationService, conv_id: Optional[str], user_id: int
    ) -> tuple[Any, bool]:
        """Fetch existing or create new conversation."""
        if conv_id:
            conversation, was_cloned = await service.get_or_clone_conversation_for_user(
                conversation_id=conv_id,
                user_id=user_id,
            )
            if not conversation:
                logger.warning(f"Invalid conversation_id: {conv_id}")
                raise BadRequestException("Conversation does not exist.")
            if was_cloned:
                logger.info(
                    "Conversation was shared from another user; deep-cloned for isolated follow-up.",
                    extra={"source_conversation_id": conv_id, "cloned_conversation_id": conversation.conversation_id},
                )
            return conversation, False

        conversation = await service.create_conversation(user_id=user_id)
        return conversation, True

    def _determine_pipeline(self, request: ChatMessageRequest) -> tuple[str, Any]:
        """Determine which RAG pipeline to use based on request."""
        pipeline_selection = (
            "hybrid"
            if request.use_hybrid_pipeline
            else (request.pipeline or "database").strip().lower()
        )
        request_filters = (
            request.filters.model_dump(exclude_none=True) if request.filters else {}
        )
        if getattr(request, "paper_ids", None):
            request_filters = dict(request_filters)
            request_filters["paper_ids"] = list(getattr(request, "paper_ids") or [])
        route_decision = route_query(pipeline_selection, request_filters)

        logger.info(f"Resolved pipeline: {route_decision.route_type}")
        return route_decision.route_type, route_decision

    async def _handle_user_message(
        self,
        service: ConversationService,
        request: ChatMessageRequest,
        conv_id: str,
        user_id: int,
        is_new: bool,
    ):
        """Check for retries and save the user message to DB."""
        if request.is_retry and request.client_message_id and not is_new:
            existing = await self.message_service.check_existing_message(
                conversation_id=conv_id, client_message_id=request.client_message_id
            )
            if existing:
                logger.info("Retry detected - skipping duplicate user message.")
                return

        await service.add_message_to_conversation(
            conversation_id=conv_id,
            user_id=user_id,
            message_text=request.query,
            role="user",
            client_message_id=request.client_message_id,
        )

    async def _synthesize_and_save_response(
        self,
        request: ChatMessageRequest,
        conv_id: str,
        user_id: int,
        event_emitter: ChatEventEmitter,
        results: RAGResult,
        pipeline_type: str,
        start_time: float,
    ):
        """Format chunks, stream final LLM answer, save to DB, and trigger background tasks."""
        from app.domain.papers.schemas import PaperMetadata

        async with db_session_context() as context_session:
            conv_context, _ = await self.context_manager.get_conversation_context(
                conv_id, context_session, include_current_query=True
            )
        context_string, _ = self.response_builder.build_context_from_results(results)
        enhanced_query = self.response_builder.build_enhanced_query(
            request.query,
            conversation_history=conv_context,
            context_string=context_string,
        )

        async for evt in event_emitter.emit_paper_metadata_events(
            [PaperMetadata.from_ranked_paper(p) for p in results.papers]
        ):
            yield evt

        response_chunks = []
        reasoning_chunks = []
        async for chunk_text in self.llm_service.stream_citation_based_response(
            context=enhanced_query
        ):
            text = get_stream_response_content(chunk_text)
            reasoning = get_stream_response_reasoning(chunk_text)

            if reasoning and reasoning not in reasoning_chunks:
                async for evt in event_emitter.emit_reasoning_event(reasoning):
                    yield evt
                reasoning_chunks.append(reasoning)

            if text:
                async for evt in event_emitter.emit_chunk_event(text):
                    yield evt
                response_chunks.append(text)

        event_emitter._finalize_reasoning()
        full_response = "".join(response_chunks)

        # 3. Save Assistant Message
        try:
            async with db_session_context() as save_session:
                conv_service = ConversationService(save_session)
                msg_id = await conv_service.add_message_to_conversation(
                    conversation_id=conv_id,
                    user_id=user_id,
                    message_text=full_response,
                    role="assistant",
                    pipeline_type=pipeline_type,
                    completion_time_ms=int((time.time() - start_time) * 1000),
                    paper_ids=self.response_builder.get_retrieved_paper_ids(results),
                    paper_snapshots=self.response_builder.extract_metadata_from_results(
                        results
                    ),
                    progress_events=event_emitter.get_collected_events(),
                )
        except Exception as e:
            logger.error(f"Failed to save assistant message: {e}")
            msg_id = None

        # 4. Background Tasks
        if msg_id:
            val_req = ValidationRequest(
                query=request.query,
                context=context_string,
                enhanced_query=enhanced_query,
                context_chunks=self.response_builder.extract_context_chunks_from_results(
                    results
                ),
                generated_answer=full_response,
                model_name=self.llm_service.llm_provider.get_model(),
                message_id=msg_id,
            )
            asyncio.create_task(
                self.background_tasks.run_validation_with_new_session(val_req)
            )
            asyncio.create_task(
                self.background_tasks.run_summarization_with_new_session(conv_id)
            )

        async for evt in event_emitter.emit_done_event():
            yield evt

    async def _get_conversation_history(
        self, conversation_id: str, db_session: AsyncSession
    ):
        """Get conversation history for a given conversation ID."""
        conversation_history = None
        if conversation_id:
            try:
                from app.domain.conversations.context_manager import (
                    ConversationContextManager,
                )

                context_mgr = ConversationContextManager()
                conversation_history, _ = await context_mgr.get_conversation_context(
                    conversation_id=conversation_id,
                    db_session=self.db_session,
                    include_current_query=False,
                )
                logger.info(f"Loaded {len(conversation_history)} messages for context")
            except Exception as e:
                logger.warning(f"Failed to load conversation history: {e}")
        return conversation_history

    async def _decompose_query(
        self, query: str, conversation_histories: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[GeneratedQueryPlanResponse]:
        """Decompose query into subqueries using LLM."""
        decomposition_start_time = time.time()
        try:
            breakdown_response = await self.llm_service.decompose_user_query_v3(
                query, conversation_history=conversation_histories
            )
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            breakdown_response = None

        decomposition_time_ms = int((time.time() - decomposition_start_time) * 1000)
        logger.info(f"Query decomposition took {decomposition_time_ms} ms")

        return breakdown_response

    async def test_streaming_response(self) -> AsyncGenerator[str, None]:
        """Test method to simulate streaming response with events."""
        event_emitter = ChatEventEmitter()

        async for evt in event_emitter.emit_conversation_event(
            conversation_id="test-12345"
        ):
            yield evt

        await asyncio.sleep(1)

        async for evt in event_emitter.emit_conversation_event(
            conversation_id="test-12345", title="Test Conversation"
        ):
            yield evt

        async for evt in event_emitter.emit_searching_event(
            query=["test query 1", "test query 2"]
        ):
            yield evt
        await asyncio.sleep(1)

        async for evt in event_emitter.emit_ranking_event(total_papers=2, chunks=10):
            yield evt

        await asyncio.sleep(1)

        from app.domain.papers.schemas import PaperMetadata

        async for evt in event_emitter.emit_paper_metadata_events(
            [
                PaperMetadata(
                    paper_id="paper1",
                    title="Test Paper 1",
                    authors=[
                        AuthorMetadata(name="Author A", display_name="Author A", orcid="0000-0001-2345-6789", h_index=10, citation_count=100, paper_count=20, author_id="author1"),
                        AuthorMetadata(name="Author B", author_id="1", display_name="Author B", orcid="0000-0002-1825-0097"),
                    ],
                    abstract="This is a test abstract for paper 1.",
                    year=2023,
                    publication_date=datetime.datetime.fromtimestamp(time.time()),
                    venue="Test Venue",
                    url="http://example.com/paper1",
                    pdf_url="http://example.com/paper1.pdf",
                    citation_count=5,
                    influential_citation_count=2,
                    reference_count=10,
                    author_trust_score=0.8,
                    institutional_trust_score=0.7,
                    fields_of_study=["Computer Science", "Artificial Intelligence"],
                    fwci=1.2,
                    is_open_access=True,
                    is_retracted=False,
                    topics=[{}],
                    keywords=[{}],
                ),
                PaperMetadata(
                    paper_id="paper2",
                    title="Test Paper 2",
                    authors=[AuthorMetadata(author_id="author3", name="Author C", display_name="Author C")],
                    abstract="This is a test abstract for paper 2.",
                    year=2022,
                    publication_date=datetime.datetime.fromtimestamp(time.time()),
                    venue="Test Venue",
                    url="http://example.com/paper2",
                    pdf_url="http://example.com/paper2.pdf",
                    citation_count=3,
                    influential_citation_count=1,
                    reference_count=5,
                    author_trust_score=0.6,
                    institutional_trust_score=0.5,
                    fields_of_study=["Computer Science"],
                    fwci=0.9,
                    is_open_access=False,
                    is_retracted=False,
                    topics=[{}],
                    keywords=[{}],
                ),
            ]
        ):
            yield evt

        await asyncio.sleep(1)

        async for evt in event_emitter.emit_reasoning_event(
            "This is a test reasoning chunk."
        ):
            yield evt

        await asyncio.sleep(1)

        async for evt in event_emitter.emit_chunk_event(
            "This is a test response chunk."
        ):
            yield evt
            
        async for evt in event_emitter.emit_chunk_event(
            "This is another test response chunk."
        ):
            yield evt
        async for evt in event_emitter.emit_chunk_event(
            "This is the final test response chunk."
        ):
            yield evt

        async for evt in event_emitter.emit_done_event():
            yield evt

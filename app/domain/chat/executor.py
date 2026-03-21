"""
Chat pipeline executor for background task processing.
Runs RAG pipelines asynchronously and saves events to database.
"""
import asyncio
import time
from typing import Dict, Any, Optional, List
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from fastapi.encoders import jsonable_encoder
from app.core.config import settings
from app.core.container import ServiceContainer
from app.models.pipeline_tasks import PipelineTaskStatus, PipelinePhase, PipelineEventType
from app.extensions.logger import create_logger
from app.domain.chat.query_router import route_query
from app.domain.chat.pre_response import PreResponsePresets
from app.domain.chat.redis_streams import publish_task_event

logger = create_logger(__name__)


def _get_no_results_message() -> str:
    """Return user-facing fallback message when retrieval yields no papers."""
    return PreResponsePresets.NO_RELEVANCE_PAPER.message


def _serialize_ranked_papers_for_cache(rag_result) -> list[Dict[str, Any]]:
    """Convert ranked papers into JSON-safe lightweight dictionaries for task cache."""
    serialized_papers: list[Dict[str, Any]] = []

    for ranked in (rag_result.papers or []):
        paper_obj = getattr(ranked, "paper", None)
        serialized_papers.append(
            {
                "id": getattr(ranked, "id", None),
                "paper_id": getattr(ranked, "paper_id", None),
                "title": getattr(paper_obj, "title", None) if paper_obj else None,
                "year": getattr(paper_obj, "year", None) if paper_obj else None,
                "relevance_score": getattr(ranked, "relevance_score", None),
                "ranking_scores": getattr(ranked, "ranking_scores", None),
            }
        )

    return jsonable_encoder(serialized_papers)


def _serialize_chunks_for_cache(rag_result) -> list[Dict[str, Any]]:
    """Convert chunks into JSON-safe dictionaries for task cache."""
    serialized_chunks = []

    for chunk in (rag_result.chunks or []):
        if hasattr(chunk, "model_dump"):
            serialized_chunks.append(chunk.model_dump(mode="json"))
        else:
            serialized_chunks.append(jsonable_encoder(chunk))

    return jsonable_encoder(serialized_chunks)


def _build_step_event_from_rag_event(event_type: str, event_data: Any) -> Optional[Dict[str, Any]]:
    """Map RAG pipeline events to legacy-style progress payloads."""
    payload = event_data if isinstance(event_data, dict) else {}

    if event_type in ("search_queries", "searching"):
        queries = payload.get("queries", [])
        return {
            "type": "searching",
            "content": "Searching academic databases...",
            "metadata": {"queries": queries},
            "phase": PipelinePhase.SEARCH,
            "progress_percent": 20,
        }

    if event_type == "ranking":
        total_papers = payload.get("total_papers", 0)
        total_chunks = payload.get("total_chunks", 0)
        return {
            "type": "ranking",
            "content": f"Filtering {total_papers} retrieved papers by content relevance, quality, authors,...",
            "metadata": {"total_papers": total_papers, "chunks": total_chunks},
            "phase": PipelinePhase.RANKING,
            "progress_percent": 50,
        }

    return None


def _build_scoped_quote_index(rag_result) -> Dict[str, Dict[str, Any]]:
    """Index scoped chunk evidence for quote metadata emission."""
    quote_index: Dict[str, Dict[str, Any]] = {}

    for chunk in (getattr(rag_result, "chunks", None) or []):
        paper_id = str(getattr(chunk, "paper_id", "") or "")
        chunk_id = str(getattr(chunk, "chunk_id", "") or "")
        if not paper_id or not chunk_id:
            continue

        quote_index[f"{paper_id}|{chunk_id}"] = {
            "paper_id": paper_id,
            "chunk_id": chunk_id,
            "section": getattr(chunk, "section_title", None),
            "quote": getattr(chunk, "text", None),
        }

    return quote_index


async def execute_chat_pipeline(
    task_id: str,
    user_id: int,
    conversation_id: str,
    query: str,
    pipeline_type: str,
    filters: Dict[str, Any]
) -> None:
    """
    Execute chat pipeline in background and save events to database.
    
    This function:
    1. Creates its own database session (worker thread)
    2. Updates task status to RUNNING
    3. Executes the RAG pipeline
    4. Saves events to pipeline_events table
    5. Creates assistant message on completion
    6. Updates task status to COMPLETED/FAILED
    
    Args:
        task_id: Pipeline task identifier
        user_id: User who initiated the chat
        conversation_id: Conversation this belongs to
        query: User's question
        pipeline_type: Pipeline to use (database, hybrid, standard)
        filters: Optional search filters
    """
    pipeline_start_time = time.time()

    # Create worker-specific database session
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    
    container = None  # Define outside try block
    
    async with async_session() as db:
        try:
            container = ServiceContainer(db)

            task_record = await container.pipeline_task_service.get_task(task_id)
            client_message_id = task_record.client_message_id if task_record else None
            progress_events: List[Dict[str, Any]] = []
            route_decision = route_query(pipeline_type, filters or {})
            emitted_step_types: set[str] = set()
            current_step = 0

            async def emit_event(event_type: str, event_data: Dict[str, Any]) -> None:
                """Persist event to DB and publish to Redis Stream for live delivery."""
                db_event = await container.pipeline_event_store.save_event(
                    task_id=task_id,
                    event_type=event_type,
                    event_data=event_data,
                )

                try:
                    await publish_task_event(
                        task_id=task_id,
                        event_type=event_type,
                        sequence=db_event.sequence_number,
                        event_data=event_data,
                    )
                except Exception as redis_error:
                    logger.warning(
                        f"Task {task_id}: Redis stream publish failed for {event_type}: {redis_error}"
                    )

            async def emit_step_event(mapped_step_event: Dict[str, Any]) -> None:
                nonlocal current_step
                step_type = str(mapped_step_event.get("type", "")).strip()
                if not step_type or step_type in emitted_step_types:
                    return

                emitted_step_types.add(step_type)
                current_step += 1

                payload = {
                    **mapped_step_event,
                    "current_step": current_step,
                    "total_steps": route_decision.total_steps,
                }

                await emit_event(
                    event_type=PipelineEventType.STEP,
                    event_data=payload,
                )

                progress_events.append(
                    {
                        "type": payload.get("type"),
                        "timestamp": int(time.time() * 1000),
                        "content": payload.get("content"),
                        "metadata": payload.get("metadata"),
                    }
                )

                await container.pipeline_task_service.update_progress(
                    task_id=task_id,
                    phase=payload.get("phase", PipelinePhase.INIT),
                    progress_percent=payload.get("progress_percent", 10),
                )
            
            # Update task to RUNNING
            await container.pipeline_task_service.update_status(
                task_id=task_id,
                status=PipelineTaskStatus.RUNNING
            )
            
            # Send step count first (skip init step)
            await emit_event(
                event_type=PipelineEventType.STEP,
                event_data={
                    "type": "step_count",
                    "total_steps": route_decision.total_steps,
                },
            )
            
            # Create user message first
            user_message = await container.message_service.create_message(
                user_id=user_id,
                conversation_id=conversation_id,
                role="user",
                content=query
            )
            
            # Execute RAG pipeline based on type
            logger.info(f"Task {task_id}: Executing {pipeline_type} pipeline")
            
            # Execute pipeline and save events
            rag_result = None
            
            if route_decision.route_type == "scoped":
                logger.info(f"Task {task_id}: Using scoped pipeline with {len(route_decision.scoped_paper_ids)} paper IDs")
                async for event in container.scoped_pipeline.run_scoped_search_workflow(
                    query=query,
                    paper_ids=route_decision.scoped_paper_ids,
                    top_chunks=40,
                    top_papers=20,
                    enable_reranking=True,
                ):
                    event_type_str = event.type
                    event_data = event.data

                    mapped_step_event = _build_step_event_from_rag_event(event_type_str, event_data)
                    if mapped_step_event:
                        await emit_step_event(mapped_step_event)

                    if event_type_str == "result":
                        from app.rag_pipeline.schemas import RAGResult
                        rag_result = event_data if isinstance(event_data, RAGResult) else None

            elif pipeline_type == "database":
                # Use database pipeline (fast, DB-only)
                async for event in container.database_pipeline.run_database_search_workflow(
                    query=query,
                    top_papers=50,
                    top_chunks=40,
                    filters=filters or {},
                ):
                    event_type_str = event.type
                    event_data = event.data
                    
                    mapped_step_event = _build_step_event_from_rag_event(event_type_str, event_data)
                    if mapped_step_event:
                        await emit_step_event(mapped_step_event)

                    if event_type_str == "result":
                        # Store the RAG result
                        from app.rag_pipeline.schemas import RAGResult
                        rag_result = event_data if isinstance(event_data, RAGResult) else None
                        
            elif pipeline_type == "hybrid":
                # Use hybrid pipeline (BM25 + Semantic with S2/OA)
                async for event in container.hybrid_pipeline.run_hybrid_rag_workflow(
                    query=query,
                    max_subtopics=3,
                    per_subtopic_limit=50,
                    top_chunks=40,
                    filters=filters or {},
                ):
                    event_type_str = event.type
                    event_data = event.data
                    
                    mapped_step_event = _build_step_event_from_rag_event(event_type_str, event_data)
                    if mapped_step_event:
                        await emit_step_event(mapped_step_event)

                    if event_type_str == "result":
                        from app.rag_pipeline.schemas import RAGResult
                        rag_result = event_data if isinstance(event_data, RAGResult) else None
            else:
                # Use standard pipeline (legacy)
                async for event in container.pipeline.run_paper_rag_workflow(
                    query=query,
                    max_subtopics=3,
                    per_subtopic_limit=30,
                    filters=filters or {},
                ):
                    event_type_str = event.type
                    event_data = event.data
                    
                    mapped_step_event = _build_step_event_from_rag_event(event_type_str, event_data)
                    if mapped_step_event:
                        await emit_step_event(mapped_step_event)

                    if event_type_str == "result":
                        from app.rag_pipeline.schemas import RAGResult
                        rag_result = event_data if isinstance(event_data, RAGResult) else None
            
            # Graceful no-results path: return assistant explanation instead of failing task
            if not rag_result or not rag_result.papers:
                llm_no_results_chunks: List[str] = []
                no_results_message = _get_no_results_message()

                try:
                    from app.extensions import get_stream_response_content

                    async for chunk_text in container.llm_service.stream_citation_based_response(
                        query=query,
                        context=(
                            "No relevant papers were retrieved from current search indexes. "
                            "Provide guidance and possible intended query reformulations."
                        ),
                        prompt_name="generate_no_results_guidance",
                    ):
                        text = get_stream_response_content(chunk_text)
                        if not text:
                            continue

                        llm_no_results_chunks.append(text)

                        await emit_event(
                            event_type=PipelineEventType.CHUNK,
                            event_data={
                                "type": "chunk",
                                "content": text,
                            },
                        )

                    full_llm_no_results_message = "".join(llm_no_results_chunks).strip()
                    if full_llm_no_results_message:
                        no_results_message = full_llm_no_results_message
                except Exception as no_results_llm_error:
                    logger.warning(
                        f"Task {task_id}: No-results LLM guidance failed, using preset fallback: {no_results_llm_error}"
                    )

                if not llm_no_results_chunks:
                    await emit_event(
                        event_type=PipelineEventType.CHUNK,
                        event_data={
                            "type": "chunk",
                            "content": no_results_message,
                        },
                    )

                await container.pipeline_task_service.update_progress(
                    task_id=task_id,
                    phase=PipelinePhase.LLM_GENERATION,
                    progress_percent=85,
                )

                progress_events.append(
                    {
                        "type": "reasoning",
                        "timestamp": int(time.time() * 1000),
                        "content": "No relevant papers were retrieved. Returned guidance with suggested query reformulations.",
                        "metadata": {
                            "reason": "no_results",
                        },
                    }
                )

                pipeline_completion_time = int((time.time() - pipeline_start_time) * 1000)
                assistant_message_id = await container.conversation_service.add_message_to_conversation(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    message_text=no_results_message,
                    role="assistant",
                    paper_ids=[],
                    paper_snapshots=[],
                    progress_events=progress_events,
                    scoped_quote_refs=[],
                    client_message_id=client_message_id,
                    pipeline_type=pipeline_type,
                    completion_time_ms=pipeline_completion_time,
                )

                await container.pipeline_task_service.save_results(
                    task_id=task_id,
                    papers=[],
                    chunks=[],
                    response_text=no_results_message,
                )

                await container.pipeline_task_service.update_progress(
                    task_id=task_id,
                    phase=PipelinePhase.DONE,
                    progress_percent=100,
                )

                await emit_event(
                    event_type=PipelineEventType.DONE,
                    event_data={
                        "type": "done",
                        "status": "no_results",
                        "message_id": assistant_message_id,
                        "cited_papers": [],
                        "retrieved_count": 0,
                    },
                )

                await container.pipeline_task_service.complete_task(
                    task_id=task_id,
                    message_id=assistant_message_id,
                )

                logger.info(f"Task {task_id}: Completed with no results (graceful fallback)")
                return
            
            # Emit paper metadata
            from app.domain.papers.schemas import PaperMetadata
            papers_metadata = [PaperMetadata.from_ranked_paper(p) for p in rag_result.papers]
            await emit_event(
                event_type=PipelineEventType.METADATA,
                event_data={
                    "type": "papers_metadata",
                    "papers": [p.model_dump(mode='json', by_alias=True) for p in papers_metadata]
                },
            )
            
            # Update progress - LLM generation
            await container.pipeline_task_service.update_progress(
                task_id=task_id,
                phase=PipelinePhase.LLM_GENERATION,
                progress_percent=60
            )

            if route_decision.total_steps >= 3:
                await emit_step_event(
                    {
                        "type": "reasoning",
                        "content": "Generating answer from ranked evidence...",
                        "metadata": {"stage": "llm_generation"},
                        "phase": PipelinePhase.LLM_GENERATION,
                        "progress_percent": 75,
                    }
                )
            
            # Build context for LLM
            from app.domain.chat.response_builder import ChatResponseBuilder
            response_builder = ChatResponseBuilder()
            context, chunk_papers = response_builder.build_context_from_results(rag_result)
            retrieved_paper_ids = response_builder.get_retrieved_paper_ids(rag_result)
            paper_snapshots = response_builder.extract_metadata_from_results(rag_result)

            scoped_quote_index: Dict[str, Dict[str, Any]] = {}
            scoped_seen_markers: set[str] = set()
            scoped_quote_refs: List[Dict[str, Any]] = []
            scoped_text_buffer = ""
            prompt_name = "generate_answer"

            if route_decision.route_type == "scoped":
                prompt_name = "generate_answer_scoped"
                scoped_quote_index = _build_scoped_quote_index(rag_result)
            
            # Get conversation history
            from app.domain.conversations.context_manager import ConversationContextManager
            context_manager = ConversationContextManager(max_context_tokens=8000, max_messages=10)
            conversation_history, _ = await context_manager.get_conversation_context(
                conversation_id=conversation_id,
                db_session=db,
                include_current_query=False
            )
            
            enhanced_query = response_builder.build_enhanced_query(query, conversation_history)
            
            # Stream LLM response and save chunks
            assistant_response_chunks = []
            reasoning_chunks = []
            
            from app.extensions import get_stream_response_content, get_stream_response_reasoning
            from app.extensions.citation_extractor import CitationExtractor
            
            async for chunk_text in container.llm_service.stream_citation_based_response(
                query=enhanced_query,
                context=context,
                prompt_name=prompt_name,
            ):
                text = get_stream_response_content(chunk_text)
                reasoning_chunk = get_stream_response_reasoning(chunk_text)
                
                # Save reasoning chunks
                if reasoning_chunk and reasoning_chunk not in reasoning_chunks:
                    await emit_event(
                        event_type=PipelineEventType.REASONING,
                        event_data={
                            "type": "reasoning",
                            "content": reasoning_chunk
                        },
                    )
                    reasoning_chunks.append(reasoning_chunk)
                
                # Save text chunks
                if text:
                    await emit_event(
                        event_type=PipelineEventType.CHUNK,
                        event_data={
                            "type": "chunk",
                            "content": text
                        },
                    )
                    assistant_response_chunks.append(text)

                    if route_decision.route_type == "scoped":
                        scoped_text_buffer = (scoped_text_buffer + text)[-1200:]
                        refs = CitationExtractor.extract_scoped_citation_refs(scoped_text_buffer)
                        for ref in refs:
                            marker = str(ref.get("marker") or "")
                            if not marker or marker in scoped_seen_markers:
                                continue

                            scoped_seen_markers.add(marker)
                            key = f"{ref.get('paper_id')}|{ref.get('chunk_id')}"
                            quote_meta = scoped_quote_index.get(key)
                            if not quote_meta:
                                continue

                            quote_ref_payload = {
                                "paper_id": quote_meta.get("paper_id"),
                                "chunk_id": quote_meta.get("chunk_id"),
                                "section": quote_meta.get("section"),
                                "quote": quote_meta.get("quote"),
                                "char_start": ref.get("char_start"),
                                "char_end": ref.get("char_end"),
                                "marker": marker,
                            }

                            scoped_quote_refs.append(quote_ref_payload)

                            await emit_event(
                                event_type=PipelineEventType.METADATA,
                                event_data={
                                    "type": "quote_ref",
                                    **quote_ref_payload,
                                },
                            )
            
            # Build full response
            full_response = "".join(assistant_response_chunks)

            if reasoning_chunks:
                progress_events.append(
                    {
                        "type": "reasoning",
                        "timestamp": int(time.time() * 1000),
                        "content": "".join(reasoning_chunks),
                    }
                )
            
            # Extract citations
            from app.extensions.citation_extractor import CitationExtractor
            cited_paper_ids = CitationExtractor.extract_citations_from_text(full_response)
            
            # Create assistant message with same metadata style as legacy streaming flow
            pipeline_completion_time = int((time.time() - pipeline_start_time) * 1000)
            assistant_message_id = await container.conversation_service.add_message_to_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                message_text=full_response,
                role="assistant",
                paper_ids=retrieved_paper_ids,
                paper_snapshots=paper_snapshots,
                progress_events=progress_events,
                scoped_quote_refs=scoped_quote_refs,
                client_message_id=client_message_id,
                pipeline_type=pipeline_type,
                completion_time_ms=pipeline_completion_time,
            )
            
            # Save results to task
            serialized_papers = _serialize_ranked_papers_for_cache(rag_result)
            serialized_chunks = _serialize_chunks_for_cache(rag_result)

            await container.pipeline_task_service.save_results(
                task_id=task_id,
                papers=serialized_papers,
                chunks=serialized_chunks,
                response_text=full_response
            )
            
            # Update progress to 100%
            await container.pipeline_task_service.update_progress(
                task_id=task_id,
                phase=PipelinePhase.DONE,
                progress_percent=100
            )
            
            # Emit done event
            await emit_event(
                event_type=PipelineEventType.DONE,
                event_data={
                    "type": "done",
                    "status": "success",
                    "message_id": assistant_message_id,
                    "cited_papers": list(cited_paper_ids),
                    "retrieved_count": len(rag_result.papers)
                },
            )
            
            # Complete task
            await container.pipeline_task_service.complete_task(
                task_id=task_id,
                message_id=assistant_message_id
            )
            
            logger.info(f"Task {task_id}: Completed successfully")
        
        except Exception as e:
            logger.error(f"Task {task_id}: Failed with error: {e}", exc_info=True)
            
            # Save error event (container might be None if init failed)
            if container:
                try:
                    error_payload = {
                        "message": str(e),
                        "error_type": type(e).__name__
                    }

                    db_error_event = await container.pipeline_event_store.save_event(
                        task_id=task_id,
                        event_type=PipelineEventType.ERROR,
                        event_data=error_payload,
                    )

                    try:
                        await publish_task_event(
                            task_id=task_id,
                            event_type=PipelineEventType.ERROR,
                            sequence=db_error_event.sequence_number,
                            event_data=error_payload,
                        )
                    except Exception as redis_error:
                        logger.warning(
                            f"Task {task_id}: Redis stream publish failed for error event: {redis_error}"
                        )
                    
                    # Mark task as failed
                    await container.pipeline_task_service.complete_task(
                        task_id=task_id,
                        error_message=str(e)
                    )
                except Exception as inner_e:
                    logger.error(f"Task {task_id}: Failed to save error state: {inner_e}")
        
        finally:
            await engine.dispose()

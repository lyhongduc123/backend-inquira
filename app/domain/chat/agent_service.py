"""Agent-mode chat service powered by LangGraph orchestration."""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from openai import max_retries
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.container import ServiceContainer
from app.domain.chat.agent_graph import AgentGraphOrchestrator, AgentGraphState
from app.domain.chat.live_stream import get_live_task_stream_broker
from app.extensions.citation_extractor import CitationExtractor
from app.extensions.logger import create_logger
from app.models.pipeline_tasks import PipelineEventType, PipelinePhase, PipelineTaskStatus
from app.validation.schemas import ValidationRequest
from app.validation.repository import save_validation_result
from app.validation.service import validate_answer

logger = create_logger(__name__)

_RESPONSES_DIR = Path(__file__).resolve().parent / "responses"


def _load_response_template(filename: str, default: str) -> str:
    """Load canned response content from response template files."""
    try:
        content = (_RESPONSES_DIR / filename).read_text(encoding="utf-8").strip()
        return content or default
    except Exception:
        return default


def _build_step_event_from_rag_event(
    event_type: str,
    event_data: Any,
) -> Optional[Dict[str, Any]]:
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
            "content": (
                f"Filtering {total_papers} retrieved papers by content relevance, quality, authors,..."
            ),
            "metadata": {"total_papers": total_papers, "chunks": total_chunks},
            "phase": PipelinePhase.RANKING,
            "progress_percent": 50,
        }

    return None

class ChatAgentService:
    """Execute agent-mode workflow with live streaming and task tracking."""

    async def stream_agent_workflow(
        self,
        task_id: str,
        user_id: int,
        conversation_id: str,
        query: str,
        filters: Dict[str, Any],
    ) -> None:
        """Run LangGraph agent workflow and stream events in real time."""
        pipeline_start_time = time.time()

        engine = create_async_engine(settings.DATABASE_URL, echo=False)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)

        container = None
        async with session_factory() as db:
            try:
                container = ServiceContainer(db)
                task_record = await container.pipeline_task_service.get_task(task_id)
                client_message_id = task_record.client_message_id if task_record else None

                progress_events: List[Dict[str, Any]] = []
                emitted_step_types: set[str] = set()

                stream_broker = get_live_task_stream_broker()
                live_stream_event_types = {
                    PipelineEventType.STEP,
                    "progress",
                    PipelineEventType.METADATA,
                    PipelineEventType.REASONING,
                    PipelineEventType.CHUNK,
                    PipelineEventType.ERROR,
                    PipelineEventType.DONE,
                }

                async def emit_event(event_type: str, event_data: Dict[str, Any]) -> None:
                    if event_type not in live_stream_event_types:
                        return
                    await stream_broker.publish(task_id=task_id, event_type=event_type, data=event_data)

                async def emit_step_event(mapped_step_event: Dict[str, Any]) -> None:
                    step_type = str(mapped_step_event.get("type", "")).strip()
                    if not step_type or step_type in emitted_step_types:
                        return

                    emitted_step_types.add(step_type)

                    existing_metadata = mapped_step_event.get("metadata")
                    metadata: Dict[str, Any] = (
                        dict(existing_metadata)
                        if isinstance(existing_metadata, dict)
                        else {}
                    )

                    payload = {
                        "type": step_type,
                        "content": str(mapped_step_event.get("content", "")),
                        "pipeline_type": "agent",
                        "phase": mapped_step_event.get("phase", PipelinePhase.INIT),
                        "progress_percent": mapped_step_event.get("progress_percent", 10),
                        "metadata": metadata,
                    }

                    await emit_event("progress", payload)

                    progress_events.append(
                        {
                            "type": payload.get("type"),
                            "pipeline_type": payload.get("pipeline_type"),
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

                await container.pipeline_task_service.update_status(
                    task_id=task_id,
                    status=PipelineTaskStatus.RUNNING,
                )

                user_message = await container.message_service.create_message(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    role="user",
                    content=query,
                )

                from app.domain.conversations.context_manager import ConversationContextManager

                context_manager = ConversationContextManager(
                    max_context_tokens=8000,
                    max_messages=10,
                )
                conversation_history, _ = await context_manager.get_conversation_context(
                    conversation_id=conversation_id,
                    db_session=db,
                    include_current_query=True,
                    exclude_message_id=user_message.id,
                )

                async def on_graph_event(payload: Dict[str, Any]) -> None:
                    if payload.get("event") == "rag_event":
                        mapped = _build_step_event_from_rag_event(
                            str(payload.get("type", "")),
                            payload.get("data", {}),
                        )
                        if mapped:
                            await emit_step_event(mapped)
                        return

                    if payload.get("event") == "step":
                        step_payload = payload.get("step", {})
                        if isinstance(step_payload, dict):
                            await emit_step_event(step_payload)
                        return

                    if payload.get("event") == "metadata":
                        data = payload.get("data", {})
                        if isinstance(data, dict):
                            await emit_event(PipelineEventType.METADATA, data)
                        return

                    if payload.get("event") == "reasoning":
                        data = payload.get("data", {})
                        if isinstance(data, dict):
                            await emit_event(PipelineEventType.REASONING, data)
                            reasoning_text = str(data.get("content", "")).strip()
                            if reasoning_text:
                                progress_events.append(
                                    {
                                        "type": "reasoning",
                                        "timestamp": int(time.time() * 1000),
                                        "content": reasoning_text,
                                    }
                                )
                        return

                    if payload.get("event") == "chunk":
                        data = payload.get("data", {})
                        if isinstance(data, dict):
                            await emit_event(PipelineEventType.CHUNK, data)
                        return

                orchestrator = AgentGraphOrchestrator(container=container, on_event=on_graph_event)
                graph_state = await orchestrator.run(
                    AgentGraphState(
                        query=query,
                        filters=filters or {},
                        conversation_history=conversation_history,
                        num_retries=0
                    )
                )

                direct_response = str(graph_state.get("direct_response") or "").strip()
                full_response = str(graph_state.get("response_text") or "").strip() or direct_response

                if direct_response:

                    completion_time_ms = int((time.time() - pipeline_start_time) * 1000)
                    assistant_message_id = await container.conversation_service.add_message_to_conversation(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        message_text=full_response,
                        role="assistant",
                        paper_ids=[],
                        paper_snapshots=[],
                        progress_events=progress_events,
                        scoped_quote_refs=[],
                        client_message_id=client_message_id,
                        pipeline_type="agent",
                        completion_time_ms=completion_time_ms,
                    )

                    await container.pipeline_task_service.save_results(
                        task_id=task_id,
                        papers=[],
                        chunks=[],
                        response_text=full_response,
                    )

                    await container.pipeline_task_service.update_progress(
                        task_id=task_id,
                        phase=PipelinePhase.DONE,
                        progress_percent=100,
                    )

                    await emit_event(
                        PipelineEventType.DONE,
                        {
                            "type": "done",
                            "status": "success",
                            "message_id": assistant_message_id,
                            "cited_papers": [],
                            "retrieved_count": 0,
                        },
                    )

                    await container.pipeline_task_service.complete_task(
                        task_id=task_id,
                        message_id=assistant_message_id,
                    )
                    logger.info("Agent task %s: completed with direct graph response", task_id)
                    return

                rag_result = graph_state.get("final_rag_result")
                if not rag_result or not rag_result.papers:
                    logger.warning("Agent task %s: no references found", task_id)
                    await emit_event(
                        PipelineEventType.CHUNK,
                        {
                            "type": "chunk",
                            "content": _load_response_template(
                                "not_found.txt",
                                "I couldn't find any relevant research papers for your query.",
                            ),
                        },
                    )
                    await container.pipeline_task_service.complete_task(
                        task_id,
                        error_message="No references found.",
                    )
                    return
                completion_time_ms = int((time.time() - pipeline_start_time) * 1000)

                retrieved_paper_ids = list(graph_state.get("retrieved_paper_ids") or [])
                paper_snapshots = list(graph_state.get("paper_snapshots") or [])
                context_str = str(graph_state.get("response_context") or "")
                context_chunks = list(graph_state.get("response_context_chunks") or [])

                if not full_response:
                    logger.warning("Agent task %s: graph did not produce response text", task_id)
                    await container.pipeline_task_service.complete_task(
                        task_id,
                        error_message="No response generated.",
                    )
                    return

                assistant_message_id = await container.conversation_service.add_message_to_conversation(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    message_text=full_response,
                    role="assistant",
                    paper_ids=retrieved_paper_ids,
                    paper_snapshots=paper_snapshots,
                    progress_events=progress_events,
                    scoped_quote_refs=[],
                    client_message_id=client_message_id,
                    pipeline_type="agent",
                    completion_time_ms=completion_time_ms,
                )

                try:
                    validation_request = ValidationRequest(
                        query=query,
                        context=context_str,
                        enhanced_query=query,
                        context_chunks=context_chunks,
                        generated_answer=full_response,
                        model_name=container.llm_service.llm_provider.get_model(),
                        message_id=assistant_message_id,
                    )
                    validation_result = await validate_answer(validation_request)
                    await save_validation_result(db, validation_request, validation_result)
                except Exception as validation_error:
                    logger.error("Agent validation error: %s", validation_error)

                from app.domain.chat.executor import (
                    _serialize_chunks_for_cache,
                    _serialize_ranked_papers_for_cache,
                )

                await container.pipeline_task_service.save_results(
                    task_id=task_id,
                    papers=_serialize_ranked_papers_for_cache(rag_result),
                    chunks=_serialize_chunks_for_cache(rag_result),
                    response_text=full_response,
                )

                await container.pipeline_task_service.update_progress(
                    task_id=task_id,
                    phase=PipelinePhase.DONE,
                    progress_percent=100,
                )

                cited_paper_ids = CitationExtractor.extract_citations_from_text(full_response)
                await emit_event(
                    PipelineEventType.DONE,
                    {
                        "type": "done",
                        "status": "success",
                        "message_id": assistant_message_id,
                        "cited_papers": list(cited_paper_ids),
                        "retrieved_count": len(rag_result.papers),
                    },
                )

                await container.pipeline_task_service.complete_task(
                    task_id=task_id,
                    message_id=assistant_message_id,
                )
                logger.info("Agent task %s: completed", task_id)

            except Exception as exc:
                logger.error("Agent task %s failed: %s", task_id, exc, exc_info=True)
                if container:
                    try:
                        await get_live_task_stream_broker().publish(
                            task_id,
                            PipelineEventType.ERROR,
                            {"message": str(exc), "error_type": type(exc).__name__},
                        )
                        await container.pipeline_task_service.complete_task(
                            task_id,
                            error_message=str(exc),
                        )
                    except Exception as inner:
                        logger.error("Error finalizing agent task %s: %s", task_id, inner)
            finally:
                await engine.dispose()

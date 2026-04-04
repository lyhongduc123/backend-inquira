# app/domain/chat/agent_service.py
"""
Chat Agent Service

This service orchestration class replaces the standard chat executor for
Agent Mode queries. It acts as an explicit State Machine:

- State 1: Query Router (Identify simple vs complex)
- State 2: Decomposition (Extract filters, sub-queries via strict JSON)
- State 3: Execution (Call pure AgentPipeline)
- State 4: Multimodal Synthesis (Extract top R2 figure URLs, stream to VLM)
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.core.config import settings
from app.core.container import ServiceContainer
from app.models.pipeline_tasks import PipelineTaskStatus, PipelinePhase, PipelineEventType
from app.extensions.logger import create_logger

from app.domain.chat.live_stream import get_live_task_stream_broker
from app.domain.chat.response_builder import ChatResponseBuilder
from app.validation.service import save_validation_result, validate_answer
from app.validation.schemas import ValidationRequest

logger = create_logger(__name__)


def _build_step_event_from_rag_event(event_type: str, event_data: Any) -> Optional[Dict[str, Any]]:
    payload = event_data if isinstance(event_data, dict) else {}

    if event_type in ("search_queries", "searching"):
        queries = payload.get("queries", [])
        return {
            "type": "searching",
            "content": "Agent is executing explicit sub-queries against academic corpus...",
            "metadata": {"queries": queries},
            "phase": PipelinePhase.SEARCH,
            "progress_percent": 30,
        }

    if event_type == "ranking":
        total_papers = payload.get("total_papers", 0)
        total_chunks = payload.get("total_chunks", 0)
        return {
            "type": "ranking",
            "content": f"Agent is ranking {total_papers} retrieved papers and extracting figures...",
            "metadata": {"total_papers": total_papers, "chunks": total_chunks},
            "phase": PipelinePhase.RANKING,
            "progress_percent": 60,
        }

    return None


class ChatAgentService:
    """Orchestrates the Agentic State Machine execution."""

    async def stream_agent_workflow(
        self,
        task_id: str,
        user_id: int,
        conversation_id: str,
        query: str,
        filters: Dict[str, Any]
    ) -> None:
        """
        Executes the 4-State Agent Workflow asynchronously.
        """
        pipeline_start_time = time.time()

        # Worker DB session
        engine = create_async_engine(settings.DATABASE_URL, echo=False)
        async_session = async_sessionmaker(engine, expire_on_commit=False)
        
        container = None
        
        async with async_session() as db:
            try:
                container = ServiceContainer(db)
                task_record = await container.pipeline_task_service.get_task(task_id)
                client_message_id = task_record.client_message_id if task_record else None
                progress_events: List[Dict[str, Any]] = []
                
                emitted_step_types: set[str] = set()
                current_step = 0
                total_steps = 4  # Decompose, Execute, Extract Figures, Synthesize

                stream_broker = get_live_task_stream_broker()

                async def emit_event(event_type: str, event_data: Dict[str, Any]) -> None:
                    if event_type not in {PipelineEventType.CHUNK, PipelineEventType.ERROR, PipelineEventType.DONE}:
                        return
                    await stream_broker.publish(task_id=task_id, event_type=event_type, data=event_data)

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
                        "total_steps": total_steps,
                    }

                    await stream_broker.publish(task_id=task_id, event_type=PipelineEventType.STEP, data=payload)
                    progress_events.append({
                        "type": payload.get("type"),
                        "timestamp": int(time.time() * 1000),
                        "content": payload.get("content"),
                        "metadata": payload.get("metadata"),
                    })

                    await container.pipeline_task_service.update_progress(
                        task_id=task_id, phase=payload.get("phase", PipelinePhase.INIT), progress_percent=payload.get("progress_percent", 10))

                
                # Setup base task status
                await container.pipeline_task_service.update_status(task_id=task_id, status=PipelineTaskStatus.RUNNING)
                await stream_broker.publish(task_id=task_id, event_type=PipelineEventType.STEP, data={"type": "step_count", "total_steps": total_steps})
                
                # Save user message
                user_message = await container.message_service.create_message(
                    user_id=user_id, conversation_id=conversation_id, role="user", content=query
                )

                # ==========================================
                # STATE 1 & 2: Query Router + Decomposition
                # ==========================================
                await emit_step_event({
                    "type": "reasoning",
                    "content": "Agent is decomposing user query to build explicit search strategy...",
                    "phase": PipelinePhase.INIT,
                    "progress_percent": 10,
                })
                
                from app.domain.conversations.context_manager import ConversationContextManager
                context_manager = ConversationContextManager(max_context_tokens=8000, max_messages=10)
                conversation_history, _ = await context_manager.get_conversation_context(
                    conversation_id=conversation_id, db_session=db, include_current_query=False
                )

                breakdown = await container.llm_service.decompose_user_query_v2(
                    user_question=query, conversation_history=conversation_history
                )
                
                merged_filters = filters or {}
                if breakdown.filters:
                    merged_filters.update(breakdown.filters)

                # ==========================================
                # STATE 3: Pipeline Execution
                # ==========================================
                rag_result = None
                from app.rag_pipeline.schemas import RAGResult
                
                async for event in container.agent_pipeline.run_explicit_workflow(
                    original_query=query,
                    search_queries=breakdown.search_queries or [query],
                    intent=breakdown.intent,
                    filters=merged_filters,
                ):
                    mapped_step = _build_step_event_from_rag_event(event.type, event.data)
                    if mapped_step:
                        await emit_step_event(mapped_step)
                        
                    if event.type == "result":
                        rag_result = event.data if isinstance(event.data, RAGResult) else None

                if not rag_result or not rag_result.papers:
                    logger.warning(f"Agent Task {task_id}: No references found.")
                    await emit_event(PipelineEventType.CHUNK, {"type": "chunk", "content": "I couldn't find any relevant papers for your query."})
                    await container.pipeline_task_service.complete_task(task_id, error_message="No references found.")
                    return

                # Emit metadata to UI
                from app.domain.papers.schemas import PaperMetadata
                papers_metadata = [PaperMetadata.from_ranked_paper(p) for p in rag_result.papers]
                await stream_broker.publish(task_id=task_id, event_type=PipelineEventType.METADATA, data={
                    "type": "papers_metadata", "papers": [p.model_dump(mode='json', by_alias=True) for p in papers_metadata]
                })

                # ==========================================
                # STATE 4: Multimodal Synthesis & Figure Extraction
                # ==========================================
                await emit_step_event({
                    "type": "reasoning",
                    "content": "Agent is extracting R2 figures and synthesizing multimodal response...",
                    "phase": PipelinePhase.LLM_GENERATION,
                    "progress_percent": 75,
                })

                rb = ChatResponseBuilder()
                context_str, chunk_papers = rb.build_context_from_results(rag_result)
                retrieved_paper_ids = rb.get_retrieved_paper_ids(rag_result)
                paper_snapshots = rb.extract_metadata_from_results(rag_result)

                # Figure Extraction (Top 3)
                figure_urls = set()
                top_chunks = rag_result.chunks[:top_chunks] if hasattr(rag_result, 'chunks') else []
                for chunk in top_chunks:
                    meta = getattr(chunk, "docling_metadata", None) or {}
                    assets = meta.get("figure_assets", [])
                    for asset in assets:
                        if "url" in asset and asset["url"].startswith("http"):
                            figure_urls.add(asset["url"])
                    if len(figure_urls) >= 3:
                        break  # Limit strictly to top 3 images!
                
                limited_figure_urls = list(figure_urls)[:3]

                # Format system prompt payload manually to inject Image URLs for Vision LLMs
                system_prompt_str = f"Question: {breakdown.clarified_question}\n\n{context_str}"
                
                content_payload = [{"type": "text", "text": system_prompt_str}]
                for img_url in limited_figure_urls:
                    content_payload.append({"type": "image_url", "image_url": {"url": img_url}})
                    
                messages = [
                    {"role": "system", "content": "You are a Research Assistant. Analyze the text and available charts."},
                    {"role": "user", "content": content_payload}
                ]
                
                assistant_response_chunks = []
                reasoning_chunks = []
                
                from app.extensions import get_stream_response_content, get_stream_response_reasoning
                from app.llm.prompts import PromptPresets
                
                config = PromptPresets.merge_with_overrides(PromptPresets.FACTUAL)

                try:
                    for chunk in container.llm_service.llm_provider.stream_completion(messages=messages, **config):
                        text = get_stream_response_content(chunk)
                        reasoning = get_stream_response_reasoning(chunk)
                        
                        if reasoning and reasoning not in reasoning_chunks:
                            await stream_broker.publish(task_id, PipelineEventType.REASONING, {"type": "reasoning", "content": reasoning})
                            reasoning_chunks.append(reasoning)

                        if text:
                            await emit_event(PipelineEventType.CHUNK, {"type": "chunk", "content": text})
                            assistant_response_chunks.append(text)
                
                except Exception as stream_err:
                    logger.error(f"Agent streaming failed: {stream_err}")
                    raise stream_err
                
                # Clean up and finalize
                full_response = "".join(assistant_response_chunks)
                pipeline_completion_time = int((time.time() - pipeline_start_time) * 1000)
                
                if reasoning_chunks:
                    progress_events.append({"type": "reasoning", "timestamp": int(time.time() * 1000), "content": "".join(reasoning_chunks)})

                assistant_message_id = await container.conversation_service.add_message_to_conversation(
                    conversation_id=conversation_id, user_id=user_id, message_text=full_response, role="assistant",
                    paper_ids=retrieved_paper_ids, paper_snapshots=paper_snapshots, progress_events=progress_events,
                    scoped_quote_refs=[], client_message_id=client_message_id, pipeline_type="agent", completion_time_ms=pipeline_completion_time,
                )

                # Validation
                try:
                    val_req = ValidationRequest(
                        query=query, context=context_str, enhanced_query=breakdown.clarified_question,
                        context_chunks=rb.extract_context_chunks_from_results(rag_result),
                        generated_answer=full_response, model_name=container.llm_service.llm_provider.get_model(),
                        message_id=assistant_message_id
                    )
                    await save_validation_result(db, val_req, await validate_answer(val_req))
                except Exception as e:
                    logger.error(f"Agent Validation error: {e}")

                # Save Results
                from app.domain.chat.executor import _serialize_ranked_papers_for_cache, _serialize_chunks_for_cache
                await container.pipeline_task_service.save_results(
                    task_id=task_id, papers=_serialize_ranked_papers_for_cache(rag_result),
                    chunks=_serialize_chunks_for_cache(rag_result), response_text=full_response
                )
                
                await container.pipeline_task_service.update_progress(task_id=task_id, phase=PipelinePhase.DONE, progress_percent=100)
                
                # Output DOING event so frontend knows we are finished streaming
                from app.extensions.citation_extractor import CitationExtractor
                cited_paper_ids = CitationExtractor.extract_citations_from_text(full_response)
                
                await emit_event(PipelineEventType.DONE, {
                    "type": "done", "status": "success", "message_id": assistant_message_id,
                    "cited_papers": list(cited_paper_ids), "retrieved_count": len(rag_result.papers)
                })
                
                await container.pipeline_task_service.complete_task(task_id=task_id, message_id=assistant_message_id)
                logger.info(f"Agent Task {task_id}: Completed seamlessly.")

            except Exception as e:
                logger.error(f"Agent Task {task_id} Failed: {e}", exc_info=True)
                if container:
                    try:
                        await get_live_task_stream_broker().publish(task_id, PipelineEventType.ERROR, {"message": str(e), "error_type": type(e).__name__})
                        await container.pipeline_task_service.complete_task(task_id, error_message=str(e))
                    except Exception as inner:
                        logger.error(f"Error finalizing Agent task: {inner}")
            finally:
                await engine.dispose()

"""LangGraph orchestration for Agent-mode chat workflow."""

from __future__ import annotations

from pathlib import Path
import stat
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired

from app.core.container import ServiceContainer
from app.domain.chat.agent.tools import AgentTools
from app.extensions.context_builder import ContextBuilder
from app.extensions import get_stream_response_content, get_stream_response_reasoning
from app.extensions.logger import create_logger
from app.extensions.stream import stream_like_llm
from app.domain.papers.schemas import PaperMetadata
from app.llm.schemas.chat import GeneratedQueryPlanResponse, QueryIntent
from app.models.pipeline_tasks import PipelinePhase
from app.rag_pipeline.schemas import RAGResult

logger = create_logger(__name__)
_RESPONSES_DIR = Path(__file__).resolve().parent / "responses"

GraphEventCallback = Callable[[Dict[str, Any]], Awaitable[None]]
MAX_LOCAL_RETRIES = 2
MAX_EXTERNAL_RETRIES = 2


def _build_gibberish_response() -> str:
    default_response = (
        "Hello! I'm Inquira, an academic research assistant. "
        "Please ask a clear research question in a full sentence so I can find relevant papers."
    )
    try:
        content = (_RESPONSES_DIR / "gibberish.txt").read_text(encoding="utf-8").strip()
        return content or default_response
    except Exception:
        return default_response


def _build_system_explanation_response() -> str:
    default_response = (
        "Inquira is an academic research assistant with a retrieval-first workflow.\n\n"
        "1) It decomposes your question into focused academic queries and intent.\n"
        "2) It searches local indexed papers first.\n"
        "3) If local coverage is weak, it expands queries and retries locally once.\n"
        "4) If local evidence is still weak, it escalates to Semantic Scholar and ingests returned papers.\n"
        "5) It generates a citation-grounded answer from the retrieved evidence.\n\n"
        "If you want, ask a specific topic and I can run the full retrieval pipeline."
    )
    try:
        content = (_RESPONSES_DIR / "system.txt").read_text(encoding="utf-8").strip()
        return content or default_response
    except Exception:
        return default_response


class AgentGraphState(TypedDict):
    query: str
    filters: Dict[str, Any]
    conversation_history: List[Dict[str, str]]

    plan: NotRequired[GeneratedQueryPlanResponse]
    search_queries: NotRequired[List[str]]
    result: NotRequired[Optional[RAGResult]] 
    results_array: NotRequired[List[RAGResult]]  
    local_retries: NotRequired[int]
    external_retries: NotRequired[int]
    enough_evidence: NotRequired[bool]  
    stop_reason: NotRequired[str]
    direct_response: NotRequired[str]
    response_text: NotRequired[str]


class AgentGraphOrchestrator:
    """LangGraph orchestrator for agent-mode chat workflow."""

    def __init__(
        self,
        container: ServiceContainer,
        on_event: Optional[GraphEventCallback] = None,
    ) -> None:
        self.container = container
        self.tools = AgentTools(container)
        self._on_event = on_event
        self._graph = self._build_graph()

    def get_graph(self):
        return self._graph

    def _build_graph(self):
        builder = StateGraph(AgentGraphState)
        builder.add_node("decompose", self._node_decompose)
        builder.add_node("preresponse", self._node_preresponse)
        builder.add_node("general_agent", self._node_general_agent)
        builder.add_node("local_retrieve", self._node_local_retrieve)
        builder.add_node("evaluate_local", self._node_evaluate_local)
        builder.add_node("expand_queries", self._node_expand_queries)
        builder.add_node("retry_local_retrieve", self._node_retry_local_retrieve)
        builder.add_node("external_retrieve", self._node_external_retrieve)
        builder.add_node("evaluate_external", self._node_evaluate_external)
        builder.add_node("merge_results_rrf", self._node_merge_results_rrf)
        builder.add_node("answer", self._node_answer)
        builder.add_node("expand_external_queries", self._node_expand_external_queries)

        builder.add_edge(START, "decompose")
        builder.add_conditional_edges(
            "decompose",
            self._route_after_decompose,
            {
                "need_predefined_response": "preresponse",
                "is_general_question": "general_agent",
                "is_research_question": "local_retrieve",
            },
        )
        builder.add_edge("preresponse", END)
        builder.add_edge("general_agent", END)
        builder.add_edge("local_retrieve", "merge_results_rrf")
        builder.add_edge("merge_results_rrf", "evaluate_local")
        builder.add_conditional_edges(
            "evaluate_local",
            self._route_gate,
            {
                "local": "expand_queries",
                "external": "expand_external_queries",
                "answer": "answer"
            }
        )
        builder.add_edge("expand_queries", "local_retrieve")
        builder.add_edge("expand_external_queries", "external_retrieve")
        # builder.add_edge("retry_local_retrieve", "merge_results_rrf")
        # builder.add_conditional_edges(
        #     "evaluate_local",
        #     self._route_after_local_evaluation,
        #     {
        #         "answer": "answer",
        #         "expand_queries": "expand_queries",
        #         "external_retrieve": "external_retrieve",
        #     },
        # )
        # builder.add_conditional_edges(
        #     "expand_queries",
        #     self._route_after_expand_queries,
        #     {
        #         "retry_local_retrieve": "local_retrieve",
        #         "external_retrieve": "external_retrieve",
        #     },
        # )
        # builder.add_edge("retry_local_retrieve", "evaluate_local")
        # builder.add_edge("external_retrieve", "evaluate_external")
        builder.add_edge("external_retrieve", "merge_results_rrf")
        # builder.add_conditional_edges(
        #     "evaluate_external",
        #     self._route_after_external_evaluation,
        #     {
        #         "answer": "answer",
        #         "expand_queries": "expand_queries",
        #     },
        # )
        builder.add_edge("answer", END)
        return builder.compile()

    async def run(self, state: AgentGraphState) -> AgentGraphState:
        """Run the graph and return the final state."""
        logger.info(
        "--- Starting Agent Graph Execution for Query: '%s' ---",
        state.get("query")
    )
        final_state = state
        async for event in self._graph.astream(
            state,
            stream_mode="updates",
        ):
            for node_name, node_output in event.items():
                logger.info("=== NODE EXECUTED: %s ===", node_name)
                if isinstance(node_output, dict):
                    for key, value in node_output.items():
                        if isinstance(value, list):
                            logger.debug(
                                "Node '%s' updated '%s' with %d items",
                                node_name,
                                key,
                                len(value),
                            )
                        else:
                            logger.debug(
                                "Node '%s' updated '%s' = %s",
                                node_name,
                                key,
                                str(value)[:200],
                            )
                    final_state.update(node_output)
        logger.info("--- Agent Graph Execution Completed ---")
        return cast(AgentGraphState, final_state)

    async def _emit(self, payload: Dict[str, Any]) -> None:
        if not self._on_event:
            return
        try:
            # Normalize payload into frontend-compatible event object.
            # Expected shape downstream (ChatEventEmitter style):
            # - event: outer event channel (e.g., "progress", "chunk", "conversation")
            # - type / pipeline_type / content / metadata / phase / progress_percent
            event_type = (
                payload.get("event")
                or payload.get("event_type")
                or payload.get("type")
            )

            # Build data fields preferring explicit keys on the payload,
            # but accept nested `data` payloads when present.
            raw_data = payload.get("data")
            nested = raw_data if isinstance(raw_data, dict) else {}

            data_type = payload.get("type") or (nested.get("type") if isinstance(nested, dict) else None) or None
            content = payload.get("content") or (nested.get("content") if isinstance(nested, dict) else None) or ""
            metadata = payload.get("metadata") or (nested.get("metadata") if isinstance(nested, dict) else None) or {}

            event_obj: Dict[str, Any] = {
                "event": event_type,
                "type": data_type or event_type,
                "pipeline_type": "agent",
                "content": content,
                "metadata": metadata,
            }

            # preserve optional progress tracking fields when provided
            if "phase" in payload:
                event_obj["phase"] = payload["phase"]
            if "progress_percent" in payload:
                event_obj["progress_percent"] = payload["progress_percent"]

            await self._on_event(event_obj)
        except Exception:
            logger.exception("Failed to call on_event callback")

    @staticmethod
    def _route_after_decompose(state: AgentGraphState) -> str:
        plan = state.get("plan")
        if plan and plan.intent in {QueryIntent.SYSTEM, QueryIntent.GIBBERISH}:
            return "need_predefined_response"
        if plan and plan.intent == QueryIntent.GENERAL:
            return "is_general_question"
        return "is_research_question"

    @staticmethod
    def _route_after_local_evaluation(state: AgentGraphState) -> str:
        if state.get("enough_evidence"):
            return "answer"
        if state.get("local_retries", 0) < MAX_LOCAL_RETRIES:
            return "expand_queries"
        return "external_retrieve"

    @staticmethod
    def _route_after_expand_queries(state: AgentGraphState) -> str:
        if state.get("local_retries", 0) < MAX_LOCAL_RETRIES:
            return "retry_local_retrieve"
        return "external_retrieve"

    @staticmethod
    def _route_after_external_evaluation(state: AgentGraphState) -> str:
        if state.get("enough_evidence"):
            return "answer"
        if state.get("external_retries", 0) < MAX_EXTERNAL_RETRIES:
            return "expand_queries"
        return "answer"

    async def _node_decompose(self, state: AgentGraphState) -> AgentGraphState:
        query = state["query"]
        history = state.get("conversation_history", [])
        filters = state.get("filters", {})

        await self._emit(
            {
                "event": "progress",
                "type": "thinking",
                "content": "Understanding your question...",
                "phase": PipelinePhase.INIT,
                "progress_percent": 15,
                "metadata": {},
            }
        )

        plan = await self.tools.decompose_query(
            query=query,
            conversation_history=history,
            filters=filters,
        )

        initial_queries = plan.hybrid_queries or [plan.clarified_question or query]
        return {
            **state,
            "plan": plan,
            "filters": plan.filters or filters,
            "search_queries": initial_queries,
            "local_retries": 0,
            "external_retries": 0,
            "enough_evidence": False,
            "results_array": [],  # NEW: initialize results array for RRF merging
        }

    async def _node_preresponse(self, state: AgentGraphState) -> AgentGraphState:
        plan = state.get("plan")
        intent = plan.intent if plan else None
        if intent == QueryIntent.GIBBERISH:
            direct_response = _build_gibberish_response()
        else:
            direct_response = _build_system_explanation_response()

        for evt in stream_like_llm(direct_response):
            await self._emit(
                {
                    "event": "chunk",
                    "data": {
                        "type": "chunk",
                        "content": evt,
                    },
                }
            )

        return {
            **state,
            "direct_response": direct_response,
            "response_text": direct_response,
        }

    async def _node_general_agent(self, state: AgentGraphState) -> AgentGraphState:
        system_prompt = (
            "You are Inquira, an academic research assistant. Answer as a helpful assistant for normal user requests "
            "that do not require academic retrieval. Be concise, clear, and avoid claiming citations or external retrieval was used."
        )
        query = state.get("query") or ""
        try:
            conversation_history = state.get("conversation_history", [])
            full_response = ""
            async for evt in self.container.llm_service.stream_response(
                history_messages=conversation_history,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
            ):
                reasoning = get_stream_response_reasoning(evt)
                if reasoning:
                    await self._emit(
                        {
                            "event": "chunk",
                            "data": {
                                "type": "reasoning",
                                "content": reasoning,
                            },
                        }
                    )
                    break
                text = get_stream_response_content(evt)
                if text:
                    full_response += text
                    await self._emit(
                        {
                            "event": "chunk",
                            "data": {
                                "type": "chunk",
                                "content": text,
                            },
                        }
                    )

            if not full_response:
                raise ValueError("No response from LLM")
            return {
                **state,
                "direct_response": full_response,
                "response_text": full_response,
            }
        except Exception as exc:
            logger.warning("General direct response generation failed: %s", exc)

        fallback = (
            "Sorry, I couldn't generate a response to that request. Please try asking in a different way or ask about a specific research topic for me to assist with."
        )
        for evt in stream_like_llm(fallback):
            await self._emit(
                {
                    "event": "chunk",
                    "data": {
                        "type": "chunk",
                        "content": evt,
                    },
                }
            )
        return {
            **state,
            "direct_response": fallback,
            "response_text": fallback,
        }

    async def _node_local_retrieve(self, state: AgentGraphState) -> AgentGraphState:
        plan = state.get("plan")
        if not plan:
            return state
        
        if not state.get("search_queries"):
            return {
                **state,
                "local_retries": state.get("local_retries", 0) + 1,
            }
        
        
        await self._emit(
            {
                "event": "progress",
                "type": "searching",
                "content": "Searching relevant works and documents",
                "phase": PipelinePhase.SEARCH,
                "progress_percent": 25,
                "metadata": {
                    "queries": state.get("search_queries", []) or [state["query"]],
                },
            }
        )

        result = await self.tools.retrieve_local(
            query=state["query"],
            filters=state.get("filters", {}),
            plan=plan,
            search_queries=state.get("search_queries"),
        )
        
        results_array = state.get("results_array", []).copy()
        if result and (result.papers or result.chunks):
            results_array.append(result)
        
        return {
            **state,
            "result": result, 
            "results_array": results_array,
            "local_retries": state.get("local_retries", 0) + 1,
        }

    async def _node_evaluate_local(self, state: AgentGraphState) -> AgentGraphState:
        result = state.get("result")
        if not result:
            return {
                **state,
                "enough_evidence": False,
                "stop_reason": "no_local_results",
            }

        await self._emit(
            {
                "event": "progress",
                "type": "ranking",
                "content": (
                    f"Filtering {len(result.papers)} retrieved papers by content relevance, quality, authors,..."
                ),
                "phase": PipelinePhase.RANKING,
                "progress_percent": 50,
                "metadata": {
                    "total_papers": len(result.papers),
                    "chunks": len(result.chunks),
                },
            }
        )

        enough_evidence = self.tools.evaluate_sufficiency(result=result, stage="local")
        return {
            **state,
            "enough_evidence": enough_evidence,
            "stop_reason": "sufficient" if enough_evidence else "insufficient_local_evidence",
        }

    async def _node_expand_queries(self, state: AgentGraphState) -> AgentGraphState:
        plan = state.get("plan")
        base_query = plan.clarified_question if plan else state["query"]
        search_queries = await self.tools.expand_queries(
            query=base_query,
            rag_result=state.get("result"),
            existing_queries=state.get("search_queries", []),
        )
        if not search_queries:
            search_queries = state.get("search_queries", [])

        return {
            **state,
            "search_queries": search_queries,
        }
        
    async def _node_expand_external_queries(self, state: AgentGraphState) -> AgentGraphState:
        plan = state.get("plan") 
        base_query = plan.clarified_question if plan else state["query"]
        search_queries = await self.tools.expand_external_queries(
            query=base_query,
            rag_result=state.get("result"),
            existing_queries=state.get("search_queries", []),
        )
        if not search_queries:
            search_queries = state.get("search_queries", [])

        return {
            **state,
            "search_queries": search_queries,
        }

    async def _node_retry_local_retrieve(self, state: AgentGraphState) -> AgentGraphState:
        updated_state = {
            **state,
            "local_retries": state.get("local_retries", 0) + 1,
        }
        plan = updated_state.get("plan")
        if not plan:
            return cast(AgentGraphState, updated_state)

        result = await self.tools.retrieve_local(
            query=updated_state["query"],
            filters=updated_state.get("filters", {}),
            plan=plan,
            search_queries=updated_state.get("search_queries"),
        )
        
        results_array = updated_state.get("results_array", []).copy()
        if result and (result.papers or result.chunks):
            results_array.append(result)
        
        return cast(
            AgentGraphState,
            {
            **updated_state,
            "result": result, 
            "results_array": results_array,
            },
        )

    async def _node_external_retrieve(self, state: AgentGraphState) -> AgentGraphState:
        plan = state.get("plan")
        if not plan:
            return state
        
        await self._emit(
            {
                "event": "progress",
                "type": "searching_external",
                "content": "Expanding search to external sources...",
                "phase": PipelinePhase.SEARCH,
                "progress_percent": 45,
                "metadata": {
                    "queries": state.get("search_queries", []),
                },
            }
        )

        updated_state = {
            **state,
            "external_retries": state.get("external_retries", 0) + 1,
        }
        result = await self.tools.retrieve_external(
            query=updated_state["query"],
            filters=updated_state.get("filters", {}),
            search_queries=updated_state.get("search_queries"),
        )
        
        # NEW: apply reranker to external results
        if result and (result.papers or result.chunks):
            result = await self.tools.rerank_external_results(
                result=result,
                query=updated_state["query"],
            )
            
            # Append reranked result to results_array for RRF merging
            results_array = updated_state.get("results_array", []).copy()
            results_array.append(result)
            updated_state["results_array"] = results_array
        
        return cast(
            AgentGraphState,
            {
            **updated_state,
            "result": result,
            },
        )

    async def _node_evaluate_external(self, state: AgentGraphState) -> AgentGraphState:
        result = state.get("result")
        if not result:
            return {
                **state,
                "enough_evidence": False,
                "stop_reason": "no_external_results",
            }

        await self._emit(
            {
                "event": "progress",
                "type": "ranking",
                "content": (
                    f"Filtering {len(result.papers)} retrieved papers by content relevance, quality, authors,..."
                ),
                "phase": PipelinePhase.RANKING,
                "progress_percent": 50,
                "metadata": {
                    "total_papers": len(result.papers),
                    "chunks": len(result.chunks),
                },
            }
        )

        enough_evidence = self.tools.evaluate_sufficiency(result=result, stage="external")
        return {
            **state,
            "enough_evidence": enough_evidence,
            "stop_reason": "sufficient" if enough_evidence else "insufficient_external_evidence",
        }
        
    def _route_gate(self, state: AgentGraphState):
        if state.get("enough_evidence"):
            return "answer"
        if (state.get("local_retries", 0) < MAX_LOCAL_RETRIES ):
            return "local"
        if (state.get("external_retries", 0) < MAX_EXTERNAL_RETRIES):
            return "external"
        return "answer"

    async def _node_merge_results_rrf(self, state: AgentGraphState) -> AgentGraphState:
        """NEW: Merge all accumulated results using RRF (Reciprocal Rank Fusion) on chunk scores."""
        results_array = state.get("results_array", [])
 
        if not results_array:
            merged_result = RAGResult(papers=[], chunks=[])
        else:
            # Apply RRF to merge all results based on chunk scores
            merged_result = self.tools.merge_results_with_rrf(results_array)
        
        return {
            **state,
            "result": merged_result,  # Final merged result for answer generation
        }

    async def _node_answer(self, state: AgentGraphState) -> AgentGraphState:
        rag_result = state.get("result")
        if not rag_result:
            fallback = "I could not retrieve enough evidence to answer this question confidently."
            for evt in stream_like_llm(fallback):
                await self._emit(
                    {
                        "event": "chunk",
                        "data": {
                            "type": "chunk",
                            "content": evt,
                        },
                    }
                )
            return {
                **state,
                "response_text": fallback,
            }

        await self._emit(
            {
                "event": "progress",
                "type": "reasoning",
                "content": "Reading retrieved contents and generating answer...",
                "phase": PipelinePhase.LLM_GENERATION,
                "progress_percent": 90,
                "metadata": {
                    "retrieved_papers": len(rag_result.papers),
                },
            }
        )

        answer_context = await self.tools.build_answer_context(rag_result)
        paper_snapshots = answer_context["paper_snapshots"]
        if paper_snapshots:
            paper_snapshots = [
                snapshot.model_dump(mode="json", by_alias=True)
                if hasattr(snapshot, "model_dump")
                else PaperMetadata.model_validate(snapshot).model_dump(mode="json", by_alias=True)
                for snapshot in paper_snapshots
            ]
        await self._emit(
            {
                "event": "metadata",
                "data": {
                    "type": "papers_metadata",
                    "content": paper_snapshots,
                },
            }
        )

        chat_history = state.get("conversation_history", [])
        llm_input = ContextBuilder.build_enhanced_query(
            query=state["query"],
            context_string=answer_context["context"],
        )

        response_chunks: List[str] = []
        async for chunk in self.container.llm_service.stream_citation_based_response(
            history_messages=chat_history,
            context=llm_input,
        ):
            text = get_stream_response_content(chunk)
            if text:
                response_chunks.append(text)
                await self._emit(
                    {
                        "event": "chunk",
                        "data": {
                            "type": "chunk",
                            "content": text,
                        },
                    }
                )

        return {
            **state,
            "response_text": "".join(response_chunks),
        }

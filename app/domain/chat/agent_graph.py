"""LangGraph orchestration for Agent-mode chat workflow."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypedDict, cast
from openai import max_retries
from typing_extensions import NotRequired

from langgraph.graph import END, START, StateGraph

from app.core.container import ServiceContainer
from app.core.dtos.paper import PaperDTO, PaperEnrichedDTO
from app.domain.chunks.schemas import ChunkRetrieved
from app.domain.papers import LoadOptions
from app.domain.chat.response_builder import ChatResponseBuilder
from app.domain.chat.tools import AgentRAGTools
from app.extensions.logger import create_logger
from app.llm.prompts import PromptBuilder, PromptPresets
from app.llm.schemas.chat import QueryIntent, GeneratedQueryPlanResponse
from app.models.papers import DBPaper
from app.models.pipeline_tasks import PipelinePhase
from app.rag_pipeline import agent_pipeline
from app.rag_pipeline.schemas import RAGEventType, RAGResult
from app.extensions.stream import get_simple_response_content, stream_like_llm
from app.extensions import get_stream_response_content, get_stream_response_reasoning

logger = create_logger(__name__)
_RESPONSES_DIR = Path(__file__).resolve().parent / "responses"

GraphEventCallback = Callable[[Dict[str, Any]], Awaitable[None]]
DB_TOP_PAPERS_INITIAL = 30
DB_TOP_CHUNKS_INITIAL = 20
DB_TOP_PAPERS_FINAL = 40
DB_TOP_CHUNKS_FINAL = 40
MAX_RETRIES = 2


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


class AgentGraphState(TypedDict):
    query: str
    filters: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    skip_retrieval: NotRequired[bool]
    direct_response: NotRequired[str]
    breakdown: NotRequired[GeneratedQueryPlanResponse]
    intent: NotRequired[QueryIntent]
    search_queries: NotRequired[List[str]]
    external_search_queries: NotRequired[List[str]]
    merged_filters: NotRequired[Dict[str, Any]]
    initial_rag_result: NotRequired[Optional[RAGResult]]
    is_multi_hop: NotRequired[bool]
    agent_tasks: NotRequired[List[Dict[str, Any]]]
    agent_outputs: NotRequired[List[Dict[str, Any]]]
    summaries: NotRequired[List[str]]
    final_rag_result: NotRequired[Optional[RAGResult]]
    tool_plan: NotRequired[List[str]]
    messages: NotRequired[List[Dict[str, str]]]
    num_retries: int
    response_text: NotRequired[str]
    response_context: NotRequired[str]
    response_context_chunks: NotRequired[List[Dict[str, Any]]]
    retrieved_paper_ids: NotRequired[List[str]]
    paper_snapshots: NotRequired[List[Dict[str, Any]]]
    papers: NotRequired[List[Any]]
    chunks: NotRequired[List[Any]]
    gap_queries: NotRequired[List[str]]
    needs_gap_retrieval: NotRequired[bool]
    confidence_reason: NotRequired[str]


class SubAgentGraphState(TypedDict):
    query: str
    task: Dict[str, Any]
    intent: QueryIntent
    filters: Dict[str, Any]
    search_queries: List[str]
    external_search_queries: List[str]
    initial_result: NotRequired[Optional[RAGResult]]
    external_candidates: NotRequired[List[PaperEnrichedDTO]]
    external_iterations: NotRequired[int]
    external_stop_reason: NotRequired[str]
    ingested_paper_ids: NotRequired[List[str]]
    ingest_embedding_count: NotRequired[int]
    selected_fulltext_paper_ids: NotRequired[List[str]]
    processed_fulltext_count: NotRequired[int]
    rag_result: NotRequired[Optional[RAGResult]]
    summary: NotRequired[str]
    citations: NotRequired[List[str]]
    papers: NotRequired[int]
    chunks: NotRequired[int]


class AgentGraphOrchestrator:
    """LangGraph orchestrator for dual retrieval + ingest-then-cite flow."""

    def __init__(
        self,
        container: ServiceContainer,
        on_event: Optional[GraphEventCallback] = None,
    ) -> None:
        self.container = container
        self._on_event = on_event
        self._graph = self._build_graph()
        # self._subagent_graph = self._build_subagent_graph()

    def get_graph(self):
        return self._graph

    def _build_graph(self):
        builder = StateGraph(AgentGraphState)
        builder.add_node("decompose", self._node_decompose)
        builder.add_node("preresponse", self._node_preresponse)
        builder.add_node("search_database", self._node_search_database)
        builder.add_node("judge_results", self._node_judge_results)
        builder.add_node("targeted_retrieval", self._node_targeted_retrieval)
        builder.add_node("summarize_agent", self._node_summarize_agent)
        builder.add_node("general_agent", self._node_general_agent)

        builder.add_edge(START, "decompose")
        builder.add_conditional_edges(
            "decompose",
            self._route_after_decompose,
            {
                "need_predefined_response": "preresponse",
                "is_general_question": "general_agent",
                "is_research_question": "search_database",
            },
        )
        builder.add_edge("preresponse", END)
        builder.add_edge("general_agent", END)
        builder.add_edge("search_database", "judge_results")
        builder.add_conditional_edges(
            "judge_results",
            self._route_after_judge,
            {
                "enough_evidence": "summarize_agent",
                "not_enough_evidence": "targeted_retrieval",
            },
        )
        builder.add_edge("targeted_retrieval", "judge_results")
        builder.add_edge("summarize_agent", END)
        return builder.compile()

    # def _build_subagent_graph(self):
    #     builder = StateGraph(SubAgentGraphState)
    #     builder.add_node("initial_db", self._sub_node_initial_db)
    #     builder.add_node("external_search", self._sub_node_external_search)
    #     builder.add_node("ingest_external", self._sub_node_ingest_external)
    #     builder.add_node("final_db", self._sub_node_final_db)

    #     builder.add_edge(START, "initial_db")
    #     builder.add_edge("initial_db", "external_search")
    #     builder.add_edge("external_search", "ingest_external")
    #     builder.add_edge("ingest_external", "final_db")
    #     builder.add_edge("final_db", END)

    #     return builder.compile()

    async def run(self, state: AgentGraphState) -> AgentGraphState:
        """Run graph and return final state."""
        logger.info(
            f"--- Starting Agent Graph Execution for Query: '{state.get('query')}' ---"
        )
        final_state = state
        async for output in self._graph.astream(state, stream_mode="values"):
            logger.info("--- Graph State Updated ---")
            for k, v in output.items():
                if isinstance(v, list) and len(v) > 0:
                    logger.debug(f"State key '{k}' has {len(v)} items")
                elif isinstance(v, (str, int, float, bool)):
                    logger.debug(f"State key '{k}' = {str(v)[:200]}")
            final_state = output

        logger.info("--- Agent Graph Execution Completed ---")
        return cast(AgentGraphState, final_state)

    async def _emit(self, payload: Dict[str, Any]) -> None:
        if self._on_event:
            await self._on_event(payload)

    async def _emit_step(
        self,
        step_type: str,
        content: str,
        *,
        phase: str,
        progress_percent: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self._emit(
            {
                "event": "step",
                "step": {
                    "type": step_type,
                    "content": content,
                    "phase": phase,
                    "progress_percent": progress_percent,
                    "metadata": metadata or {},
                },
            }
        )

    async def _node_preresponse(self, state: AgentGraphState) -> AgentGraphState:
        intent = state.get("intent")
        if intent == QueryIntent.GIBBERISH:
            direct_response = _build_gibberish_response()
        else:
            direct_response = self._build_system_explanation_response()

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

    @staticmethod
    def _route_after_decompose(state: AgentGraphState) -> str:
        if state.get("intent") in {QueryIntent.SYSTEM, QueryIntent.GIBBERISH}:
            return "need_predefined_response"
        if state.get("intent") == QueryIntent.GENERAL:
            return "is_general_question"
        return "is_research_question"

    @staticmethod
    def _route_after_judge(state: AgentGraphState) -> str:
        return (
            "not_enough_evidence"
            if state.get("needs_gap_retrieval")
            else "enough_evidence"
        )

    async def _node_decompose(self, state: AgentGraphState) -> AgentGraphState:
        query = state["query"]
        history = state.get("conversation_history", [])
        incoming_filters = state.get("filters", {})

        await self._emit_step(
            "Thinking",
            "Understanding your question and thinking what to do...",
            phase=PipelinePhase.INIT,
            progress_percent=15,
        )

        breakdown = await self.container.llm_service.decompose_user_query_v3(
            user_question=query,
            conversation_history=history,
        )

        raw_intent = breakdown.intent
        intent_value = (
            raw_intent.value
            if isinstance(raw_intent, QueryIntent)
            else str(raw_intent or "").strip().lower()
        )

        if intent_value in {"system", "gibberish"}:
            return {
                **state,
                "intent": (
                    raw_intent
                    if isinstance(raw_intent, QueryIntent)
                    else QueryIntent.SYSTEM
                ),
            }
        if intent_value == QueryIntent.GENERAL.value:
            return {**state, "intent": QueryIntent.GENERAL}

        merged_filters = dict(incoming_filters)
        if breakdown.filters:
            merged_filters.update(breakdown.filters)

        search_queries = breakdown.hybrid_queries or [
            breakdown.clarified_question or query
        ]
        intent = breakdown.intent or QueryIntent.COMPREHENSIVE_SEARCH
        external_search_queries = list(search_queries)
        initial_top_papers = DB_TOP_PAPERS_INITIAL
        initial_top_chunks = DB_TOP_CHUNKS_INITIAL
        final_top_papers = DB_TOP_PAPERS_FINAL
        final_top_chunks = DB_TOP_CHUNKS_FINAL

        tools = AgentRAGTools(self.container)
        tool_plan = tools.select_tools(
            query=breakdown.clarified_question or query,
            filters=merged_filters,
            intent=intent,
        )

        agent_tasks = self._build_sub_agent_tasks(
            query=breakdown.clarified_question or query,
            search_queries=search_queries,
            external_search_queries=external_search_queries,
            intent=intent,
        )

        await self._emit_step(
            "planning",
            "Understanding your question and planning the search strategy...",
            phase=PipelinePhase.INIT,
            progress_percent=12,
            metadata={
                "intent": intent.value,
                "search_queries": search_queries,
                "external_search_queries": external_search_queries,
                "tool_plan": tool_plan,
                "multi_hop": len(search_queries) > 1,
                "agent_count": len(agent_tasks),
                "db_budget": {
                    "initial_top_papers": initial_top_papers,
                    "initial_top_chunks": initial_top_chunks,
                    "final_top_papers": final_top_papers,
                    "final_top_chunks": final_top_chunks,
                },
            },
        )

        return {
            **state,
            "breakdown": breakdown,
            "intent": intent,
            "search_queries": search_queries,
            "external_search_queries": external_search_queries,
            "merged_filters": merged_filters,
            "tool_plan": tool_plan,
            "is_multi_hop": False,
            "agent_tasks": agent_tasks,
            "needs_gap_retrieval": False,
        }

    # async def _node_search_database(self, state: AgentGraphState) -> AgentGraphState:
    #     agent_tasks = state.get("agent_tasks") or self._build_sub_agent_tasks(
    #         query=state.get("query", ""),
    #         search_queries=state.get("search_queries", [state["query"]]),
    #         external_search_queries=state.get("external_search_queries", [state["query"]]),
    #         intent=state.get("intent") or QueryIntent.COMPREHENSIVE_SEARCH,
    #     )
    #     if not agent_tasks:
    #         return {
    #             **state,
    #             "agent_outputs": [],
    #             "summaries": [],
    #             "papers": 0,
    #             "chunks": 0,
    #             "is_multi_hop": False,
    #         }

    #     intent = state.get("intent") or QueryIntent.COMPREHENSIVE_SEARCH
    #     merged_filters = state.get("merged_filters", {})

    #     collected_results: List[Dict[str, Any]] = []
    #     rag_results: List[RAGResult] = []
    #     total_papers = 0
    #     total_chunks = 0

    #     for idx, agent_task in enumerate(agent_tasks, start=1):
    #         await self._emit_step(
    #             "searching",
    #             f"{agent_task.get('objective', 'Focused evidence search')}.",
    #             phase=PipelinePhase.SEARCH,
    #             progress_percent=min(50, 24 + (idx * 6)),
    #             metadata={
    #                 "step_number": 2,
    #                 "task_index": idx,
    #                 "task_count": len(agent_tasks),
    #                 "queries": agent_task.get("search_queries", []),
    #                 "external_search_queries": agent_task.get("external_search_queries", []),
    #             },
    #         )

    #         result = await self._run_sub_agent(
    #             query=state["query"],
    #             task=agent_task,
    #             intent=intent,
    #             filters=merged_filters,
    #         )
    #         collected_results.append(result)
    #         total_papers += int(result.get("papers") or 0)
    #         total_chunks += int(result.get("chunks") or 0)

    #         rag_result = result.get("rag_result")
    #         if isinstance(rag_result, RAGResult):
    #             rag_results.append(rag_result)

    #     final_rag_result = self._merge_rag_results(rag_results)
    #     initial_rag_result = rag_results[0] if rag_results else None

    #     literature_review_brief = await self._build_literature_review_brief(
    #         query=state["query"],
    #         agent_outputs=collected_results,
    #     )
    #     summary_texts = [str(item.get("summary") or "").strip() for item in collected_results]

    #     return {
    #         **state,
    #         "initial_rag_result": initial_rag_result,
    #         "final_rag_result": final_rag_result,
    #         "agent_outputs": [
    #             {
    #                 k: v
    #                 for k, v in result.items()
    #                 if k != "rag_result"
    #             }
    #             for result in collected_results
    #         ],
    #         "summaries": [
    #             value
    #             for value in [*summary_texts, literature_review_brief]
    #             if value
    #         ],
    #         "papers": total_papers,
    #         "chunks": total_chunks,
    #         "is_multi_hop": len(agent_tasks) > 1,
    #     }

    async def _node_search_database(self, state: AgentGraphState) -> AgentGraphState:
        queries = state.get("search_queries", [state["query"]])
        breakdown = state.get("breakdown")
        clarified_question = (
            breakdown.clarified_question
            if breakdown and breakdown.clarified_question
            else state["query"]
        )

        merged_result = await self.container.agent_pipeline.run_explicit_workflow(
            original_query=state["query"],
            search_queries=queries,
            rerank_query=clarified_question,
            intent=state.get("intent") or QueryIntent.COMPREHENSIVE_SEARCH,
            filters=state.get("merged_filters", {}),
            top_papers=DB_TOP_PAPERS_INITIAL,
            top_chunks=DB_TOP_CHUNKS_INITIAL,
        )

        return {
            **state,
            "initial_rag_result": merged_result,
            "final_rag_result": merged_result,
            "papers": list(merged_result.papers) if merged_result else [],
            "chunks": list(merged_result.chunks) if merged_result else [],
        }

    async def _node_summarize_agent(self, state: AgentGraphState) -> AgentGraphState:
        rag_result = state.get("final_rag_result")
        if not rag_result:
            return state

        response_builder = ChatResponseBuilder()
        paper_snapshots = response_builder.extract_metadata_from_results(rag_result)
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
        context_str, _ = response_builder.build_context_from_results(rag_result)
        retrieved_paper_ids = response_builder.get_retrieved_paper_ids(rag_result)

        llm_input = response_builder.build_enhanced_query(
            query=state["query"],
            conversation_history=chat_history,
            context_string=context_str,
        )

        # summaries = state.get("summaries") or []
        # summary_blocks = [
        #     str(item).strip()
        #     for item in summaries
        #     if str(item).strip()
        # ]
        # if summary_blocks:
        #     llm_input = (
        #         f"<multi_agent_literature_review_brief>\n"
        #         f"{'\n\n'.join(summary_blocks)}\n"
        #         f"</multi_agent_literature_review_brief>\n\n"
        #         f"{llm_input}"
        #     )

        assistant_response_chunks: List[str] = []
        reasoning_chunks: List[str] = []

        async for chunk in self.container.llm_service.stream_citation_based_response(
            context=llm_input,
            prompt_name="generate_answer",
        ):
            text = get_stream_response_content(chunk)
            reasoning = get_stream_response_reasoning(chunk)

            if reasoning and reasoning not in reasoning_chunks:
                reasoning_chunks.append(reasoning)
                await self._emit(
                    {
                        "event": "reasoning",
                        "data": {
                            "type": "reasoning",
                            "content": reasoning,
                        },
                    }
                )

            if text:
                assistant_response_chunks.append(text)
                await self._emit(
                    {
                        "event": "chunk",
                        "data": {
                            "type": "chunk",
                            "content": text,
                        },
                    }
                )

        return cast(
            AgentGraphState,
            {
                **state,
                "response_text": "".join(assistant_response_chunks),
                "response_context": context_str,
                "response_context_chunks": response_builder.extract_context_chunks_from_results(
                    rag_result
                ),
                "retrieved_paper_ids": retrieved_paper_ids,
                "paper_snapshots": paper_snapshots,
            },
        )

    async def _run_sub_agent(
        self,
        *,
        query: str,
        task: Dict[str, Any],
        intent: QueryIntent,
        filters: Dict[str, Any],
    ) -> Dict[str, Any]:
        search_queries = self._normalize_queries(task.get("search_queries", [])) or [
            query
        ]
        external_queries = self._normalize_queries(
            task.get("external_search_queries", [])
        ) or [query]

        sub_state: SubAgentGraphState = {
            "query": query,
            "task": task,
            "intent": intent,
            "filters": filters,
            "search_queries": search_queries,
            "external_search_queries": external_queries,
        }

        final_sub_state = sub_state
        # async for output in self._subagent_graph.astream(sub_state, stream_mode="values"):
        #     final_sub_state = cast(SubAgentGraphState, output)

        final_result = final_sub_state.get("rag_result")
        return {
            "agent_id": task.get("agent_id"),
            "objective": task.get("objective"),
            "search_queries": search_queries,
            "external_search_queries": external_queries,
            "summary": str(final_sub_state.get("summary") or ""),
            "citations": list(final_sub_state.get("citations") or []),
            "papers": int(final_sub_state.get("papers") or 0),
            "chunks": int(final_sub_state.get("chunks") or 0),
            "external_candidates": len(
                final_sub_state.get("external_candidates") or []
            ),
            "external_iterations": int(final_sub_state.get("external_iterations") or 0),
            "external_stop_reason": str(
                final_sub_state.get("external_stop_reason") or "db_sufficient"
            ),
            "ingested_paper_ids": list(final_sub_state.get("ingested_paper_ids") or []),
            "ingest_embedding_count": int(
                final_sub_state.get("ingest_embedding_count") or 0
            ),
            "selected_fulltext_paper_ids": list(
                final_sub_state.get("selected_fulltext_paper_ids") or []
            ),
            "processed_fulltext_count": int(
                final_sub_state.get("processed_fulltext_count") or 0
            ),
            "rag_result": final_result,
        }

    # async def _sub_node_initial_db(self, state: SubAgentGraphState) -> SubAgentGraphState:
    #     initial_result = await self._run_db_workflow(
    #         query=state["query"],
    #         search_queries=state["search_queries"],
    #         intent=state["intent"],
    #         filters=state.get("filters", {}),
    #         top_papers=DB_TOP_PAPERS_INITIAL,
    #         top_chunks=DB_TOP_CHUNKS_INITIAL,
    #         enable_paper_ranking=False,
    #     )
    #     return {
    #         **state,
    #         "initial_result": initial_result,
    #     }

    async def _sub_node_external_search(
        self, state: SubAgentGraphState
    ) -> SubAgentGraphState:
        task = state.get("task", {})
        initial_result = state.get("initial_result")
        is_sufficient = await self._evaluate_agent_results(
            str(task.get("objective") or state["query"]),
            initial_result,
            "internal",
        )

        if is_sufficient:
            return {
                **state,
                "external_candidates": [],
                "external_iterations": 0,
                "external_stop_reason": "db_sufficient",
            }

        (
            external_candidates,
            external_iterations,
            external_stop_reason,
            _,
        ) = await self._run_external_search_cycles(
            query=state["query"],
            intent=state["intent"],
            initial_result=initial_result,
            filters=state.get("filters", {}),
            search_queries=state.get("external_search_queries", [state["query"]]),
            subtopics=state.get("search_queries", [state["query"]]),
            emit_cycle_events=True,
        )

        return {
            **state,
            "external_candidates": external_candidates,
            "external_iterations": external_iterations,
            "external_stop_reason": external_stop_reason,
        }

    async def _sub_node_ingest_external(
        self, state: SubAgentGraphState
    ) -> SubAgentGraphState:
        external_candidates = state.get("external_candidates") or []
        initial_result = state.get("initial_result")
        intent = state["intent"]
        task = state.get("task", {})

        budget = self._compute_ingest_budget(
            intent=intent,
            initial_result=initial_result,
            external_available=len(external_candidates),
        )

        ingested_ids: List[str] = []
        embedded_count = 0
        selected_fulltext_ids: List[str] = []
        processed_count = 0

        if budget > 0:
            selected = external_candidates[:budget]
            selected_for_ingest: List[PaperDTO | PaperEnrichedDTO] = list(selected)
            created = (
                await self.container.paper_service.batch_create_papers_from_schema(
                    selected_for_ingest,
                    enrich=True,
                )
            )
            ingested_ids = [paper.paper_id for paper in created]

            if ingested_ids:
                (
                    embedded_count,
                    processed_count,
                    selected_fulltext_ids,
                ) = await self._run_selective_post_ingest_processing(
                    query=str(task.get("objective") or state["query"]),
                    intent=intent,
                    selected_candidates=selected,
                    ingested_ids=ingested_ids,
                )

        return {
            **state,
            "ingested_paper_ids": ingested_ids,
            "ingest_embedding_count": embedded_count,
            "selected_fulltext_paper_ids": selected_fulltext_ids,
            "processed_fulltext_count": processed_count,
        }

    # async def _sub_node_final_db(self, state: SubAgentGraphState) -> SubAgentGraphState:
    #     final_result = await self._run_db_workflow(
    #         query=state["query"],
    #         search_queries=state["search_queries"],
    #         intent=state["intent"],
    #         filters=state.get("filters", {}),
    #         top_papers=DB_TOP_PAPERS_FINAL,
    #         top_chunks=DB_TOP_CHUNKS_FINAL,
    #         enable_paper_ranking=False,
    #     )
    #     if final_result is None:
    #         final_result = state.get("initial_result")

    #     summary = await self._build_sub_agent_summary(task=state.get("task", {}), rag_result=final_result)
    #     citations = self._extract_sub_agent_citations(final_result)

    #     return {
    #         **state,
    #         "rag_result": final_result,
    #         "summary": summary,
    #         "citations": citations,
    #         "papers": len(final_result.papers) if final_result else 0,
    #         "chunks": len(final_result.chunks) if final_result else 0,
    #     }

    async def _node_judge_results(self, state: AgentGraphState) -> AgentGraphState:
        result = state.get("final_rag_result")
        if not result:
            return {
                **state,
                "needs_gap_retrieval": True,
                "confidence_reason": "no_initial_results",
            }

        if state["num_retries"] < MAX_RETRIES:
            return {
                **state,
                "needs_gap_retrieval": True,
            }
        is_sufficient = await self._evaluate_agent_results(
            objective=state["query"],
            result=result,
            mode="initial_db",
        )

        return {
            **state,
            "needs_gap_retrieval": False,
            "confidence_reason": "sufficient" if is_sufficient else "missing_info",
        }

    async def _is_sufficient_results(
        self,
        query: str,
        rag_result: Optional[RAGResult],
        result_type: str,
    ) -> bool:
        return await self._evaluate_agent_results(
            objective=query,
            result=rag_result,
            mode=result_type,
        )

    async def _node_targeted_retrieval(self, state: AgentGraphState) -> AgentGraphState:
        if not state.get("needs_gap_retrieval"):
            return state

        current_result = state.get("final_rag_result")
        breakdown = state.get("breakdown")
        clarified_question = (
            breakdown.clarified_question
            if breakdown and breakdown.clarified_question
            else state["query"]
        )

        await self._emit_step(
            "gap_analysis",
            "Initial evidence looks incomplete. Generating focused gap queries...",
            phase=PipelinePhase.SEARCH,
            progress_percent=62,
            metadata={"reason": state.get("confidence_reason", "missing_info")},
        )

        gap_queries = await self._generate_gap_queries(
            clarified_question=clarified_question,
            rag_result=current_result,
            existing_queries=state.get("search_queries", []),
        )

        if not gap_queries:
            return {**state, "gap_queries": [], "needs_gap_retrieval": False}

        await self._emit_step(
            "targeted_retrieval",
            "Running targeted retrieval for missing evidence...",
            phase=PipelinePhase.SEARCH,
            progress_percent=72,
            metadata={"gap_queries": gap_queries},
        )

        targeted_result = await self.container.agent_pipeline.run_explicit_workflow(
            original_query=state["query"],
            search_queries=gap_queries,
            rerank_query=clarified_question,
            intent=state.get("intent") or QueryIntent.COMPREHENSIVE_SEARCH,
            filters=state.get("merged_filters", {}),
            top_papers=DB_TOP_PAPERS_FINAL,
            top_chunks=DB_TOP_CHUNKS_FINAL,
        )

        merged_candidates = [
            result
            for result in [current_result, targeted_result]
            if isinstance(result, RAGResult)
        ]
        merged_result = self._merge_rag_results(merged_candidates)
        final_result = self._final_rerank_result(
            query=clarified_question,
            rag_result=merged_result,
        )
        selected_result = (
            final_result if isinstance(final_result, RAGResult) else merged_result
        )

        return {
            **state,
            "gap_queries": gap_queries,
            "needs_gap_retrieval": False,
            "final_rag_result": selected_result,
            "num_retries": state["num_retries"] + 1,
            "papers": list(selected_result.papers) if selected_result else [],
            "chunks": list(selected_result.chunks) if selected_result else [],
        }

    async def _generate_gap_queries(
        self,
        *,
        clarified_question: str,
        rag_result: Optional[RAGResult],
        existing_queries: List[str],
    ) -> List[str]:
        if not clarified_question.strip():
            return []

        evidence_lines: List[str] = []
        if rag_result and rag_result.chunks:
            for idx, chunk in enumerate(rag_result.chunks[:6], start=1):
                chunk_text = self._truncate_text(getattr(chunk, "text", ""), 240)
                if chunk_text:
                    evidence_lines.append(f"{idx}. {chunk_text}")

        system_prompt = (
            "You are an academic retrieval gap analyzer. "
            "Given the clarified question and current evidence snippets, output concise missing-information queries. "
            'Return ONLY JSON: {"gap_queries": [string], "reason": string}.'
        )
        payload = {
            "clarified_question": clarified_question,
            "existing_queries": existing_queries,
            "evidence": evidence_lines,
            "constraints": {
                "max_queries": 3,
                "avoid_repeating_existing_queries": True,
                "query_style": "short academic search phrases",
            },
        }

        data = self.container.llm_service.prompt_json(
            system_prompt=system_prompt,
            user_payload=payload,
        )
        raw_gap_queries = data.get("gap_queries", [])
        if not isinstance(raw_gap_queries, list):
            return []

        normalized = self._normalize_queries(
            [str(item).strip() for item in raw_gap_queries if str(item).strip()]
        )
        existing_lower = {query.lower() for query in existing_queries}
        return [query for query in normalized if query.lower() not in existing_lower][
            :3
        ]

    def _final_rerank_result(
        self, *, query: str, rag_result: Optional[RAGResult]
    ) -> Optional[RAGResult]:
        if not rag_result:
            return None

        chunks = list(rag_result.chunks or [])
        papers = list(rag_result.papers or [])

        try:
            if chunks:
                chunks = self.container.ranking_service.rerank_chunks(query, chunks)
        except Exception as exc:
            logger.warning("Final chunk rerank failed; keeping merged order: %s", exc)

        try:
            paper_models = [
                paper
                for paper in [getattr(item, "paper", None) for item in papers]
                if isinstance(paper, DBPaper)
            ]
            if paper_models:
                papers = self.container.ranking_service.rank_papers_v2(
                    papers=paper_models,
                    chunks=chunks,
                )
        except Exception as exc:
            logger.warning("Final paper rerank failed; keeping merged order: %s", exc)

        return RAGResult(
            papers=papers[:DB_TOP_PAPERS_FINAL],
            chunks=chunks[:DB_TOP_CHUNKS_FINAL],
        )

    async def _run_external_search_cycles(
        self,
        *,
        query: str,
        intent: QueryIntent,
        initial_result: Optional[RAGResult],
        filters: Dict[str, Any],
        search_queries: List[str],
        subtopics: List[str],
        emit_cycle_events: bool,
    ) -> tuple[List[PaperEnrichedDTO], int, str, List[Dict[str, Any]]]:
        hard_stop_cycles = 4
        max_cycles = 5
        probe_limit = 12
        default_limit = 50

        expanded_queries = self._normalize_queries(search_queries) or [query]

        if emit_cycle_events:
            await self._emit_step(
                "searching_external",
                "Searching external provider with adaptive query reformulation...",
                phase=PipelinePhase.SEARCH,
                progress_percent=40,
                metadata={
                    "queries": expanded_queries,
                    "provider": "semantic_scholar",
                    "policy": {
                        "max_cycles": max_cycles,
                        "hard_stop_cycles": hard_stop_cycles,
                        "decision": "react_agent_controller",
                    },
                },
            )

        async def _search(
            search_query: str, *, semantic_limit: int
        ) -> List[PaperEnrichedDTO]:
            papers, _ = await self.container.retrieval_service.hybrid_search(
                query=search_query,
                semantic_limit=semantic_limit,
                filters=filters,
            )
            return papers

        external_candidates: List[PaperEnrichedDTO] = []
        seen: set[str] = set()
        stop_reason = "max_cycles_reached"
        executed_cycles = 0
        cycle_summaries: List[Dict[str, Any]] = []

        adaptive_queries = expanded_queries
        query_history: List[str] = []
        latest_cycle_candidates: List[PaperEnrichedDTO] = []

        for cycle_idx in range(max_cycles):
            cycle_num = cycle_idx + 1
            action = await self._react_external_action(
                query=query,
                intent=intent,
                cycle_num=cycle_num,
                max_cycles=max_cycles,
                hard_stop_cycles=hard_stop_cycles,
                initial_result=initial_result,
                total_external_candidates=len(external_candidates),
                query_history=query_history,
                cycle_summaries=cycle_summaries,
                latest_cycle_candidates=latest_cycle_candidates,
                candidate_queries=adaptive_queries,
                default_probe_limit=probe_limit,
                default_full_limit=default_limit,
            )

            if not action.get("continue_search", False):
                stop_reason = str(action.get("reason", "react_agent_stop"))
                break

            executed_cycles += 1
            active_query = str(action.get("next_query") or "").strip()
            semantic_limit = int(action.get("semantic_limit") or default_limit)
            query_form = str(action.get("query_form") or "keyword")

            if not active_query:
                if cycle_idx < len(adaptive_queries):
                    active_query = adaptive_queries[cycle_idx]
                else:
                    active_query = adaptive_queries[-1]

            query_history.append(active_query)

            if emit_cycle_events:
                await self._emit_step(
                    "searching_external",
                    (
                        f"External search cycle {cycle_num}/{max_cycles} "
                        f"using {query_form} query ({semantic_limit} papers)."
                    ),
                    phase=PipelinePhase.SEARCH,
                    progress_percent=min(48, 40 + (cycle_num * 2)),
                    metadata={
                        "cycle": cycle_num,
                        "max_cycles": max_cycles,
                        "query": active_query,
                        "semantic_limit": semantic_limit,
                        "mode": "react_agent",
                        "strategy": query_form,
                        "controller_reason": action.get("reason"),
                    },
                )

            search_results = await asyncio.gather(
                _search(active_query, semantic_limit=semantic_limit),
                return_exceptions=True,
            )

            cycle_new_candidates = 0
            cycle_candidates = []

            for result in search_results:
                if isinstance(result, Exception):
                    logger.warning("External search branch failed: %s", result)
                    continue

                if not isinstance(result, list):
                    continue

                for paper in result:
                    if paper.paper_id in seen:
                        continue
                    seen.add(paper.paper_id)
                    external_candidates.append(paper)
                    cycle_candidates.append(paper)
                    cycle_new_candidates += 1

            next_query_raw = str(action.get("next_query") or "").strip()
            next_query = next_query_raw if next_query_raw else None
            if next_query and next_query.lower() not in {
                q.lower() for q in adaptive_queries
            }:
                adaptive_queries.append(next_query)

            cycle_summaries.append(
                {
                    "cycle": cycle_num,
                    "tool": "semantic_scholar_search",
                    "query": active_query,
                    "new_candidates": cycle_new_candidates,
                    "total_candidates": len(external_candidates),
                    "decision": "continue",
                    "reason": str(action.get("reason", "react_continue")),
                    "next_query": next_query,
                    "query_refiner_used": True,
                    "query_form": query_form,
                    "semantic_limit": semantic_limit,
                }
            )
            latest_cycle_candidates = cycle_candidates

            if len(subtopics) > 1 and cycle_candidates:
                optimized = await self._parallel_evaluate_and_optimize_external_search(
                    user_query=query,
                    subtopics=subtopics,
                    cycle_candidates=cycle_candidates,
                    query_history=query_history,
                    cycle_num=cycle_num,
                    max_cycles=max_cycles,
                )
                refined_queries = self._normalize_queries(
                    optimized.get("refined_queries", [])
                )
                for refined in refined_queries:
                    if refined.lower() not in {q.lower() for q in adaptive_queries}:
                        adaptive_queries.append(refined)

                if not bool(optimized.get("continue_search", True)):
                    stop_reason = str(
                        optimized.get("reason", "parallel_evaluator_stop")
                    )
                    break

        if executed_cycles >= max_cycles and stop_reason == "max_cycles_reached":
            stop_reason = "react_cycle_cap_reached"

        return external_candidates, executed_cycles, stop_reason, cycle_summaries

    async def _parallel_evaluate_and_optimize_external_search(
        self,
        *,
        user_query: str,
        subtopics: List[str],
        cycle_candidates: List[PaperEnrichedDTO],
        query_history: List[str],
        cycle_num: int,
        max_cycles: int,
    ) -> Dict[str, Any]:
        evidence = self._build_external_cycle_evidence(cycle_candidates)
        normalized_subtopics = self._normalize_queries(subtopics)[:6]

        async def _evaluate_subtopic(subtopic: str) -> Dict[str, Any]:
            prompt = (
                "Evaluate if current external candidates are sufficient for this subtopic. "
                'Return ONLY JSON: {"subtopic": string, "sufficient": bool, "gap": string, "suggested_query": string}.'
            )
            payload = {
                "subtopic": subtopic,
                "user_query": user_query,
                "evidence": evidence,
                "query_history": query_history[-8:],
            }
            try:
                response = self.container.llm_service.llm_provider.simple_prompt(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(payload)},
                    ],
                    **PromptPresets.merge_with_overrides(
                        PromptPresets.DETERMINISTIC,
                        response_format={"type": "json_object"},
                    ),
                )

                return json.loads(get_simple_response_content(response).strip())
            except Exception:
                return {
                    "subtopic": subtopic,
                    "sufficient": False,
                    "gap": "evaluator_failed",
                    "suggested_query": subtopic,
                }

        evaluations = await asyncio.gather(
            *[_evaluate_subtopic(topic) for topic in normalized_subtopics]
        )

        optimizer_prompt = (
            "You are an optimizer that consolidates subtopic evaluations into external search decisions. "
            'Return ONLY JSON: {"continue_search": bool, "reason": string, "refined_queries": [string]}.'
        )
        optimizer_payload = {
            "user_query": user_query,
            "cycle": cycle_num,
            "max_cycles": max_cycles,
            "evaluations": evaluations,
            "query_history": query_history[-10:],
        }

        try:
            response = self.container.llm_service.llm_provider.simple_prompt(
                messages=[
                    {"role": "system", "content": optimizer_prompt},
                    {"role": "user", "content": json.dumps(optimizer_payload)},
                ],
                **PromptPresets.merge_with_overrides(
                    PromptPresets.DETERMINISTIC,
                    response_format={"type": "json_object"},
                ),
            )
            optimized = json.loads(get_simple_response_content(response).strip())
            if not isinstance(optimized, dict):
                raise ValueError("optimizer_result_not_dict")

            refined = optimized.get("refined_queries", [])
            if not isinstance(refined, list):
                refined = []

            return {
                "continue_search": bool(optimized.get("continue_search", True)),
                "reason": str(optimized.get("reason", "parallel_optimizer")),
                "refined_queries": [str(q).strip() for q in refined if str(q).strip()],
            }
        except Exception as exc:
            logger.warning(
                "Parallel evaluator optimizer failed, fallback to continue: %s", exc
            )
            fallback_queries = [
                str(item.get("suggested_query", "")).strip()
                for item in evaluations
                if not bool(item.get("sufficient", False))
            ]
            fallback_queries = [q for q in fallback_queries if q]
            return {
                "continue_search": True,
                "reason": "parallel_optimizer_fallback",
                "refined_queries": fallback_queries,
            }

    async def _run_selective_post_ingest_processing(
        self,
        *,
        query: str,
        intent: QueryIntent,
        selected_candidates: List[PaperEnrichedDTO],
        ingested_ids: List[str],
    ) -> tuple[int, int, List[str]]:
        """Post-ingest flow: embed metadata first, then selectively process full text."""
        embedded_count = await self._ensure_title_abstract_embeddings(ingested_ids)

        fulltext_budget = self._compute_fulltext_budget(
            intent=intent,
            ingested_count=len(ingested_ids),
        )
        selected_fulltext_ids = await self._select_fulltext_candidates(
            query=query,
            ingested_ids=ingested_ids,
            budget=fulltext_budget,
        )

        if not selected_fulltext_ids:
            return embedded_count, 0, []

        selected_map = {paper.paper_id: paper for paper in selected_candidates}
        papers_for_processing = [
            selected_map[paper_id]
            for paper_id in selected_fulltext_ids
            if paper_id in selected_map
        ]

        if not papers_for_processing:
            return embedded_count, 0, selected_fulltext_ids

        process_results = await self.container.paper_processor.process_papers_v2(
            papers=papers_for_processing,
            max_workers=2,
            pdf_parser="docling",
        )
        processed_count = sum(1 for ok in process_results.values() if ok)
        return embedded_count, processed_count, selected_fulltext_ids

    async def _ensure_title_abstract_embeddings(self, paper_ids: List[str]) -> int:
        """Generate missing title+abstract embeddings for ingested papers only."""
        papers = await self._load_db_papers_by_ids(paper_ids)
        papers_to_embed = [
            paper
            for paper in papers
            if getattr(paper, "embedding", None) is None
            and self._build_title_abstract_text(paper)
        ]
        if not papers_to_embed:
            return 0

        texts = [self._build_title_abstract_text(paper) for paper in papers_to_embed]
        embeddings = await self.container.embedding_service.create_embeddings_batch(
            texts,
            batch_size=20,
            task="search_document",
        )

        payload = {
            str(paper.paper_id): embedding
            for paper, embedding in zip(papers_to_embed, embeddings)
            if embedding
        }
        if not payload:
            return 0

        return await self.container.paper_repository.update_paper_embeddings_bulk(
            payload
        )

    async def _select_fulltext_candidates(
        self,
        *,
        query: str,
        ingested_ids: List[str],
        budget: int,
    ) -> List[str]:
        """Select a small subset of ingested papers for full-text processing."""
        if budget <= 0:
            return []

        papers = await self._load_db_papers_by_ids(ingested_ids)
        if not papers:
            return []

        ranked_ids = self._rank_papers_by_title_abstract(query=query, papers=papers)
        if not ranked_ids:
            ranked_ids = [str(paper.paper_id) for paper in papers]

        shortlist_size = min(max(budget * 2, budget), len(ranked_ids))
        shortlist = ranked_ids[:shortlist_size]
        agent_selected = await self._agent_select_fulltext_papers(
            query=query,
            papers=papers,
            shortlist=shortlist,
            budget=budget,
        )

        allowed = set(shortlist)
        ordered: List[str] = []
        for paper_id in [*agent_selected, *ranked_ids]:
            if paper_id in allowed and paper_id not in ordered:
                ordered.append(paper_id)

        return ordered[:budget]

    async def _agent_select_fulltext_papers(
        self,
        *,
        query: str,
        papers: List[DBPaper],
        shortlist: List[str],
        budget: int,
    ) -> List[str]:
        """Optional LLM selector for deciding which papers need full-text processing."""
        if not shortlist or budget <= 0:
            return []

        paper_map = {str(paper.paper_id): paper for paper in papers}
        candidate_rows: List[Dict[str, Any]] = []
        for paper_id in shortlist:
            paper = paper_map.get(paper_id)
            if not paper:
                continue
            candidate_rows.append(
                {
                    "paper_id": paper_id,
                    "title": self._truncate_text(paper.title, 240),
                    "abstract": self._truncate_text(paper.abstract, 400),
                    "year": getattr(paper, "year", None),
                    "is_open_access": bool(getattr(paper, "is_open_access", False)),
                    "has_pdf_url": bool(getattr(paper, "pdf_url", None)),
                }
            )

        if not candidate_rows:
            return []

        system_prompt = (
            "You are selecting papers for expensive full-text processing in an academic RAG pipeline. "
            "Prioritize papers that are most likely to contain decisive evidence beyond title/abstract. "
            'Return ONLY JSON: {"paper_ids": [string], "reason": string}.'
        )
        payload = {
            "query": query,
            "budget": budget,
            "candidates": candidate_rows,
            "selection_rules": {
                "max_selected": budget,
                "prefer_open_access_or_pdf": True,
                "skip_redundant_near_duplicates": True,
            },
        }

        try:
            response = self.container.llm_service.llm_provider.simple_prompt(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                **PromptPresets.merge_with_overrides(
                    PromptPresets.DETERMINISTIC,
                    response_format={"type": "json_object"},
                ),
            )

            raw_content = get_simple_response_content(response).strip()
            data = json.loads(raw_content)
            raw_ids = data.get("paper_ids", [])
            if not isinstance(raw_ids, list):
                return []

            allowed = set(shortlist)
            selected: List[str] = []
            for value in raw_ids:
                paper_id = str(value or "").strip()
                if not paper_id or paper_id not in allowed or paper_id in selected:
                    continue
                selected.append(paper_id)
                if len(selected) >= budget:
                    break
            return selected
        except Exception as exc:
            logger.warning(
                "Full-text selector agent failed, using rank fallback: %s", exc
            )
            return []

    def _rank_papers_by_title_abstract(
        self,
        *,
        query: str,
        papers: List[DBPaper],
    ) -> List[str]:
        """Rank papers by reranking virtual chunks built from title+abstract."""
        if not papers:
            return []

        virtual_chunks: List[ChunkRetrieved] = []
        for paper in papers:
            text = self._build_title_abstract_text(paper)
            if not text:
                continue
            virtual_chunks.append(
                ChunkRetrieved(
                    chunk_id=f"{paper.paper_id}_title_abstract",
                    paper_id=str(paper.paper_id),
                    text=text,
                    token_count=len(text.split()),
                    chunk_index=0,
                    section_title="Title+Abstract",
                    page_number=None,
                    label="title_abstract",
                    level=0,
                    id=int(getattr(paper, "id", 0) or 0),
                    char_start=None,
                    char_end=None,
                    docling_metadata=None,
                    embedding=None,
                    created_at=datetime.utcnow(),
                    relevance_score=0.0,
                )
            )

        if not virtual_chunks:
            return [str(paper.paper_id) for paper in papers]

        try:
            reranked = self.container.ranking_service.rerank_chunks(
                query, virtual_chunks
            )
        except Exception as exc:
            logger.warning(
                "Virtual title+abstract reranking failed, preserving ingest order: %s",
                exc,
            )
            return [str(paper.paper_id) for paper in papers]

        ranked_ids: List[str] = []
        for chunk in reranked:
            paper_id = str(getattr(chunk, "paper_id", "") or "").strip()
            if paper_id and paper_id not in ranked_ids:
                ranked_ids.append(paper_id)

        for paper in papers:
            paper_id = str(paper.paper_id)
            if paper_id not in ranked_ids:
                ranked_ids.append(paper_id)

        return ranked_ids

    async def _load_db_papers_by_ids(self, paper_ids: List[str]) -> List[DBPaper]:
        """Load DB papers for a list of IDs while preserving requested order."""
        normalized_ids = [
            str(paper_id).strip() for paper_id in paper_ids if str(paper_id).strip()
        ]
        if not normalized_ids:
            return []

        papers, _ = await self.container.paper_repository.get_papers(
            skip=0,
            limit=len(normalized_ids),
            paper_ids=normalized_ids,
            load_options=LoadOptions.none(),
        )

        paper_map = {str(paper.paper_id): paper for paper in papers}
        return [
            paper_map[paper_id] for paper_id in normalized_ids if paper_id in paper_map
        ]

    @staticmethod
    def _build_title_abstract_text(paper: DBPaper) -> str:
        """Build robust text payload for metadata-level scoring and embedding."""
        title = str(getattr(paper, "title", "") or "").strip()
        abstract = str(getattr(paper, "abstract", "") or "").strip()

        if title and abstract:
            return f"Title: {title}\n\nAbstract: {abstract}"
        return title or abstract

    @staticmethod
    def _compute_fulltext_budget(*, intent: QueryIntent, ingested_count: int) -> int:
        """Cap expensive full-text processing to a focused subset."""
        if ingested_count <= 0:
            return 0

        base_map: Dict[QueryIntent, int] = {
            QueryIntent.COMPREHENSIVE_SEARCH: 4,
            QueryIntent.AUTHOR_PAPERS: 3,
            QueryIntent.COMPARISON: 6,
            QueryIntent.FOUNDATIONAL: 4,
        }
        budget = base_map.get(intent, 4)

        if ingested_count <= 3:
            return ingested_count

        budget = min(budget, ingested_count)
        return max(2, budget)

    # async def _run_db_workflow(
    #     self,
    #     *,
    #     query: str,
    #     search_queries: List[str],
    #     intent: QueryIntent,
    #     filters: Dict[str, Any],
    #     top_papers: int,
    #     top_chunks: int,
    #     enable_paper_ranking: bool,
    # ) -> Optional[RAGResult]:
    #     rag_result: Optional[RAGResult] = None

    #     async for event in self.container.agent_pipeline.run_explicit_workflow(
    #         original_query=query,
    #         search_queries=search_queries,
    #         intent=intent,
    #         filters=filters,
    #         top_papers=top_papers,
    #         top_chunks=top_chunks,
    #         enable_reranking=True,
    #         enable_paper_ranking=enable_paper_ranking,
    #     ):
    #         if event.type in (RAGEventType.SEARCHING, RAGEventType.RANKING):
    #             await self._emit(
    #                 {
    #                     "event": "rag_event",
    #                     "type": event.type,
    #                     "data": event.data if isinstance(event.data, dict) else {},
    #                 }
    #             )

    #         if event.type == RAGEventType.RESULT and isinstance(event.data, RAGResult):
    #             rag_result = event.data

    #     return rag_result

    @staticmethod
    def _compute_ingest_budget(
        *,
        intent: QueryIntent,
        initial_result: Optional[RAGResult],
        external_available: int,
    ) -> int:
        if external_available <= 0:
            return 0

        base_budget_map: Dict[QueryIntent, int] = {
            QueryIntent.COMPREHENSIVE_SEARCH: 10,
            QueryIntent.AUTHOR_PAPERS: 6,
            QueryIntent.COMPARISON: 14,
            QueryIntent.FOUNDATIONAL: 8,
        }
        budget = base_budget_map.get(intent, 8)

        paper_count = len(initial_result.papers) if initial_result else 0
        chunk_count = len(initial_result.chunks) if initial_result else 0
        top_score = (
            float(initial_result.papers[0].relevance_score)
            if initial_result
            and initial_result.papers
            and initial_result.papers[0].relevance_score is not None
            else 0.0
        )

        if paper_count < 8:
            budget += 6
        elif paper_count < 15:
            budget += 3

        if chunk_count < 12:
            budget += 3

        if top_score < 0.45:
            budget += 4
        elif top_score < 0.60:
            budget += 2

        budget = max(4, min(25, budget))
        return min(budget, external_available)

    @staticmethod
    def _build_sub_agent_tasks(
        *,
        query: str,
        search_queries: List[str],
        external_search_queries: List[str],
        intent: QueryIntent,
    ) -> List[Dict[str, Any]]:
        normalized_search = AgentGraphOrchestrator._normalize_queries(
            search_queries
        ) or [query]
        normalized_external = AgentGraphOrchestrator._normalize_queries(
            external_search_queries
        )

        tasks: List[Dict[str, Any]] = []
        max_agents = min(4, len(normalized_search))
        for idx, objective in enumerate(normalized_search[:max_agents], start=1):
            anchored_external: List[str] = []
            if normalized_external:
                anchor = normalized_external[min(idx - 1, len(normalized_external) - 1)]
                anchored_external.append(anchor)
            for ext in normalized_external:
                if ext.lower() not in {q.lower() for q in anchored_external}:
                    anchored_external.append(ext)

            tasks.append(
                {
                    "agent_id": f"agent_{idx}",
                    "objective": objective,
                    "search_queries": [objective],
                    "external_search_queries": anchored_external or [objective],
                    "intent": intent.value,
                }
            )

        return tasks

    @staticmethod
    def _merge_rag_results(results: List[RAGResult]) -> Optional[RAGResult]:
        if not results:
            return None

        paper_by_id: Dict[str, Any] = {}
        chunk_by_id: Dict[str, Any] = {}

        for result in results:
            for ranked in result.papers:
                paper_id = str(
                    getattr(ranked, "paper_id", "")
                    or getattr(getattr(ranked, "paper", None), "paper_id", "")
                )
                if not paper_id:
                    continue
                existing = paper_by_id.get(paper_id)
                existing_score = (
                    float(getattr(existing, "relevance_score", 0.0) or 0.0)
                    if existing
                    else -1.0
                )
                current_score = float(getattr(ranked, "relevance_score", 0.0) or 0.0)
                if existing is None or current_score > existing_score:
                    paper_by_id[paper_id] = ranked

            for chunk in result.chunks:
                chunk_id = str(getattr(chunk, "chunk_id", "") or "")
                if not chunk_id:
                    continue
                if chunk_id not in chunk_by_id:
                    chunk_by_id[chunk_id] = chunk

        merged_papers = sorted(
            paper_by_id.values(),
            key=lambda item: float(getattr(item, "relevance_score", 0.0) or 0.0),
            reverse=True,
        )
        merged_chunks = list(chunk_by_id.values())

        return RAGResult(
            papers=merged_papers[:DB_TOP_PAPERS_FINAL],
            chunks=merged_chunks[:DB_TOP_CHUNKS_FINAL],
        )

    @staticmethod
    def _extract_sub_agent_citations(rag_result: Optional[RAGResult]) -> List[str]:
        if not rag_result or not rag_result.papers:
            return []
        citations: List[str] = []
        for paper in rag_result.papers[:8]:
            citations.append(str(paper.paper_id))
        return citations

    async def _build_sub_agent_summary(
        self, task: Dict[str, Any], rag_result: Optional[RAGResult]
    ) -> str:
        objective = str(task.get("objective") or "sub-task")
        if not rag_result or not rag_result.papers:
            return f"Sub-agent objective: {objective}. No relevant papers found."

        # Generate LLM Summary if chunks exist
        if rag_result.chunks:
            try:
                response_builder = ChatResponseBuilder()
                context_string, _ = response_builder.build_context_from_results(
                    rag_result
                )
                enhanced_query = response_builder.build_enhanced_query(
                    query=objective,
                    conversation_history=None,
                    context_string=context_string,
                )

                messages, _ = PromptBuilder.build(
                    prompt_name="generate_sub_agent_summary",
                    user_input=enhanced_query,
                )

                response = self.container.llm_service.llm_provider.simple_prompt(
                    messages=messages,
                    **PromptPresets.merge_with_overrides(PromptPresets.DETERMINISTIC),
                )

                content = get_simple_response_content(response).strip()
                if content:
                    top_titles = [
                        getattr(rp.paper, "title", "")
                        for rp in rag_result.papers[:3]
                        if getattr(rp.paper, "title", "")
                    ]
                    titles_str = (
                        "; ".join(top_titles)
                        if top_titles
                        else "No top titles extracted"
                    )
                    return f"Sub-agent objective: {objective}.\nSummary: {content}\nTop supporting papers: {titles_str}."
            except Exception as e:
                logger.warning(
                    f"Failed to generate LLM summary for sub-agent, using fallback: {e}"
                )

        # Fallback to heuristic
        paper_count = len(rag_result.papers)
        chunk_count = len(rag_result.chunks)
        top_titles: List[str] = []
        for ranked in rag_result.papers[:3]:
            title = getattr(ranked.paper, "title", "")
            if title:
                top_titles.append(title)

        titles = "; ".join(top_titles) if top_titles else "No top titles extracted"
        return (
            f"Sub-agent objective: {objective}. "
            f"Retrieved {paper_count} papers and {chunk_count} chunks. "
            f"Top supporting papers: {titles}."
        )

    async def _build_literature_review_brief(
        self,
        *,
        query: str,
        agent_outputs: List[Dict[str, Any]],
    ) -> str:
        if not agent_outputs:
            return ""

        structured_items = []
        for output in agent_outputs:
            structured_items.append(
                {
                    "agent_id": output.get("agent_id"),
                    "objective": output.get("objective"),
                    "summary": output.get("summary"),
                    "citations": output.get("citations", []),
                }
            )

        response_builder = ChatResponseBuilder()
        findings_lines: List[str] = []
        for item in structured_items:
            citations = item.get("citations", []) or []
            citations_text = " ".join(
                f"(cite:{str(c)})" for c in citations if str(c).strip()
            )
            findings_lines.append(
                (
                    f"AGENT: {item.get('agent_id')}\n"
                    f"OBJECTIVE: {item.get('objective')}\n"
                    f"SUMMARY: {item.get('summary')}\n"
                    f"CITATIONS: {citations_text or 'none'}"
                )
            )

        synthesis_context = "\n\n".join(findings_lines)
        enhanced_query = response_builder.build_enhanced_query(
            query=query,
            conversation_history=None,
            context_string=synthesis_context,
        )

        try:
            messages, _ = PromptBuilder.build(
                prompt_name="generate_literature_review_brief",
                user_input=enhanced_query,
            )

            response = self.container.llm_service.llm_provider.simple_prompt(
                messages=messages,
                **PromptPresets.merge_with_overrides(PromptPresets.DETERMINISTIC),
            )

            content = get_simple_response_content(response).strip()
            if content:
                return f"# Research Synthesis\n\n{content}\n"
        except Exception as exc:
            logger.warning(
                "Literature review LLM brief generation failed, fallback used: %s", exc
            )

        lines = [f"**Main Original Query**: {query}"]
        for output in structured_items:
            lines.append(
                f"- {output.get('objective')}: {output.get('summary')} (citations: {output.get('citations')})"
            )
        return "\n".join(lines)

    @staticmethod
    def _build_semantic_scholar_queries(
        *,
        primary_query: str,
        clarified_question: Optional[str],
        semantic_queries: Optional[List[str]],
        fallback_queries: List[str],
    ) -> List[str]:
        normalized_semantic = AgentGraphOrchestrator._normalize_queries(
            semantic_queries
        )
        if normalized_semantic:
            return normalized_semantic

        preferred_seed = (clarified_question or primary_query).strip()
        semantic_query = AgentGraphOrchestrator._to_semantic_scholar_query(
            preferred_seed
        )

        normalized_fallbacks = AgentGraphOrchestrator._normalize_queries(
            fallback_queries
        )
        if (
            normalized_fallbacks
            and semantic_query.lower() != normalized_fallbacks[0].lower()
        ):
            return [semantic_query, normalized_fallbacks[0]]

        return [semantic_query]

    @staticmethod
    def _normalize_queries(queries: Optional[List[str]]) -> List[str]:
        if not queries:
            return []

        normalized: List[str] = []
        seen: set[str] = set()
        for query in queries:
            q = (query or "").strip()
            if not q:
                continue
            q_key = q.lower()
            if q_key in seen:
                continue
            seen.add(q_key)
            normalized.append(q)

        return normalized

    @staticmethod
    def _to_semantic_scholar_query(query: str) -> str:
        cleaned = re.sub(r"\s+", " ", query or "").strip()
        cleaned = cleaned.rstrip("?.! ")

        conversational_prefixes = [
            "what is ",
            "what are ",
            "how does ",
            "how do ",
            "can you ",
            "could you ",
            "please ",
            "tell me ",
            "show me ",
            "find ",
        ]

        lowered = cleaned.lower()
        for prefix in conversational_prefixes:
            if lowered.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                break

        if not cleaned:
            return "academic research papers"

        lowered_cleaned = cleaned.lower()
        if not any(
            token in lowered_cleaned
            for token in ["paper", "study", "research", "survey"]
        ):
            cleaned = f"{cleaned} research papers"

        return cleaned

    @staticmethod
    def _build_system_explanation_response() -> str:
        default_response = (
            "Inquira is an academic research assistant with a retrieval-first workflow.\n\n"
            "1) It decomposes your question into focused academic queries and intent.\n"
            "2) It searches local indexed papers first (database-first strategy).\n"
            "3) If local coverage is weak, it expands to external scholarly search (for example, Semantic Scholar) with adaptive queries.\n"
            "4) It ingests selected external papers, re-runs retrieval and ranking, then generates a citation-grounded answer.\n"
            "5) Progress is streamed live (planning, searching, ranking, reasoning) for transparency.\n\n"
            "If you want, ask a specific topic and I can run the full retrieval pipeline."
        )
        try:
            content = (
                (_RESPONSES_DIR / "system.txt").read_text(encoding="utf-8").strip()
            )
            return content or default_response
        except Exception:
            return default_response

    async def _node_general_agent(
        self,
        state: AgentGraphState,
    ) -> AgentGraphState:
        """Direct LLM answer for normal non-retrieval requests."""
        system_prompt = (
            "You are Inquira. Answer as a helpful assistant for normal user requests "
            "that do not require academic retrieval. "
            "Be concise, clear, and avoid claiming citations or external retrieval was used."
        )
        query = state.get("query") or ""
        try:
            conversation_history = state.get("conversation_history", [])

            response = self.container.llm_service.llm_provider.simple_prompt(
                messages=[
                    {"role": "system", "content": system_prompt},
                    *conversation_history,
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                **PromptPresets.merge_with_overrides(PromptPresets.CREATIVE),
            )
            content = get_simple_response_content(response).strip()
            if content:
                for evt in stream_like_llm(content):
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
                    "direct_response": content,
                    "response_text": content,
                    "retrieved_paper_ids": [],
                    "paper_snapshots": [],
                }
        except Exception as exc:
            logger.warning("General direct response generation failed: %s", exc)

        fallback = "Sorry, I couldn't generate a response to that request. Please try asking in a different way or ask about a specific research topic for me to assist with."
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
            "retrieved_paper_ids": [],
            "paper_snapshots": [],
        }

    async def _plan_initial_external_queries(
        self,
        *,
        query: str,
        intent: QueryIntent,
        breakdown: GeneratedQueryPlanResponse,
        base_queries: List[str],
    ) -> List[str]:
        """Let the agent create initial Semantic Scholar queries (title or short keyword)."""
        system_prompt = (
            "You are a ReAct search planner for Semantic Scholar. "
            "Generate concise external search queries as either exact paper titles or short keyword phrases. "
            "Do not generate long conversational sentences. "
            'Return ONLY JSON: {"queries": [string], "query_forms": ["title"|"keyword"], "reason": string}.'
        )

        payload = {
            "user_query": query,
            "intent": intent.value,
            "clarified_question": breakdown.clarified_question,
            "specific_papers": breakdown.specific_papers or [],
            "semantic_queries": breakdown.hybrid_queries or [],
            "search_queries": breakdown.hybrid_queries or [],
            "candidate_queries": base_queries,
            "constraints": {
                "max_queries": 6,
                "prefer_title_or_short_keywords": True,
            },
        }

        try:
            response = self.container.llm_service.llm_provider.simple_prompt(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                **PromptPresets.merge_with_overrides(
                    PromptPresets.DETERMINISTIC,
                    response_format={"type": "json_object"},
                ),
            )

            raw_content = get_simple_response_content(response).strip()
            data = json.loads(raw_content)
            raw_queries = data.get("queries", [])
            if not isinstance(raw_queries, list):
                raw_queries = []

            planned_queries = [self._sanitize_external_query(q) for q in raw_queries]
            planned_queries = [q for q in planned_queries if q]
            normalized = self._normalize_queries(planned_queries)
            if normalized:
                return normalized[:6]
        except Exception as exc:
            logger.warning(
                "Initial external query planner failed, fallback to base queries: %s",
                exc,
            )

        fallback = [self._sanitize_external_query(q) for q in base_queries]
        fallback = [q for q in fallback if q]
        return self._normalize_queries(fallback)[:6]

    async def _react_external_action(
        self,
        *,
        query: str,
        intent: QueryIntent,
        cycle_num: int,
        max_cycles: int,
        hard_stop_cycles: int,
        initial_result: Optional[RAGResult],
        total_external_candidates: int,
        query_history: List[str],
        cycle_summaries: List[Dict[str, Any]],
        latest_cycle_candidates: List[PaperEnrichedDTO],
        candidate_queries: List[str],
        default_probe_limit: int,
        default_full_limit: int,
    ) -> Dict[str, Any]:
        """ReAct controller: decide continue/stop, next query, and semantic limit for this cycle."""
        if cycle_num > hard_stop_cycles:
            return {
                "continue_search": False,
                "reason": "hard_stop_cycle_limit",
                "next_query": "",
                "query_form": "keyword",
                "semantic_limit": default_full_limit,
            }

        if cycle_num > max_cycles:
            return {
                "continue_search": False,
                "reason": "max_cycles_reached",
                "next_query": "",
                "query_form": "keyword",
                "semantic_limit": default_full_limit,
            }

        paper_count = len(initial_result.papers) if initial_result else 0
        chunk_count = len(initial_result.chunks) if initial_result else 0
        top_score = (
            float(initial_result.papers[0].relevance_score)
            if initial_result
            and initial_result.papers
            and initial_result.papers[0].relevance_score is not None
            else 0.0
        )

        system_prompt = (
            "You are a ReAct controller for external academic search over Semantic Scholar. "
            "At each step, decide whether to continue searching, what SINGLE query to run next, "
            "and how many papers to fetch. Query must be either an exact title or short keywords. "
            "Prefer title form when searching for a specific paper; otherwise use concise keyword form. "
            "Return ONLY JSON: "
            '{"continue_search": bool, "next_query": string, "query_form": "title"|"keyword", "semantic_limit": int, "reason": string}.'
        )

        payload = {
            "user_query": query,
            "intent": intent.value,
            "cycle": cycle_num,
            "max_cycles": max_cycles,
            "hard_stop_cycles": hard_stop_cycles,
            "initial_db": {
                "paper_count": paper_count,
                "chunk_count": chunk_count,
                "top_score": top_score,
            },
            "total_external_candidates": total_external_candidates,
            "query_history": query_history[-8:],
            "candidate_queries": candidate_queries[:8],
            "cycle_summaries": cycle_summaries[-4:],
            "latest_cycle_evidence": self._build_external_cycle_evidence(
                latest_cycle_candidates
            ),
            "constraints": {
                "semantic_limit_min": 8,
                "semantic_limit_max": 60,
                "short_keyword_query": True,
                "title_query_allowed": True,
            },
        }

        try:
            response = self.container.llm_service.llm_provider.simple_prompt(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                **PromptPresets.merge_with_overrides(
                    PromptPresets.DETERMINISTIC,
                    response_format={"type": "json_object"},
                ),
            )

            raw_content = get_simple_response_content(response).strip()
            data = json.loads(raw_content)

            continue_search = bool(data.get("continue_search", False))
            next_query = self._sanitize_external_query(
                str(data.get("next_query", "") or "")
            )
            query_form = (
                str(data.get("query_form", "keyword") or "keyword").strip().lower()
            )
            if query_form not in {"title", "keyword"}:
                query_form = "keyword"

            semantic_limit_raw = int(
                data.get("semantic_limit", default_full_limit) or default_full_limit
            )
            semantic_limit = max(8, min(60, semantic_limit_raw))
            reason = str(data.get("reason", "react_agent_decision"))

            if continue_search and not next_query:
                continue_search = False
                reason = "react_missing_next_query"

            return {
                "continue_search": continue_search,
                "next_query": next_query,
                "query_form": query_form,
                "semantic_limit": semantic_limit,
                "reason": reason,
            }
        except Exception as exc:
            logger.warning(
                "ReAct external controller failed, fallback action used: %s", exc
            )

        fallback_query = ""
        for candidate in candidate_queries:
            normalized = self._sanitize_external_query(candidate)
            if normalized and normalized.lower() not in {
                q.lower() for q in query_history
            }:
                fallback_query = normalized
                break
        if not fallback_query and candidate_queries:
            fallback_query = self._sanitize_external_query(candidate_queries[0])

        return {
            "continue_search": bool(fallback_query),
            "next_query": fallback_query,
            "query_form": "keyword",
            "semantic_limit": (
                default_probe_limit if cycle_num == 1 else default_full_limit
            ),
            "reason": "react_fallback_action",
        }

    @staticmethod
    def _sanitize_external_query(query: str) -> str:
        """Keep Semantic Scholar query as title-like or short keyword phrase."""
        value = re.sub(r"\s+", " ", str(query or "")).strip().strip("?.!;")
        if not value:
            return ""

        # keep exact-title quoting if already present
        if value.startswith('"') and value.endswith('"') and len(value) > 2:
            return value

        tokens = value.split(" ")
        if len(tokens) > 10:
            value = " ".join(tokens[:10])
        return value.strip()

    async def _evaluate_agent_results(
        self, objective: str, result: Optional[RAGResult], mode: str
    ) -> bool:
        """Second-level check LLM Evaluator for retrieved context."""
        if not result or not result.chunks:
            return False

        system_prompt = (
            "You are a critical research evaluator acting as a second-level check. "
            f"Your objective is to answer: '{objective}'\n\n"
            "Review the provided retrieval excerpts. Does the context contain sufficient "
            "detailed evidence to comprehensively answer the objective, or do we need to search externally/further? "
            'Respond ONLY with a JSON object: {"sufficient": true/false, "reason": "string"}'
        )

        context_text = "\n\n".join(
            f"Excerpt {i+1}:\n{AgentGraphOrchestrator._truncate_text(getattr(c, 'text', ''), 400)}"
            for i, c in enumerate(result.chunks[:12])
        )
        payload = {"context": context_text}

        try:
            data = self.container.llm_service.prompt_json(
                system_prompt=system_prompt,
                user_payload=json.dumps(payload)[:8000],
            )

            is_sufficient = bool(data.get("sufficient", False))
            reason = data.get("reason", "")
            await self._emit_step(
                f"evaluating_{mode}",
                f"LLM Evaluator checked {mode} context. Sufficient? {is_sufficient}. Reason: {reason}",
                phase=PipelinePhase.SEARCH,
                progress_percent=35 if mode == "internal" else 55,
                metadata={"sufficient": is_sufficient, "reason": reason},
            )

            return is_sufficient
        except Exception as e:
            logger.warning("Sub-agent evaluator failed, fallback to heuristic: %s", e)
            return False

    @staticmethod
    def _truncate_text(text: Optional[str], max_chars: int) -> str:
        value = (text or "").strip()
        if not value:
            return ""
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 1].rstrip() + "…"

    @staticmethod
    def _extract_bibtex(paper: PaperEnrichedDTO) -> str:
        styles = (
            paper.citation_styles if isinstance(paper.citation_styles, dict) else {}
        )
        bibtex = styles.get("bibtex") if isinstance(styles, dict) else ""
        return AgentGraphOrchestrator._truncate_text(str(bibtex or ""), 320)

    @classmethod
    def _build_external_cycle_evidence(
        cls,
        candidates: List[PaperEnrichedDTO],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        max_items = 8
        max_abstract_items = 4
        max_bibtex_items = 3

        evidence_rows: List[Dict[str, Any]] = []
        for idx, paper in enumerate(candidates[:max_items]):
            row: Dict[str, Any] = {
                "paper_id": paper.paper_id,
                "title": cls._truncate_text(paper.title, 220),
                "year": paper.year,
                "venue": cls._truncate_text(paper.venue, 120),
                "citation_count": paper.citation_count,
                "is_open_access": bool(paper.is_open_access),
                "has_pdf_url": bool(paper.pdf_url),
            }

            if idx < max_bibtex_items:
                bibtex = cls._extract_bibtex(paper)
                if bibtex:
                    row["bibtex"] = bibtex

            if idx < max_abstract_items:
                abstract_excerpt = cls._truncate_text(paper.abstract, 280)
                if abstract_excerpt:
                    row["abstract_excerpt"] = abstract_excerpt

            evidence_rows.append(row)

        return evidence_rows

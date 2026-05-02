"""
LLM Service that integrates with the retriever services
"""

from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from app.extensions.stream import (
    get_simple_response_content,
    get_simple_response_reasoning,
)
from app.llm import LiteLLMProvider
from app.llm.prompts import PromptPresets, PromptBuilder
from app.llm.prompts.prompt_configs import PromptConfig
from app.llm.schemas import (
    QuestionBreakdownResponse,
    GeneratedQueryPlanResponse,
    QueryIntent,
    RelatedTopicsResponse,
)
from app.extensions.logger import create_logger
import re
import json

logger = create_logger(__name__)


class LLMService:
    """Service class that provides LLM functionality for the application"""

    def __init__(self):
        self.llm_provider = LiteLLMProvider()

    async def decompose_user_query(
        self,
        user_question: str,
        num_subtopics: Optional[int] = None,
        include_explanation: Optional[bool] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **llm_params,
    ) -> QuestionBreakdownResponse:
        """
        Break down a user's question into focused sub-topics for clearer understanding
        Let the LLM decide the optimal number of subtopics and whether explanations are needed

        Args:
            user_question: The user's original question
            num_subtopics: Number of sub-topics to generate (None = let LLM decide, typically 1-3)
            include_explanation: Whether to include explanations (None = let LLM decide based on complexity)
            conversation_history: Optional conversation history for context (list of {role, content} dicts)

        Returns:
            Dictionary with original question, sub-topics, and optional explanations

        Example:
            Input: "How does machine learning work?"
            Output: {
                "original_question": "How does machine learning work?",
                "clarified_question": "Understanding the fundamentals and implementation of machine learning",
                "subtopics": [
                    "Definition and basic concepts of machine learning",
                    "Types of machine learning (supervised, unsupervised, reinforcement)",
                    "How machine learning algorithms learn from data",
                ],
                "num_subtopics": 3,
                "complexity": "intermediate"
            }
        """

        # Build prompt with optional conversation context
        if conversation_history:
            history_text = "\n".join(
                [
                    f"{msg['role'].upper()}: {msg['content']}"
                    for msg in conversation_history[-5:]  # Last 5 messages for context
                ]
            )
            prompt = f"""
        Previous Conversation:
        {history_text}
        
        Current Question: "{user_question}"
        Required Search Queries: {num_subtopics or "1-2"}
        
        NOTE: Consider the conversation context when decomposing the query. If the question uses pronouns like "it", "they", "this", or "that", resolve them using the conversation history.
        """
        else:
            prompt = f"""
        User Question: "{user_question}"
        Required Search Queries: {num_subtopics or "1-2"}
        """

        messages, version = PromptBuilder.build(
            prompt_name="decompose_query",
            user_input=prompt,
            additional_content=None,
            dynamic_instruction=None,
        )

        config = PromptPresets.merge_with_overrides(
            PromptPresets.DETERMINISTIC,
            **llm_params,
        )

        response = self.llm_provider.simple_prompt(messages=messages, **config)
        logger.info(f"LLM response for question breakdown: {response}")

        lines = get_simple_response_content(response).split("\n")
        clarified_question = ""
        bm25_query: Optional[str] = None
        semantic_queries: List[str] = []
        specific_papers: List[str] = []

        intent_str: Optional[str] = None
        skip_flags: List[str] = []
        filters_dict: Dict[str, Any] = {}

        current_section: Optional[str] = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for section headers
            if line.startswith("CLARIFIED:"):
                clarified_text = line.replace("CLARIFIED:", "").strip()
                clarified_question = clarified_text.strip('"').strip()
                current_section = None
            elif (
                "KEYWORD_QUERIES:" in line.upper() or "KEYWORD QUERIES" in line.upper()
            ):
                current_section = "keyword"
                continue
            elif (
                "SEMANTIC_QUERIES:" in line.upper()
                or "SEMANTIC QUERIES" in line.upper()
            ):
                current_section = "semantic"
                continue
            elif (
                "SPECIFIC_PAPERS:" in line.upper() or "SPECIFIC PAPERS" in line.upper()
            ):
                current_section = "specific"
                continue
            elif line.startswith("INTENT:"):
                intent_str = line.replace("INTENT:", "").strip().lower()
                current_section = None
            elif line.startswith("SKIP:"):
                skip_text = line.replace("SKIP:", "").strip().lower()
                if skip_text != "none":
                    skip_flags = [s.strip() for s in skip_text.split(",")]
                current_section = None
            elif line.startswith("FILTERS:"):
                filters_text = line.replace("FILTERS:", "").strip()
                if filters_text.lower() != "none":
                    # Parse key=value pairs
                    for pair in filters_text.split(","):
                        if "=" in pair:
                            key, value = pair.split("=", 1)
                            filters_dict[key.strip()] = value.strip()
                current_section = None
            elif current_section and line:
                # Skip section description lines
                if line.upper().startswith("THESE") or line.upper().startswith("THE "):
                    continue
                if line.startswith("(") and line.endswith(")"):
                    continue

                # Remove numbered prefixes and cleanup
                clean_line = line
                if re.match(r"^\d+\.", clean_line):
                    clean_line = re.sub(r"^\d+\.\s*", "", clean_line)

                # Remove bullet points
                for prefix in ["•", "-", "*", ">", "○", "▪"]:
                    if clean_line.startswith(prefix):
                        clean_line = clean_line[len(prefix) :].strip()
                        break

                # Remove markdown and quotes
                clean_line = clean_line.replace("**", "").strip()
                clean_line = clean_line.strip('"').strip("'").strip()

                # Skip short or empty lines
                if len(clean_line) < 3:
                    continue

                # Add to appropriate list
                if current_section == "keyword":
                    bm25_query = clean_line
                elif current_section == "semantic":
                    semantic_queries.append(clean_line)
                elif current_section == "specific":
                    specific_papers.append(clean_line)

        all_queries = (
            [bm25_query] + semantic_queries if bm25_query else semantic_queries
        )
        subtopics = [
            s.strip() for s in all_queries if s.strip() and len(s.strip()) > 5
        ][:4]
        reasoning_content = get_simple_response_reasoning(response)

        query_intent: Optional[QueryIntent] = None
        intent_confidence: Optional[float] = None
        if intent_str:
            try:
                query_intent = QueryIntent(intent_str)
                intent_confidence = 0.9  # High confidence since directly from LLM
            except ValueError:
                logger.warning(
                    f"Invalid intent: {intent_str}, defaulting to comprehensive_search"
                )
                query_intent = QueryIntent.COMPREHENSIVE_SEARCH
                intent_confidence = 0.5

        skip_ranking = "ranking" in skip_flags
        skip_title_filter = "title_filter" in skip_flags or "filter" in skip_flags

        return QuestionBreakdownResponse(
            original_question=user_question,
            clarified_question=clarified_question or user_question,
            search_queries=subtopics,
            bm25_query=bm25_query if bm25_query else None,
            semantic_queries=semantic_queries if semantic_queries else None,
            specific_papers=specific_papers if specific_papers else None,
            num_queries=len(subtopics),
            complexity="simple",
            model_used=self.llm_provider.get_model(),
            intent=query_intent,
            intent_confidence=intent_confidence,
            skip_ranking=skip_ranking,
            skip_title_abstract_filter=skip_title_filter,
            filters=filters_dict if filters_dict else None,
        )

    async def decompose_user_query_v2(
        self,
        user_question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **llm_params,
    ) -> QuestionBreakdownResponse:
        """
        V2: Break down user query using JSON output format for better structured parsing.
        Uses decompose_query_v2 prompt which returns structured JSON.

        Args:
            user_question: The user's original question
            num_subtopics: Number of search queries to generate (default: let LLM decide 1-3)
            conversation_history: Optional conversation history for context
            temperature: LLM temperature override
            max_tokens: Max tokens override
            **llm_params: Additional LLM parameters

        Returns:
            QuestionBreakdownResponse with structured query decomposition
        """
        import json

        if conversation_history:
            history_text = "\n".join(
                [
                    f"{msg['role'].upper()}: {msg['content']}"
                    for msg in conversation_history[-5:]
                ]
            )
            prompt = f"""
<conversation_context>
{history_text}
</conversation_context>

NOTE: Use the <conversation_context> above strictly to resolve pronouns (e.g., "it", "they", "this method") in the user's current question. Do not generate searches for old questions.
"""
            prompt += f'Current user Question: "{user_question}"'
        else:
            prompt = f'User Question: "{user_question}"'

        messages, version = PromptBuilder.build(
            prompt_name="decompose_query_v2",
            user_input=prompt,
            additional_content=None,
            dynamic_instruction=None,
        )

        config = PromptPresets.merge_with_overrides(
            PromptPresets.DETERMINISTIC,
            response_format={"type": "json_object"},  # Force JSON output
            **llm_params,
        )

        response = self.llm_provider.simple_prompt(messages=messages, **config)
        content = get_simple_response_content(response)
        reasoning_content = get_simple_response_reasoning(response)

        logger.info(f"LLM V2 response for question breakdown: {content[:200]}...")

        try:
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}\nContent: {content}")
            # Fallback to simple breakdown
            return QuestionBreakdownResponse(
                original_question=user_question,
                clarified_question=user_question,
                search_queries=[user_question],
                num_queries=1,
                complexity="simple",
                model_used=self.llm_provider.get_model(),
            )

        clarified_question = data.get("clarified_question", user_question)
        search_queries = data.get("search_queries", [])
        specific_papers = data.get("specific_papers", [])
        intent_str = data.get("intent", "COMPREHENSIVE").lower()
        filters_dict = data.get("filters", {})
        bm25_query = data.get("bm25_query")
        semantic_queries = data.get("semantic_queries")
        if not bm25_query and not semantic_queries and search_queries:
            bm25_query = (
                search_queries[:2] if len(search_queries) > 1 else search_queries
            )
            semantic_queries = search_queries[1:] if len(search_queries) > 1 else []

        query_intent = QueryIntent.COMPREHENSIVE_SEARCH
        intent_confidence = 0.9

        intent_mapping = {
            "author_papers": QueryIntent.AUTHOR_PAPERS,
            "comparison": QueryIntent.COMPARISON,
            "foundational": QueryIntent.FOUNDATIONAL,
            "comprehensive": QueryIntent.COMPREHENSIVE_SEARCH,
        }

        query_intent = intent_mapping.get(intent_str, QueryIntent.COMPREHENSIVE_SEARCH)

        skip_flags = data.get("skip", [])
        if isinstance(skip_flags, str):
            skip_flags = [s.strip() for s in skip_flags.split(",")]

        skip_ranking = "ranking" in skip_flags
        skip_title_filter = "title_filter" in skip_flags or "filter" in skip_flags

        if filters_dict:
            filters_dict = {k: v for k, v in filters_dict.items() if v and v != "null"}

        return QuestionBreakdownResponse(
            original_question=user_question,
            clarified_question=clarified_question,
            search_queries=search_queries,
            bm25_query=bm25_query if bm25_query else None,
            semantic_queries=semantic_queries if semantic_queries else None,
            specific_papers=specific_papers if specific_papers else None,
            num_queries=len(search_queries),
            complexity="simple",
            model_used=self.llm_provider.get_model(),
            intent=query_intent,
            intent_confidence=intent_confidence,
            skip_ranking=skip_ranking,
            skip_title_abstract_filter=skip_title_filter,
            filters=filters_dict if filters_dict else None,
        )

    async def decompose_user_query_v3(
        self,
        user_question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **llm_params,
    ) -> GeneratedQueryPlanResponse:
        """
        V3: Experimental version that uses a more advanced prompt and expects a structured JSON response with enhanced reasoning and context handling.
        This version is designed to better handle complex questions and multi-turn conversations.

        Args:
            user_question: The user's original question
            conversation_history: Optional conversation history for context

        Returns:
            QuestionBreakdownResponse with structured query decomposition and enhanced reasoning
        """
        import json

        if conversation_history:
            history_text = "\n".join(
                [
                    f"{msg['role'].upper()}: {msg['content']}"
                    for msg in conversation_history[-5:]
                ]
            )
            prompt = f"""
<conversation_context>
{history_text}
</conversation_context>

NOTE: Use the <conversation_context> above strictly to resolve pronouns (e.g., "it", "they", "this method") in the user's current question. Do not generate searches for old questions.
"""
            prompt += f'Current user Question: "{user_question}"'
        else:
            prompt = f'User Question: "{user_question}"'

        messages, version = PromptBuilder.build(
            prompt_name="decompose_query_v3",
            user_input=prompt,
            additional_content=None,
            dynamic_instruction=None,
        )

        config = PromptPresets.merge_with_overrides(
            PromptPresets.DETERMINISTIC,
            response_format={"type": "json_object"},
            **llm_params,
        )

        response = self.llm_provider.simple_prompt(messages=messages, **config)
        content = get_simple_response_content(response)

        try:
            raw = json.loads(content)
            if not isinstance(raw, dict):
                raw = {}
        except Exception:
            logger.warning("decompose_user_query_v3 returned invalid JSON; using fallback")
            raw = {}

        clarified_question = str(raw.get("clarified_question") or user_question).strip()
        hybrid_queries_raw = raw.get("hybrid_queries")
        hybrid_queries = []

        if isinstance(hybrid_queries_raw, list):
            hybrid_queries = [
                str(q).strip() for q in hybrid_queries_raw if str(q).strip()
            ]

        # Backward compatibility: accept older v2-style fields when hybrid_queries
        # is not present, then normalize to the unified hybrid list.
        if not hybrid_queries:
            bm25_query_raw = raw.get("bm25_query")
            bm25_query = str(bm25_query_raw).strip() if bm25_query_raw else None

            semantic_queries_raw = raw.get("semantic_queries") or []
            semantic_queries = (
                [str(q).strip() for q in semantic_queries_raw if str(q).strip()]
                if isinstance(semantic_queries_raw, list)
                else []
            )

            hybrid_queries = []
            if bm25_query:
                hybrid_queries.append(bm25_query)
            hybrid_queries.extend(semantic_queries)

        hybrid_queries = list(dict.fromkeys([q for q in hybrid_queries if q]))
        if not hybrid_queries:
            hybrid_queries = [clarified_question]

        specific_papers_raw = raw.get("specific_papers") or []
        specific_papers = [
            str(p).strip() for p in specific_papers_raw if str(p).strip()
        ] if isinstance(specific_papers_raw, list) else []

        intent_raw = str(raw.get("intent") or QueryIntent.COMPREHENSIVE_SEARCH.value).strip().lower()
        try:
            intent = QueryIntent(intent_raw)
        except ValueError:
            intent = QueryIntent.COMPREHENSIVE_SEARCH

        skip_raw = raw.get("skip") or []
        if isinstance(skip_raw, list):
            skip = [str(flag).strip().lower() for flag in skip_raw if str(flag).strip()]
        elif isinstance(skip_raw, str):
            skip = [value.strip().lower() for value in skip_raw.split(",") if value.strip()]
        else:
            skip = []

        filters_raw = raw.get("filters")
        filters = filters_raw if isinstance(filters_raw, dict) else {}

        return GeneratedQueryPlanResponse(
            original_question=user_question,
            clarified_question=clarified_question,
            hybrid_queries=hybrid_queries,
            specific_papers=specific_papers,
            intent=intent,
            skip=skip,
            filters=filters,
        )

    def prompt_json(
        self,
        *,
        system_prompt: str,
        user_payload: Union[str, Dict[str, Any], List[Any]],
        preset: Optional[PromptConfig] = None,
        **llm_params,
    ) -> Dict[str, Any]:
        """Execute a deterministic JSON prompt and return parsed object safely."""
        if isinstance(user_payload, str):
            user_content = user_payload
        else:
            user_content = json.dumps(user_payload)

        config = PromptPresets.merge_with_overrides(
            preset or PromptPresets.DETERMINISTIC,
            response_format={"type": "json_object"},
            **llm_params,
        )

        response = self.llm_provider.simple_prompt(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            **config,
        )

        content = get_simple_response_content(response).strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
            return {}
        except Exception:
            logger.warning("Failed to parse JSON response in prompt_json")
            return {}

    async def summarize_conversation_context(
        self,
        conversation_text: str,
        existing_summary: Optional[str] = None,
        temperature: Optional[float] = 0.3,
        max_tokens: Optional[int] = 800,
        **llm_params,
    ) -> str:
        """
        Generate a summary of conversation context using LLM.

        This method handles the prompt structure and LLM interaction for conversation summarization.
        Can create fresh summaries or incremental summaries that build on previous ones.

        Args:
            conversation_text: Formatted conversation text to summarize
            existing_summary: Previous summary (if any) for incremental summarization
            temperature: LLM temperature (default: 0.3 for consistent summaries)
            max_tokens: Maximum tokens for summary (default: 800, roughly 500 words)
            **llm_params: Additional LLM parameters

        Returns:
            Generated summary text

        Raises:
            Exception: If LLM call fails
        """
        if existing_summary:
            prompt = f"""Here's the previous summary:

<previous_summary>
{existing_summary}
</previous_summary>

And here are the recent messages since that summary:

<recent_messages>
{conversation_text}
</recent_messages>

Create a concise but comprehensive updated summary that:
1. Preserves key information from the previous summary
2. Integrates new information from recent messages
3. Maintains chronological flow
4. Highlights important research questions, findings, and decisions
5. Keeps the summary under 500 words

Updated Summary:"""
        else:
            # Fresh summarization prompt
            prompt = f"""Summarize this research conversation concisely but comprehensively:

<conversation>
{conversation_text}
</conversation>

Create a summary that:
1. Captures the main research topics and questions discussed
2. Highlights key findings and papers mentioned
3. Notes any important decisions or conclusions
4. Maintains chronological flow of the discussion
5. Keeps the summary under 500 words

Summary:"""

        config = PromptPresets.merge_with_overrides(
            PromptPresets.SUMMARIZATION,
            max_tokens=max_tokens,
            **llm_params,
        )

        messages, version = PromptBuilder.build(
            prompt_name="conversation_summarization",
            user_input=prompt,
            additional_content=None,
            dynamic_instruction=None,
        )

        response = self.llm_provider.simple_prompt(
            messages=messages,
            **config,
        )

        summary = get_simple_response_content(response)

        if not summary:
            raise ValueError("LLM returned empty summary")

        return summary.strip()

    async def stream_citation_based_response(
        self,
        context: str,
        prompt_name: str = "generate_answer",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **llm_params,
    ) -> AsyncGenerator[Any, None]:
        """
        Stream a citation-based response with thought process

        Args:
            query: User's question
            context: Retrieved papers/documents (pre-formatted string or list of dicts for backwards compatibility)

        Yields:
            Formatted chunks including thought steps and citations
        """
        prompt = context

        print(f"[DEBUG] Starting to stream completion...")
        chunk_count = 0
        messages = PromptBuilder.build(
            prompt_name=prompt_name,
            user_input=prompt,
            additional_content=None,
            dynamic_instruction=None,
        )[0]

        config = PromptPresets.merge_with_overrides(
            PromptPresets.FACTUAL,
            **llm_params,
        )

        for chunk in self.llm_provider.stream_completion(messages=messages, **config):
            chunk_count += 1
            if chunk_count % 10 == 0:
                print(f"[DEBUG] Streamed {chunk_count} chunks so far...")

            yield chunk

        print(f"[DEBUG] Finished streaming. Total chunks: {chunk_count}")

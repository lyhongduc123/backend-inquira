"""
LLM Service that integrates with the retriever services
"""

from typing import List, Dict, Any, Optional, AsyncGenerator, Union

from click import prompt
from litellm import Choices
from typer import prompt
from app.extensions.stream import (
    get_simple_response_content,
    get_simple_response_reasoning,
)
from app.llm import LiteLLMProvider
from app.llm.prompts import PromptPresets, PromptBuilder, PROMPT_REGISTRY
from app.llm.schemas import (
    QuestionBreakdownResponse,
    QueryIntent,
    RelatedTopicsResponse
)
from app.core.config import settings
from app.extensions.logger import create_logger
import re

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
            PromptPresets.FACTUAL,
            temperature=temperature or settings.LLM_FACTUAL_TEMPERATURE,
            max_tokens=max_tokens,
            **llm_params,
        )

        response = self.llm_provider.simple_prompt(messages=messages, **config)
        logger.info(f"LLM response for question breakdown: {response}")

        lines = get_simple_response_content(response).split("\n")
        clarified_question = ""
        keyword_queries: List[str] = []
        semantic_queries: List[str] = []
        specific_papers: List[str] = []
        explanations: List[str] = []
        
        intent_str: Optional[str] = None
        skip_flags: List[str] = []
        diversity = False
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
            elif "KEYWORD_QUERIES:" in line.upper() or "KEYWORD QUERIES" in line.upper():
                current_section = "keyword"
                continue
            elif "SEMANTIC_QUERIES:" in line.upper() or "SEMANTIC QUERIES" in line.upper():
                current_section = "semantic"
                continue
            elif "SPECIFIC_PAPERS:" in line.upper() or "SPECIFIC PAPERS" in line.upper():
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
            elif line.startswith("DIVERSITY:"):
                diversity_text = line.replace("DIVERSITY:", "").strip().lower()
                diversity = diversity_text in ["true", "yes", "1"]
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
                        clean_line = clean_line[len(prefix):].strip()
                        break

                # Remove markdown and quotes
                clean_line = clean_line.replace("**", "").strip()
                clean_line = clean_line.strip('"').strip("'").strip()

                # Skip short or empty lines
                if len(clean_line) < 3:
                    continue

                # Add to appropriate list
                if current_section == "keyword":
                    keyword_queries.append(clean_line)
                elif current_section == "semantic":
                    semantic_queries.append(clean_line)
                elif current_section == "specific":
                    specific_papers.append(clean_line)

        # Combine keyword and semantic queries for backward compatibility
        all_queries = keyword_queries + semantic_queries
        subtopics = [s.strip() for s in all_queries if s.strip() and len(s.strip()) > 5][:4]
        reasoning_content = get_simple_response_reasoning(response)

        query_intent: Optional[QueryIntent] = None
        intent_confidence: Optional[float] = None
        if intent_str:
            try:
                query_intent = QueryIntent(intent_str)
                intent_confidence = 0.9  # High confidence since directly from LLM
            except ValueError:
                logger.warning(f"Invalid intent: {intent_str}, defaulting to comprehensive_search")
                query_intent = QueryIntent.COMPREHENSIVE_SEARCH
                intent_confidence = 0.5

        skip_ranking = "ranking" in skip_flags
        skip_title_filter = "title_filter" in skip_flags or "filter" in skip_flags
        skip_pdf = "pdf" in skip_flags
        skip_embedding = "embedding" in skip_flags or "embed" in skip_flags

        return QuestionBreakdownResponse(
            original_question=user_question,
            clarified_question=clarified_question or user_question,
            search_queries=subtopics,
            keyword_queries=keyword_queries if keyword_queries else None,
            semantic_queries=semantic_queries if semantic_queries else None,
            specific_papers=specific_papers if specific_papers else None,
            num_queries=len(subtopics),
            complexity="simple", 
            explanations=None,
            has_explanations=False,
            reasoning_content=reasoning_content,
            model_used=self.llm_provider.get_model(),
            intent=query_intent,
            intent_confidence=intent_confidence,
            skip_ranking=skip_ranking,
            skip_title_abstract_filter=skip_title_filter,
            needs_diversity=diversity,
            filters=filters_dict if filters_dict else None,
        )

    def suggest_related_topics(
        self,
        current_topic: str,
        search_results: Optional[List[Dict[str, Any]]] = None,
        num_suggestions: int = 5,
        temperature: Optional[float] = None,
        **llm_params,
    ) -> RelatedTopicsResponse:
        """
        Suggest related research topics based on current topic and search results

        Args:
            current_topic: Current research topic
            search_results: Optional search results for context
            num_suggestions: Number of suggestions to generate

        Returns:
            List of related topic suggestions
        """
        context = ""
        if search_results:
            titles = [result.get("title", "") for result in search_results[:5]]
            context = f"\nBased on recent research: {', '.join(titles)}"

        prompt = f"""Suggest {num_suggestions} related research topics to: "{current_topic}"{context}
        
        The suggestions should be:
        - Related but distinct from the original topic
        - Currently relevant in academic research
        - Specific enough to be actionable
        - Covering different angles or approaches
        
        Return only the topic suggestions, one per line."""

        messages = [
            {
                "role": "system",
                "content": "You are a research strategist who identifies promising research directions.",
            },
            {"role": "user", "content": prompt},
        ]

        # Use CREATIVE preset for diverse suggestions
        config = PromptPresets.merge_with_overrides(
            PromptPresets.CREATIVE,
            temperature=temperature or settings.LLM_CREATIVE_TEMPERATURE,
            **llm_params,
        )

        response = self.llm_provider.simple_prompt(messages=messages, **config)

        # Parse suggestions
        suggestions = []
        for line in get_simple_response_content(response).split("\n"):
            line = line.strip()
            if line:
                # Check for bullet points or numbering
                has_marker = any(
                    line.startswith(marker)
                    for marker in [
                        "•",
                        "-",
                        "1.",
                        "2.",
                        "3.",
                        "4.",
                        "5.",
                        "6.",
                        "7.",
                        "8.",
                        "9.",
                    ]
                )

                if has_marker:
                    # Remove bullet points or numbering
                    clean_line = line.lstrip("•-").split(".", 1)[-1].strip()
                    if clean_line:
                        suggestions.append(clean_line)
                else:
                    suggestions.append(line)

        return RelatedTopicsResponse(
            current_topic=current_topic,
            suggestions=suggestions[:num_suggestions],
            num_suggestions=len(suggestions[:num_suggestions]),
            model_used=self.llm_provider.get_model(),
        )

    async def stream_citation_based_response(
        self,
        query: str,
        context: Union[str, List[Dict[str, Any]]],
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
        # Handle both string (new optimized format) and dict (legacy format)
        if isinstance(context, str):
            context_text = context
        else:
            # Legacy: Format context from list of dicts
            formatted_context = []
            for i, doc in enumerate(context, 1):
                title = doc.get("title", "Untitled")
                authors = doc.get("authors", [])
                year = doc.get("year", None)
                paper_id = doc.get("paper_id", doc.get("id", f"paper_{i}"))
                pdf_url = doc.get("pdf_url", "")
                url = doc.get("url", "")
                citation_count = doc.get("citationCount", doc.get("citation_count", 0))
                abstract = doc.get("abstract", "")

                # Prefer chunk_text (from vector search) over abstract
                chunk_text = doc.get("chunk_text")
                section = doc.get("section", doc.get("section_title"))

                if chunk_text:
                    content = chunk_text
                    if section:
                        content = f"[Section: {section}]\n{content}"
                else:
                    content = abstract or doc.get("content", "")

                # Handle authors
                if isinstance(authors, list):
                    if authors and isinstance(authors[0], dict):
                        authors_str = ", ".join(a.get("name", str(a)) for a in authors)
                    elif authors:
                        authors_str = ", ".join(str(a) for a in authors)
                    else:
                        authors_str = "Unknown"
                elif isinstance(authors, str):
                    authors_str = authors if authors else "Unknown"
                else:
                    authors_str = "Unknown"

                context_entry = f"[{i}] {title}\n"
                context_entry += f"Paper ID: {paper_id}\n"
                context_entry += f"Authors: {authors_str}\n"
                context_entry += f"Year: {year if year else 'N/A'}\n"
                context_entry += f"Citation Count: {citation_count}\n"
                context_entry += f"URL: {url}\n"
                context_entry += f"PDF URL: {pdf_url}\n"
                if section:
                    context_entry += f"Section: {section}\n"
                context_entry += f"Content: {content}\n"

                formatted_context.append(context_entry)

            context_text = "\n".join(formatted_context)

        prompt = f"""Question: {query}

        Available Research Papers:
        {context_text}
        """

        print(f"[DEBUG] Starting to stream completion...")
        chunk_count = 0
        messages = PromptBuilder.build(
            prompt_name="generate_answer",
            user_input=prompt,
            additional_content=None,
            dynamic_instruction=None,
        )[0]

        # Use FACTUAL preset for citation-based responses
        config = PromptPresets.merge_with_overrides(
            PromptPresets.FACTUAL,
            temperature=temperature or settings.LLM_FACTUAL_TEMPERATURE,
            max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            **llm_params,
        )
        
         
        for chunk in self.llm_provider.stream_completion(messages=messages, **config):
            chunk_count += 1
            if chunk_count % 10 == 0:
                print(f"[DEBUG] Streamed {chunk_count} chunks so far...")

            yield chunk

        print(f"[DEBUG] Finished streaming. Total chunks: {chunk_count}")
 
    
"""
LLM Service that integrates with the retriever services
"""
from typing import List, Dict, Any, Optional, AsyncGenerator
from app.llm import LLMProvider, ModelType
from app.llm.prompts import SummaryPrompts
from app.llm.schemas import (
    SearchSummaryResponse,
    PaperAnalysisResponse,
    QuestionBreakdownResponse,
    PaperComparisonResponse,
    ChatResponse,
    RelatedTopicsResponse,
    CitationBasedResponse,
    Citation,
    ThoughtStep
)
from app.core.config import settings
import json
import re


class LLMService():
    """Service class that provides LLM functionality for the application"""
    
    def __init__(self):
        self.llm_provider = LLMProvider(
            api_key=settings.OPENAI_API_KEY,
            default_model=ModelType.GPT_4O_MINI.value
        )
        self.prompts = SummaryPrompts
    
    def summarize_search_results(
        self, 
        search_results: List[Dict[str, Any]], 
        query: str,
        max_results: int = 5
    ) -> SearchSummaryResponse:
        """
        Summarize and synthesize search results from retrieval services
        
        Args:
            search_results: Results from semantic/arxiv search
            query: Original search query
            max_results: Maximum number of results to include in summary
        
        Returns:
            Synthesized summary with key insights
        """
        # Limit results to process
        limited_results = search_results[:max_results]
        
        # Format results for LLM processing
        formatted_results = []
        for i, result in enumerate(limited_results, 1):
            title = result.get('title', 'Untitled')
            content = result.get('abstract', result.get('content', 'No content available'))
            
            formatted_results.append(f"{i}. {title}\n{content}\n")
        
        results_text = "\n".join(formatted_results)
        
        system_message = self.prompts.get_research_summary_prompt(query, max_results)
        
        summary_generator = self.llm_provider.stream_completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": results_text}
            ]
        )
        
        # Collect the streamed response into a single string
        summary = "".join(summary_generator)
        
        return SearchSummaryResponse(
            query=query,
            results_processed=len(limited_results),
            total_results=len(search_results),
            summary=summary,
            model_used=self.llm_provider.get_model()
        )
    
    def analyze_paper_content(
        self, 
        paper_data: Dict[str, Any],
        analysis_focus: str = "comprehensive"
    ) -> PaperAnalysisResponse:
        """
        Analyze a single paper's content
        
        Args:
            paper_data: Paper data with title, abstract, content
            analysis_focus: Focus of analysis ('comprehensive', 'methodology', 'results')
        
        Returns:
            Detailed analysis of the paper
        """
        title = paper_data.get('title', 'Untitled')
        abstract = paper_data.get('abstract', '')
        content = paper_data.get('content', '')
        
        # Combine title, abstract, and content
        full_text = f"Title: {title}\n\nAbstract: {abstract}\n\nContent: {content}"
        
        analysis = self.llm_provider.analyzer.analyze_paper(
            paper_content=full_text,
            analysis_type=analysis_focus
        )
        
        # Extract keywords for the paper
        keywords = self.llm_provider.analyzer.extract_keywords(
            text=full_text,
            max_keywords=8,
            include_phrases=True
        )
        
        return PaperAnalysisResponse(
            analysis=analysis["analysis"],
            analysis_type="comprehensive",  # Cast to literal type
            paper_title=title,
            model_used=analysis["model_used"]
        )
    
    
    async def breakdown_user_question(
        self,
        user_question: str,
        num_subtopics: Optional[int] = None,
        include_explanation: Optional[bool] = None
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
        
        # Let LLM decide if parameters are None
        subtopic_instruction = f"exactly {num_subtopics} sub-topics" if num_subtopics else "an appropriate number of sub-topics (between 1-3 based on question complexity, no more than 3)"
        
        explanation_instruction = ""
        if include_explanation is True:
            explanation_instruction = """
            For each sub-topic, provide a brief one-line explanation in the format:
            Sub-topic | Brief explanation

            Example:
            Definition of machine learning | Understanding what ML is and its core principles
            """
        elif include_explanation is None:
            explanation_instruction = """
            If the question is complex or technical, provide brief explanations for each sub-topic in the format:
            Sub-topic | Brief explanation

            For simple questions, just list the sub-topics without explanations.
            """

        prompt = f"""Analyze this user question and break it down into focused sub-topics that would help answer it comprehensively:

            User Question: "{user_question}"

            Please provide:
            1. A clarified/refined version of the question
            2. {subtopic_instruction}
            3. Assess the complexity level (simple/intermediate/advanced)

            The sub-topics should:
            - Cover different aspects needed to fully answer the question
            - Progress logically (from basic to advanced concepts)
            - Be specific and actionable for research
            - Help structure a comprehensive answer
            - Avoid redundancy
            - Match the depth needed for the question's complexity

            {explanation_instruction}

            Format your response as:
            COMPLEXITY: [simple/intermediate/advanced]
            CLARIFIED: [refined question]
            SUBTOPICS:
            1. [subtopic 1]
            2. [subtopic 2]
            ...
            """

        system_message = """You are an expert at analyzing questions and breaking them down into clear, focused sub-topics. 
            Your goal is to help users understand complex questions by identifying the key components they need to explore.

            Guidelines:
            - Simple questions (e.g., "What is X?"): 1-2 subtopics, no explanations needed
            - Intermediate questions (e.g., "How does X work?"): 1-2 subtopics, brief explanations helpful
            - Advanced questions (e.g., "Compare X and Y in context Z"): 2-3 subtopics, detailed explanations needed

            Adapt the number and depth of subtopics based on the question's complexity. Though never exceed 3 sub-topics."""

        response = self.llm_provider.simple_prompt(
            prompt=prompt,
            system_message=system_message,
            temperature=0.3
        )
        
        # Parse the response
        lines = response.split('\n')
        clarified_question = ""
        subtopics = []
        explanations = []
        complexity = "intermediate"  # default
        
        parsing_subtopics = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('COMPLEXITY:'):
                complexity = line.replace('COMPLEXITY:', '').strip().lower()
            
            elif line.startswith('CLARIFIED:'):
                clarified_question = line.replace('CLARIFIED:', '').strip()
            
            elif 'SUBTOPICS:' in line.upper():
                parsing_subtopics = True
                continue
            
            # Parse subtopics
            if parsing_subtopics and line:
                # Remove numbering, bullets, etc.
                clean_line = line
                for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '•', '-', '*']:
                    if clean_line.startswith(prefix):
                        clean_line = clean_line[len(prefix):].strip()
                        break
                
                # Check for explanation (format: "subtopic | explanation")
                if '|' in clean_line:
                    parts = clean_line.split('|', 1)
                    subtopics.append(parts[0].strip())
                    explanations.append(parts[1].strip())
                elif clean_line:
                    subtopics.append(clean_line)
        
        # Validate complexity to match Literal type
        valid_complexity = complexity if complexity in ["simple", "intermediate", "advanced"] else "intermediate"
        
        return QuestionBreakdownResponse(
            original_question=user_question,
            clarified_question=clarified_question or user_question,
            subtopics=subtopics,
            num_subtopics=len(subtopics),
            complexity=valid_complexity,  # type: ignore
            explanations=explanations if explanations and len(explanations) == len(subtopics) else None,
            has_explanations=bool(explanations and len(explanations) == len(subtopics)),
            model_used=self.llm_provider.get_model()
        )
    
    def compare_papers(
        self, 
        papers: List[Dict[str, Any]],
        comparison_aspects: Optional[List[str]] = None
    ) -> PaperComparisonResponse:
        """
        Compare multiple papers and identify similarities/differences
        
        Args:
            papers: List of paper data dictionaries
            comparison_aspects: Specific aspects to compare
        
        Returns:
            Comparison analysis
        """
        if not comparison_aspects:
            comparison_aspects = ["methodology", "findings", "limitations", "future work"]
        
        # Format papers for comparison
        papers_text = []
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', f'Paper {i}')
            abstract = paper.get('abstract', 'No abstract available')
            papers_text.append(f"Paper {i}: {title}\nAbstract: {abstract}\n")
        
        combined_text = "\n".join(papers_text)
        aspects_text = ", ".join(comparison_aspects)
        
        system_message = f"""Compare the following papers focusing on: {aspects_text}
        
        Provide a structured comparison that identifies:
        1. Common themes and approaches
        2. Key differences in methodology or findings
        3. Complementary insights
        4. Gaps or contradictions
        5. Overall synthesis of the research area"""
        
        comparison = self.llm_provider.simple_prompt(
            prompt=combined_text,
            system_message=system_message,
            temperature=0.4
        )
        
        return PaperComparisonResponse(
            papers_compared=len(papers),
            comparison_aspects=comparison_aspects,
            comparison=comparison,
            model_used=self.llm_provider.get_model()
        )
    
    def suggest_related_topics(
        self, 
        current_topic: str, 
        search_results: Optional[List[Dict[str, Any]]] = None,
        num_suggestions: int = 5
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
            titles = [result.get('title', '') for result in search_results[:5]]
            context = f"\nBased on recent research: {', '.join(titles)}"
        
        prompt = f"""Suggest {num_suggestions} related research topics to: "{current_topic}"{context}
        
        The suggestions should be:
        - Related but distinct from the original topic
        - Currently relevant in academic research
        - Specific enough to be actionable
        - Covering different angles or approaches
        
        Return only the topic suggestions, one per line."""
        
        response = self.llm_provider.simple_prompt(
            prompt=prompt,
            system_message="You are a research strategist who identifies promising research directions.",
            temperature=0.8
        )
        
        # Parse suggestions
        suggestions = []
        for line in response.split('\n'):
            line = line.strip()
            if line:
                # Check for bullet points or numbering
                has_marker = any(line.startswith(marker) for marker in ['•', '-', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.'])
                
                if has_marker:
                    # Remove bullet points or numbering
                    clean_line = line.lstrip('•-').split('.', 1)[-1].strip()
                    if clean_line:
                        suggestions.append(clean_line)
                else:
                    suggestions.append(line)
        
        return RelatedTopicsResponse(
            current_topic=current_topic,
            suggestions=suggestions[:num_suggestions],
            num_suggestions=len(suggestions[:num_suggestions]),
            model_used=self.llm_provider.get_model()
        )
    
    def generate_citation_based_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        show_thought_process: bool = True
    ) -> CitationBasedResponse:
        """
        Generate a response with thought process and citations from papers
        
        Args:
            query: User's question
            context: Retrieved papers/documents
            show_thought_process: Whether to show reasoning steps
            
        Returns:
            Citation-based response with thought process
        """
        # Format context with citation info
        formatted_context = []
        citation_map = {}
        
        for i, doc in enumerate(context, 1):
            paper_id = f"paper_{i}"
            title = doc.get('title', 'Untitled')
            authors = doc.get('authors', [])
            year = doc.get('year', None)
            content = doc.get('abstract', doc.get('content', ''))
            
            # Handle authors in various formats (list of dicts, list of strings, or string)
            if isinstance(authors, list):
                if authors and isinstance(authors[0], dict):
                    # List of author objects
                    authors_str = ', '.join(a.get('name', str(a)) for a in authors)
                elif authors:
                    # List of strings
                    authors_str = ', '.join(str(a) for a in authors)
                else:
                    authors_str = 'Unknown'
            elif isinstance(authors, str):
                authors_str = authors if authors else 'Unknown'
            else:
                authors_str = 'Unknown'
            
            formatted_context.append(
                f"[{i}] {title}\n"
                f"Authors: {authors_str}\n"
                f"Year: {year if year else 'N/A'}\n"
                f"Content: {content}\n"
            )
            
            citation_map[paper_id] = {
                "title": title,
                "authors": authors,
                "year": year,
                "paper_id": paper_id
            }
        
        context_text = "\n".join(formatted_context)
        
        system_message = """You are a research assistant that provides answers based strictly on provided research papers.

CRITICAL INSTRUCTIONS:
1. Show your thought process step by step
2. For EACH thought step, cite the specific papers that support it using [1], [2], etc.
3. Include direct quotes when making specific claims
4. Only use information from the provided papers
5. If the papers don't contain enough information, explicitly state this

Format your response as:
THOUGHT PROCESS:
Step 1: [Your reasoning]
Citations: [1], [2]
Quote: "relevant quote from paper"

Step 2: [Your reasoning]
Citations: [3]
Quote: "relevant quote from paper"

FINAL ANSWER:
[Your synthesized answer with inline citations like [1][2]]
"""

        prompt = f"""Question: {query}

Available Research Papers:
{context_text}

Please provide a comprehensive answer following the format specified."""

        response = self.llm_provider.simple_prompt(
            prompt=prompt,
            system_message=system_message,
            temperature=0.3
        )
        
        # Parse the response
        thought_steps = []
        final_answer = ""
        all_citations = []
        
        # Split response into sections
        sections = response.split('FINAL ANSWER:')
        thought_section = sections[0].replace('THOUGHT PROCESS:', '').strip()
        final_answer = sections[1].strip() if len(sections) > 1 else response
        
        # Parse thought steps
        step_pattern = re.compile(r'Step (\d+):', re.IGNORECASE)
        steps_text = step_pattern.split(thought_section)
        
        for i in range(1, len(steps_text), 2):
            if i + 1 < len(steps_text):
                step_num = int(steps_text[i])
                step_content = steps_text[i + 1].strip()
                
                # Extract citations
                citation_refs = re.findall(r'\[(\d+)\]', step_content)
                step_citations = []
                
                for ref in citation_refs:
                    paper_id = f"paper_{ref}"
                    if paper_id in citation_map:
                        citation_info = citation_map[paper_id]
                        
                        # Extract quote if present
                        quote_match = re.search(r'Quote: ["\'](.+?)["\']', step_content, re.DOTALL)
                        quote = quote_match.group(1) if quote_match else None
                        
                        citation = Citation(
                            paper_id=citation_info["paper_id"],
                            page=None,
                            title=citation_info["title"],
                            authors=citation_info["authors"],
                            year=citation_info["year"],
                            quote=quote,
                            relevance=f"Supports step {step_num}"
                        )
                        step_citations.append(citation)
                        
                        # Add to all citations if not duplicate
                        if not any(c.paper_id == citation.paper_id for c in all_citations):
                            all_citations.append(citation)
                
                # Extract the main thought (remove citations and quotes)
                thought = re.sub(r'Citations?:.*?(?=\n|$)', '', step_content, flags=re.DOTALL)
                thought = re.sub(r'Quote:.*?(?=\n|$)', '', thought, flags=re.DOTALL)
                thought = thought.strip()
                
                thought_step = ThoughtStep(
                    step_number=step_num,
                    thought=thought,
                    citations=step_citations,
                    confidence=None
                )
                thought_steps.append(thought_step)
        
        return CitationBasedResponse(
            query=query,
            thought_process=thought_steps,
            final_answer=final_answer,
            all_citations=all_citations,
            sources_count=len(all_citations),
            model_used=self.llm_provider.get_model(),
            metadata={}
        )
    
    async def stream_citation_based_response(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """
        Stream a citation-based response with thought process
        
        Args:
            query: User's question
            context: Retrieved papers/documents
            
        Yields:
            Formatted chunks including thought steps and citations
        """
        # Format context with citation info
        formatted_context = []
        
        for i, doc in enumerate(context, 1):
            title = doc.get('title', 'Untitled')
            authors = doc.get('authors', [])
            year = doc.get('year', None)
            content = doc.get('abstract', doc.get('content', ''))
            
            # Handle authors in various formats (list of dicts, list of strings, or string)
            if isinstance(authors, list):
                if authors and isinstance(authors[0], dict):
                    # List of author objects
                    authors_str = ', '.join(a.get('name', str(a)) for a in authors)
                elif authors:
                    # List of strings
                    authors_str = ', '.join(str(a) for a in authors)
                else:
                    authors_str = 'Unknown'
            elif isinstance(authors, str):
                authors_str = authors if authors else 'Unknown'
            else:
                authors_str = 'Unknown'
            
            formatted_context.append(
                f"[{i}] {title}\n"
                f"Authors: {authors_str}\n"
                f"Year: {year if year else 'N/A'}\n"
                f"Content: {content}\n"
            )
        
        context_text = "\n".join(formatted_context)
        
        system_message = """
            You are a research assistant that provides answers based strictly on provided research papers.

            Stream your response in this format:
            1. Show reasoning steps prefixed with "💭 " 
            2. After each reasoning step, cite papers with "[1]", "[2]", etc.
            3. End with "📝 Final Answer:" followed by your synthesized answer
            4. Use inline citations in the final answer

            Example:
            💭 First, let's understand the core concept...
            [1] According to Smith et al., "machine learning is..."

            💭 Next, we need to consider the implementation...
            [2][3] Multiple studies show that...

            📝 Final Answer:
            Machine learning works by [1][2]...
            """

        prompt = f"""Question: {query}

        Available Research Papers:
        {context_text}

        Please provide a comprehensive answer with citations."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        print(f"[DEBUG] Starting to stream completion...")
        # Stream the response
        chunk_count = 0
        for chunk in self.llm_provider.stream_completion(messages=messages):
            chunk_count += 1
            if chunk_count % 10 == 0:
                print(f"[DEBUG] Streamed {chunk_count} chunks so far...")
            yield chunk
        print(f"[DEBUG] Finished streaming. Total chunks: {chunk_count}")

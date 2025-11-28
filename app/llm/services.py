"""
LLM Service that integrates with the retriever services
"""
from typing import List, Dict, Any, Optional, AsyncGenerator
from app.extensions.stream import get_simple_response_content
from app.llm import LLMProvider, ModelType, LiteLLMProvider
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
from app.extensions.logger import create_logger
import json
import re

logger = create_logger(__name__)


class LLMService():
    """Service class that provides LLM functionality for the application"""
    
    def __init__(self):
        # Determine which provider to use based on settings
        provider_type = getattr(settings, 'LLM_PROVIDER', 'openai').lower()
        
        # if provider_type == 'ollama':
        #     self.llm_provider = LLMProvider(
        #         provider='ollama'
        #     )
        # else:  # default to openai
        #     self.llm_provider = LLMProvider(
        #         api_key=settings.OPENAI_API_KEY,
        #         default_model=ModelType.GPT_4O_MINI.value,
        #         provider='openai'
        #     )
        self.llm_provider = LiteLLMProvider(
            model="gemini/gemini-2.5-pro",
            api_key=settings.GEMINI_API_KEY
        )
        self.prompts = SummaryPrompts
    
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
        
        # Generate exactly 1-2 focused subtopics that stay close to the user's query
        subtopic_instruction = f"exactly {num_subtopics} sub-topics" if num_subtopics else "1-2 focused sub-topics (no more than 2)"

        prompt = f"""Analyze this user question and create focused search queries that would help find relevant research papers:

            User Question: "{user_question}"

            Please provide:
            1. A clarified/refined version of the question that captures the core intent
            2. {subtopic_instruction} for searching academic papers

            IMPORTANT RULES FOR SUBTOPICS:
            - Stay VERY CLOSE to the original question's terminology and focus
            - Each subtopic should be a searchable query that directly relates to the user's question
            - Use the same technical terms, product names, and concepts from the original question
            - DO NOT expand into tangential areas or general background topics
            - Keep subtopics specific and targeted to what the user asked
            - Maximum 2 subtopics to maintain focus

            Examples:
            Question: "How does Walrus Sui decentralized storage work?"
            → Subtopic 1: "Walrus Sui decentralized storage architecture and mechanisms"
            → Subtopic 2: "Walrus Sui storage implementation and protocols"

            Question: "What are the benefits of transformer models?"
            → Subtopic 1: "transformer model advantages and performance benefits"
            → Subtopic 2: "transformer architecture computational efficiency"

            Question: "Explain quantum entanglement"
            → Subtopic 1: "quantum entanglement phenomena and mechanisms"

            Format your response as:
            COMPLEXITY: [simple/intermediate/advanced]
            CLARIFIED: [refined question]
            SUBTOPICS:
            1. [subtopic 1]
            2. [subtopic 2]
            """

        system_message = """You are an expert at converting user questions into effective academic search queries.

            Guidelines:
            - PRESERVE the exact terminology from the user's question (product names, technical terms, specific concepts)
            - Generate 1-2 focused subtopics maximum
            - Each subtopic should be a direct search query for finding relevant papers
            - Stay laser-focused on what the user specifically asked about
            - Avoid generic background topics or tangential areas
            - Simple "What is X?" questions: 1 subtopic focused on X
            - "How does X work?" questions: 1-2 subtopics about X's mechanisms/implementation
            - Comparison questions: 1-2 subtopics comparing the specific items mentioned

            NEVER exceed 2 subtopics. Quality over quantity."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_provider.simple_prompt(
            messages=messages,
            temperature=0.3
        )
        logger.info(f"LLM response for question breakdown: {response}")
        
        # Parse the response with robust handling for different formats
        lines = get_simple_response_content(response).split('\n')
        clarified_question = ""
        subtopics = []
        explanations = []
        complexity = "intermediate"  # default
        
        parsing_subtopics = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract complexity (handles both "COMPLEXITY: intermediate" and "intermediate")
            if line.startswith('COMPLEXITY:'):
                complexity_text = line.replace('COMPLEXITY:', '').strip().lower()
                # Extract first word if there's extra text
                complexity = complexity_text.split()[0] if complexity_text else "intermediate"
            
            # Extract clarified question (handles quotes and extra formatting)
            elif line.startswith('CLARIFIED:'):
                clarified_text = line.replace('CLARIFIED:', '').strip()
                # Remove quotes if present
                clarified_question = clarified_text.strip('"').strip()
            
            # Detect SUBTOPICS section (case-insensitive, handles variations)
            elif 'SUBTOPICS:' in line.upper() or 'SUBTOPIC' in line.upper():
                parsing_subtopics = True
                continue
            
            # Parse subtopics
            if parsing_subtopics and line:
                # Skip lines that are just headers or explanations
                if line.upper().startswith('THESE') or line.upper().startswith('THE SUB'):
                    continue
                
                # Remove various prefix formats
                clean_line = line
                
                # Remove numbered prefixes (1., 2., 3., etc.)
                if re.match(r'^\d+\.', clean_line):
                    clean_line = re.sub(r'^\d+\.\s*', '', clean_line)
                
                # Remove bullet points and asterisks
                for prefix in ['•', '-', '*', '>', '○', '▪']:
                    if clean_line.startswith(prefix):
                        clean_line = clean_line[len(prefix):].strip()
                        break
                
                # Remove markdown bold markers
                clean_line = clean_line.replace('**', '').strip()
                
                # Remove surrounding quotes (both single and double)
                clean_line = clean_line.strip('"').strip("'").strip()
                
                # Skip if line is too short or looks like a header
                if len(clean_line) < 5:
                    continue
                
                # Check for explanation (format: "subtopic | explanation" or "subtopic: explanation")
                if '|' in clean_line:
                    parts = clean_line.split('|', 1)
                    topic = parts[0].strip().strip('"').strip("'").strip()
                    explanation = parts[1].strip().strip('"').strip("'").strip()
                    subtopics.append(topic)
                    explanations.append(explanation)
                elif ':' in clean_line and not clean_line.startswith('COMPLEXITY:'):
                    # Handle "Subtopic: explanation" format
                    parts = clean_line.split(':', 1)
                    topic = parts[0].strip().strip('"').strip("'").strip()
                    subtopics.append(topic)
                    if len(parts) > 1:
                        explanation = parts[1].strip().strip('"').strip("'").strip()
                        explanations.append(explanation)
                elif clean_line:
                    # Plain subtopic without explanation
                    clean_line = clean_line.strip('"').strip("'").strip()
                    subtopics.append(clean_line)
        
        # Filter out empty subtopics and limit to maximum 2
        subtopics = [s.strip() for s in subtopics if s.strip() and len(s.strip()) > 5][:2]
        
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
        
        messages = [
            {"role": "system", "content": "You are a research strategist who identifies promising research directions."},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_provider.simple_prompt(
            messages=messages,
            temperature=0.8
        )
        
        # Parse suggestions
        suggestions = []
        for line in get_simple_response_content(response).split('\n'):
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
    
    async def stream_citation_based_response(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> AsyncGenerator[Any, None]:
        """
        Stream a citation-based response with thought process
        
        Args:
            query: User's question
            context: Retrieved papers/documents
            
        Yields:
            Formatted chunks including thought steps and citations
        """
        # Format context with citation info
        # Context now includes chunk_text from vector search for more relevant content
        formatted_context = []
        
        for i, doc in enumerate(context, 1):
            title = doc.get('title', 'Untitled')
            authors = doc.get('authors', [])
            year = doc.get('year', None)
            paper_id = doc.get('paper_id', doc.get('id', f'paper_{i}'))
            pdf_url = doc.get('pdf_url', '')
            
            
            # Prefer chunk_text (from vector search) over abstract
            # chunk_text is the most relevant section of the paper for the query
            chunk_text = doc.get('chunk_text')
            section = doc.get('section')
            
            if chunk_text:
                # Use the actual relevant chunk from the paper
                content = chunk_text
                if section:
                    content = f"[Section: {section}]\n{content}"
            else:
                # Fallback to abstract if no chunks available
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
            
            # Build formatted context with paper metadata
            context_entry = f"[{i}] {title}\n"
            context_entry += f"Paper ID: {paper_id}\n"
            context_entry += f"Authors: {authors_str}\n"
            context_entry += f"Year: {year if year else 'N/A'}\n"
            context_entry += f"PDF URL: {pdf_url}\n"
            context_entry += f"Content: {content}\n"
            
            formatted_context.append(context_entry)
        
        context_text = "\n".join(formatted_context)
        
        system_message = """You are an expert research assistant with deep knowledge across scientific domains. Your role is to synthesize information from academic papers into natural, engaging, and well-cited responses.

        CRITICAL GUIDELINES:
        1. Write in a natural, conversational academic style - avoid robotic formats or emoji markers
        2. Synthesize a consensus view from multiple papers when possible
        3. Use inline citations [1](paper_id1), [2](paper_id2), [3](paper_id3) to support every claim, fact, or finding
        4. Structure your response logically with clear sections when appropriate (use headers, lists, tables if helpful)
        5. Highlight key definitions, types, causes, mechanisms, or applications based on the question
        6. When papers disagree or show different perspectives, acknowledge this explicitly
        7. If the question is vague or could benefit from clarification, briefly ask what aspect the user is most interested in
        8. Base EVERY statement on the provided papers - do not add external knowledge
        9. If papers lack sufficient information, state this clearly and suggest what additional research might help
        10. If unsure about a claim, avoid making definitive statements.
        11. If only abstracts are provided, you may answer based on your own general knowledge, but you MUST clearly notify the user when doing so. Distinguish between claims supported by the abstract and those based on your own knowledge. For any information not present in the abstract, explicitly state: "This information is not available in the abstract; the following is based on general knowledge and may not be accurate for this specific paper."

        MARKDOWN FORMATTING (REQUIRED):
        - Use # for main headers, ## for subheaders, ### for sub-sections
        - Use **bold** for emphasis on key terms
        - Use bullet points with - or * for lists
        - Use numbered lists 1. 2. 3. when showing steps or rankings
        - Use > for important quotes or highlights
        - Use tables when comparing multiple items
        - Use code blocks with ``` when showing technical terms or formulas
        - Format your entire response in proper Markdown

        RESPONSE STYLE:
        - Start with a direct answer or overview (use a header if appropriate)
        - Organize information hierarchically with proper Markdown headers
        - Use evidence-based language such as: "Research shows that..." "Multiple studies indicate..." "According to [1][2]..."
        - Include direct quotes sparingly when they add significant value
        - End with a brief synthesis or conclusion when appropriate

        CITATION RULES (CRITICAL):
        - Cite papers INLINE ONLY using [1](paper_id1), [2](paper_id2), etc. where paper_id is the Paper ID provided
        - Multiple citations for the same claim: [1](paper_id1)[2](paper_id2)[5](paper_id5)
        - Always cite when stating facts, statistics, definitions, or findings
        - Group related citations together naturally in the text
        - DO NOT create a separate "References" section at the end
        - DO NOT list references separately - ALL citations must be inline only
        - Use the exact Paper ID from the context (e.g., if Paper ID is "arxiv_2301.12345", cite as [1](arxiv_2301.12345))
        
        FORBIDDEN:
        - DO NOT add a "References:" section at the end
        - DO NOT list citations separately from the text
        - DO NOT use plain text without Markdown formatting
        """

        prompt = f"""Question: {query}

        Available Research Papers:
        {context_text}

        Please provide a comprehensive, naturally-written answer that synthesizes information from these papers with proper inline citations."""
        
        print(f"[DEBUG] Starting to stream completion...")
        chunk_count = 0
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        for chunk in self.llm_provider.stream_completion(messages=messages):
            chunk_count += 1
            if chunk_count % 10 == 0:
                print(f"[DEBUG] Streamed {chunk_count} chunks so far...")
            yield chunk
        print(f"[DEBUG] Finished streaming. Total chunks: {chunk_count}")

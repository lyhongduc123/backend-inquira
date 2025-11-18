"""
LLM Service that integrates with the retriever services
"""
from typing import List, Dict, Any, Optional
from app.llm import LLMProvider, ModelType
from app.llm.prompts import SummaryPrompts
from app.core.config import settings


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
    ) -> Dict[str, Any]:
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
        
        summary = self.llm_provider.stream_completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": results_text}
            ]
        )
        
        return {
            "query": query,
            "results_processed": len(limited_results),
            "total_results": len(search_results),
            "summary": summary,
            "model_used": self.llm_provider.get_model()
        }
    
    def analyze_paper_content(
        self, 
        paper_data: Dict[str, Any],
        analysis_focus: str = "comprehensive"
    ) -> Dict[str, Any]:
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
        
        return {
            "title": title,
            "analysis": analysis["analysis"],
            "keywords": keywords,
            "analysis_type": analysis_focus,
            "model_used": analysis["model_used"]
        }
    
    def generate_research_questions(
        self, 
        topic: str, 
        context: Optional[str] = None,
        num_questions: int = 3
    ) -> List[str]:
        """
        Generate research questions for a given topic
        
        Args:
            topic: Research topic
            context: Additional context or background
            num_questions: Number of questions to generate
        
        Returns:
            List of research questions
        """
        context_text = f"\nContext: {context}" if context else ""

        prompt = f"""Generate {num_questions} insightful research questions or sentences for the topic: "{topic}"{context_text}

        The questions/sentences should be:
        - Specific and focused
        - Researchable and feasible
        - Relevant to current academic discourse
        - Varied in scope (some broad, some specific)
        
        Return only the questions, numbered 1-{num_questions}."""
        
        response = self.llm_provider.simple_prompt(
            prompt=prompt,
            system_message="You are an experienced research advisor who helps formulate compelling research questions.",
            temperature=0.7
        )
        
        # Parse the numbered questions
        questions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                # Remove numbering and clean up
                question = line.split('.', 1)[-1].strip()
                if question:
                    questions.append(question)
        
        return questions[:num_questions]
    
    def compare_papers(
        self, 
        papers: List[Dict[str, Any]],
        comparison_aspects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
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
        
        return {
            "papers_compared": len(papers),
            "comparison_aspects": comparison_aspects,
            "comparison": comparison,
            "model_used": self.llm_provider.get_model()
        }
    
    def suggest_related_topics(
        self, 
        current_topic: str, 
        search_results: Optional[List[Dict[str, Any]]] = None,
        num_suggestions: int = 5
    ) -> List[str]:
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
        
        return suggestions[:num_suggestions]

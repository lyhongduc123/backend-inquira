"""
Academic paper and content analysis module
"""
from typing import Optional, Dict, Any, List
from .base import BaseLLMClient
from .prompts import AnalysisPrompts


class Analyzer:
    """Specialized class for content analysis tasks"""
    
    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        self.prompts = AnalysisPrompts
    
    def analyze_paper(
        self,
        paper_content: str,
        analysis_type: str = "comprehensive",
        model: Optional[str] = None,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Analyze academic paper content
        
        Args:
            paper_content: Content of the paper
            analysis_type: Type of analysis ('comprehensive', 'methodology', 'results', 'literature_review')
            model: Model to use
            temperature: Randomness in response
        
        Returns:
            Analysis results as dictionary
        """
        system_message = self.prompts.get_analysis_prompt(analysis_type)
        
        analysis = self.llm.simple_prompt(
            prompt=paper_content,
            system_message=system_message,
            model=model,
            temperature=temperature
        )
        
        return {
            "analysis": analysis,
            "analysis_type": analysis_type,
            "model_used": model or self.llm.default_model
        }
    
    def extract_keywords(
        self,
        text: str,
        max_keywords: int = 10,
        include_phrases: bool = True,
        domain: Optional[str] = None,
        model: Optional[str] = None
    ) -> List[str]:
        """
        Extract keywords from text
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords
            include_phrases: Whether to include key phrases
            domain: Domain context (e.g., 'computer science', 'biology')
            model: Model to use
        
        Returns:
            List of keywords/phrases
        """
        system_message = self.prompts.get_keyword_extraction_prompt(
            max_keywords=max_keywords,
            include_phrases=include_phrases,
            domain=domain
        )
        
        keywords_text = self.llm.simple_prompt(
            prompt=text,
            system_message=system_message,
            model=model,
            temperature=0.1
        )
        
        # Parse the comma-separated keywords
        keywords = [kw.strip() for kw in keywords_text.split(',')]
        return keywords[:max_keywords]
    
    def compare_papers(
        self,
        papers: List[Dict[str, Any]],
        comparison_aspects: Optional[List[str]] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple papers across different aspects
        
        Args:
            papers: List of paper data dictionaries
            comparison_aspects: Specific aspects to compare
            model: Model to use
        
        Returns:
            Comparison analysis
        """
        if not comparison_aspects:
            comparison_aspects = ["methodology", "findings", "limitations", "contributions"]
        
        # Format papers for comparison
        papers_text = []
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', f'Paper {i}')
            abstract = paper.get('abstract', 'No abstract available')
            content = paper.get('content', '')
            papers_text.append(f"**Paper {i}: {title}**\nAbstract: {abstract}\nContent: {content[:500]}...\n")
        
        combined_text = "\n".join(papers_text)
        
        system_message = self.prompts.get_comparison_prompt(comparison_aspects)
        
        comparison = self.llm.simple_prompt(
            prompt=combined_text,
            system_message=system_message,
            model=model,
            temperature=0.4
        )
        
        return {
            "papers_compared": len(papers),
            "comparison_aspects": comparison_aspects,
            "comparison": comparison,
            "model_used": model or self.llm.default_model
        }
    
    def identify_research_gaps(
        self,
        papers: List[Dict[str, Any]],
        research_area: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Identify research gaps from a collection of papers
        
        Args:
            papers: List of paper data
            research_area: The research area being analyzed
            model: Model to use
        
        Returns:
            Identified research gaps and opportunities
        """
        # Combine abstracts and key content
        papers_text = []
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', f'Paper {i}')
            abstract = paper.get('abstract', '')
            papers_text.append(f"{i}. {title}: {abstract}")
        
        combined_text = "\n".join(papers_text)
        
        system_message = self.prompts.get_research_gaps_prompt(research_area)
        
        gaps_analysis = self.llm.simple_prompt(
            prompt=combined_text,
            system_message=system_message,
            model=model,
            temperature=0.5
        )
        
        return {
            "research_area": research_area,
            "papers_analyzed": len(papers),
            "gaps_analysis": gaps_analysis,
            "model_used": model or self.llm.default_model
        }
    
    def analyze_methodology(
        self,
        paper_content: str,
        focus_areas: Optional[List[str]] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deep analysis of research methodology
        
        Args:
            paper_content: Content of the paper
            focus_areas: Specific methodology aspects to analyze
            model: Model to use
        
        Returns:
            Methodology analysis
        """
        if not focus_areas:
            focus_areas = ["design", "data_collection", "analysis", "validity", "limitations"]
        
        system_message = self.prompts.get_methodology_deep_analysis_prompt(focus_areas)
        
        methodology_analysis = self.llm.simple_prompt(
            prompt=paper_content,
            system_message=system_message,
            model=model,
            temperature=0.3
        )
        
        return {
            "methodology_analysis": methodology_analysis,
            "focus_areas": focus_areas,
            "model_used": model or self.llm.default_model
        }
    
    def sentiment_analysis(
        self,
        text: str,
        granularity: str = "overall",
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze sentiment and tone of text
        
        Args:
            text: Text to analyze
            granularity: Level of analysis ('overall', 'detailed', 'aspect-based')
            model: Model to use
        
        Returns:
            Sentiment analysis results
        """
        system_message = self.prompts.get_sentiment_analysis_prompt(granularity)
        
        sentiment_result = self.llm.simple_prompt(
            prompt=text,
            system_message=system_message,
            model=model,
            temperature=0.2
        )
        
        return {
            "sentiment_analysis": sentiment_result,
            "granularity": granularity,
            "model_used": model or self.llm.default_model
        }

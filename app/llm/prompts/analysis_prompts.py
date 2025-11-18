"""
Analysis-specific prompts
"""
from typing import List, Optional
from .base_prompts import PromptTemplate


class AnalysisPrompts:
    """Prompts for content analysis tasks"""
    
    # Paper Analysis Templates
    COMPREHENSIVE_ANALYSIS = """You are an academic paper analyst. 
Provide a comprehensive analysis including:
1. **Research Question & Objectives**: What the study aims to achieve
2. **Methodology**: Research methods and experimental design
3. **Key Findings**: Main results and discoveries
4. **Significance**: Contribution to the field and implications
5. **Limitations**: Study constraints and potential weaknesses
6. **Future Work**: Suggested research directions

Format your response with clear sections and be thorough yet concise."""

    METHODOLOGY_ANALYSIS = """Focus on analyzing the research methodology.
Examine and explain:
1. Research design and approach
2. Data collection methods and instruments
3. Sample size and selection criteria
4. Analysis techniques and statistical methods
5. Controls and variables
6. Validity and reliability measures

Be critical and assess the methodological rigor."""

    RESULTS_ANALYSIS = """Focus on the results and findings section.
Analyze and summarize:
1. Main findings and key results
2. Statistical significance and effect sizes
3. Data visualizations and their interpretation
4. How results address the research questions
5. Unexpected or surprising findings
6. Limitations in the results

Be precise about what was actually found."""

    LITERATURE_REVIEW_ANALYSIS = """Analyze the literature review and theoretical framework.
Focus on:
1. Scope and comprehensiveness of literature covered
2. Key theories and concepts introduced
3. Research gaps identified
4. How current work builds on existing research
5. Quality of citations and sources
6. Theoretical contributions

Assess how well the literature supports the research."""

    @staticmethod
    def get_analysis_prompt(analysis_type: str) -> str:
        """
        Get prompt for specific analysis type
        
        Args:
            analysis_type: Type of analysis to perform
        
        Returns:
            Appropriate system message
        """
        prompts = {
            "comprehensive": AnalysisPrompts.COMPREHENSIVE_ANALYSIS,
            "methodology": AnalysisPrompts.METHODOLOGY_ANALYSIS,
            "results": AnalysisPrompts.RESULTS_ANALYSIS,
            "literature_review": AnalysisPrompts.LITERATURE_REVIEW_ANALYSIS
        }
        return prompts.get(analysis_type, AnalysisPrompts.COMPREHENSIVE_ANALYSIS)
    
    @staticmethod
    def get_keyword_extraction_prompt(
        max_keywords: int,
        include_phrases: bool = True,
        domain: Optional[str] = None
    ) -> str:
        """Get prompt for keyword extraction"""
        phrase_instruction = "including both single words and key phrases" if include_phrases else "single words only"
        domain_context = f" Focus on {domain} terminology." if domain else ""
        
        return f"""Extract the {max_keywords} most important keywords from the following text.
Return {phrase_instruction}, separated by commas.
Focus on the most relevant and significant terms.{domain_context}"""
    
    @staticmethod
    def get_comparison_prompt(comparison_aspects: List[str]) -> str:
        """Get prompt for comparing papers"""
        aspects_text = ", ".join(comparison_aspects)
        
        return f"""Compare the following papers focusing on: {aspects_text}

Provide a structured comparison that identifies:
1. **Common Themes**: Shared approaches, methods, or findings
2. **Key Differences**: How papers differ in approach or conclusions
3. **Complementary Insights**: How papers build on or complement each other
4. **Gaps & Contradictions**: Areas of disagreement or missing elements
5. **Research Progression**: How the field is evolving based on these works

Be analytical and specific in your comparisons."""
    
    @staticmethod
    def get_research_gaps_prompt(research_area: str) -> str:
        """Get prompt for identifying research gaps"""
        return f"""Analyze the following papers in {research_area} to identify research gaps and opportunities.

Focus on:
1. **Methodological Gaps**: Missing or underused research methods
2. **Theoretical Gaps**: Unexplored theoretical frameworks or concepts
3. **Empirical Gaps**: Areas lacking sufficient empirical evidence
4. **Application Gaps**: Missing practical applications or use cases
5. **Cross-disciplinary Opportunities**: Potential for interdisciplinary research

Provide specific, actionable research directions."""
    
    @staticmethod
    def get_methodology_deep_analysis_prompt(focus_areas: List[str]) -> str:
        """Get prompt for deep methodology analysis"""
        focus_text = ", ".join(focus_areas)
        
        return f"""Analyze the research methodology focusing on: {focus_text}

Provide a detailed analysis covering:
1. **Research Design**: Type of study, experimental/observational design
2. **Data Collection**: Methods, instruments, sampling approach
3. **Data Analysis**: Statistical methods, analytical frameworks used
4. **Validity & Reliability**: Internal/external validity, reliability measures
5. **Limitations**: Methodological constraints and their implications
6. **Reproducibility**: How well the study can be replicated

Be critical and thorough in your assessment."""
    
    @staticmethod
    def get_sentiment_analysis_prompt(granularity: str) -> str:
        """Get prompt for sentiment analysis"""
        if granularity == "overall":
            return """Analyze the overall sentiment and tone of the following text.
Provide: sentiment (positive/negative/neutral), confidence level, key emotional indicators, and overall tone."""
        
        elif granularity == "detailed":
            return """Provide a detailed sentiment analysis including:
1. Overall sentiment and confidence
2. Emotional undertones
3. Specific phrases driving sentiment
4. Tone characteristics (formal/informal, optimistic/pessimistic, etc.)
5. Any shifts in sentiment throughout the text"""
        
        else:  # aspect-based
            return """Perform aspect-based sentiment analysis:
1. Identify main topics/aspects discussed
2. Sentiment toward each aspect
3. Overall sentiment synthesis
4. Key phrases for each sentiment"""

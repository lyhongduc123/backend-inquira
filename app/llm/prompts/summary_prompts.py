"""
Summary-specific prompts
"""
from typing import List, Optional
from .base_prompts import PromptTemplate


class SummaryPrompts:
    """Prompts for text summarization tasks"""
    
    @staticmethod
    def get_summary_prompt(style: str, max_length: int) -> str:
        """
        Get prompt for specific summary style
        
        Args:
            style: Summary style
            max_length: Maximum length in words
        
        Returns:
            Formatted system message
        """
        base_prompt = f"You are a professional summarizer. Keep your summary under {max_length} words."
        
        style_prompts = {
            "concise": f"{base_prompt} Provide a concise, direct summary focusing on the main points.",
            
            "detailed": f"{base_prompt} Provide a detailed summary that captures nuances and important context while remaining clear and organized.",
            
            "bullet-points": f"{base_prompt} Summarize using bullet points for easy scanning. Use clear, actionable bullet points.",
            
            "executive": f"{base_prompt} Write an executive summary suitable for business leaders, focusing on key insights, implications, and actionable takeaways.",
            
            "academic": f"{base_prompt} Provide an academic-style summary that maintains scholarly tone and includes methodology, findings, and conclusions.",
            
            "narrative": f"{base_prompt} Create a narrative summary that tells the story or explains the progression of ideas in a flowing manner."
        }
        
        return style_prompts.get(style, style_prompts["concise"])
    
    @staticmethod
    def get_executive_summary_prompt(
        target_audience: str,
        key_points: Optional[List[str]] = None
    ) -> str:
        """Get prompt for executive summaries"""
        key_points_text = ""
        if key_points:
            key_points_text = f"\nEmphasize these key points: {', '.join(key_points)}"
        
        return f"""Create an executive summary for a {target_audience} audience.
Structure your response as:

**Overview**: Brief overview (2-3 sentences)
**Key Findings**: Main findings or conclusions (3-5 bullet points)
**Implications**: What this means and why it matters (2-3 sentences)
**Recommendations**: Actionable next steps (if applicable)

Keep it professional and accessible.{key_points_text}"""
    
    @staticmethod
    def get_summary_with_questions_prompt(num_questions: int) -> str:
        """Get prompt for summary with follow-up questions"""
        return f"""Summarize the following content and then generate {num_questions} 
thoughtful follow-up questions that would help someone understand the topic better.

Format your response as:
**Summary**: [Your summary here]

**Follow-up Questions**:
1. [Question 1]
2. [Question 2]
3. [Question 3]"""
    
    @staticmethod
    def get_research_summary_prompt(query: str, max_results: int) -> str:
        """Get prompt for summarizing research results"""
        return f"""You are a research assistant. Analyze the following search results 
for the query: "{query}"

Provide a structured summary that includes:
1. Key themes and patterns across the results
2. Most relevant findings related to the query
3. Important insights or conclusions
4. Gaps or areas for further research

Be concise but comprehensive."""

"""
Text summarization module using LLM
"""
from typing import Optional, Dict, Any, List
from .base import BaseLLMClient
from .prompts import SummaryPrompts

class Summarizer:
    """Specialized class for text summarization tasks"""
    
    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        self.prompts = SummaryPrompts
    
    def summarize_text(
        self,
        text: str,
        max_length: int = 200,
        style: str = "concise",
        model: Optional[str] = None,
        temperature: float = 0.3
    ) -> str:
        """
        Summarize given text
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary in words
            style: Summary style ('concise', 'detailed', 'bullet-points', 'executive')
            model: Model to use
            temperature: Randomness in response
        
        Returns:
            Summarized text
        """
        system_message = self.prompts.get_summary_prompt(style, max_length)
        
        return self.llm.simple_prompt(
            prompt=text,
            system_message=system_message,
            model=model,
            temperature=temperature
        )
    
    def summarize_multiple_texts(
        self,
        texts: List[str],
        max_length: int = 150,
        style: str = "concise",
        model: Optional[str] = None
    ) -> List[str]:
        """
        Summarize multiple texts
        
        Args:
            texts: List of texts to summarize
            max_length: Maximum length per summary
            style: Summary style
            model: Model to use
        
        Returns:
            List of summaries
        """
        summaries = []
        for text in texts:
            summary = self.summarize_text(
                text=text,
                max_length=max_length,
                style=style,
                model=model
            )
            summaries.append(summary)
        return summaries
    
    def stream_summarize(
        self,
        text: str,
        max_length: int = 200,
        style: str = "concise",
        model: Optional[str] = None,
        temperature: float = 0.3
    ):
        """
        Stream summarized text in chunks
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            style: Summary style
            model: Model to use
            temperature: Randomness in response
        
        Yields:
            Chunks of summarized text
        """
        system_message = self.prompts.get_summary_prompt(style, max_length)
        
        for chunk in self.llm.stream_completion(
            messages=[
                {"role": "system", "content": "You are a scientific assistant. Use the provided papers to answer questions, and always cite them."},
                {"role": "user", "content": text}
            ],
            system_message=system_message,
            model=model,
            temperature=temperature
        ):
            yield chunk
    
    def create_executive_summary(
        self,
        content: str,
        key_points: Optional[List[str]] = None,
        target_audience: str = "general",
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an executive summary with structured output
        
        Args:
            content: Content to summarize
            key_points: Specific points to emphasize
            target_audience: Target audience (general, technical, business)
            model: Model to use
        
        Returns:
            Structured executive summary
        """
        system_message = self.prompts.get_executive_summary_prompt(
            target_audience=target_audience,
            key_points=key_points
        )
        
        summary = self.llm.simple_prompt(
            prompt=content,
            system_message=system_message,
            model=model,
            temperature=0.3
        )
        
        return {
            "executive_summary": summary,
            "target_audience": target_audience,
            "key_points": key_points or [],
            "model_used": model or self.llm.default_model
        }
    
    def summarize_with_questions(
        self,
        content: str,
        num_questions: int = 3,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Summarize content and generate follow-up questions
        
        Args:
            content: Content to summarize
            num_questions: Number of follow-up questions to generate
            model: Model to use
        
        Returns:
            Summary with questions
        """
        system_message = self.prompts.get_summary_with_questions_prompt(num_questions)
        
        response = self.llm.simple_prompt(
            prompt=content,
            system_message=system_message,
            model=model,
            temperature=0.4
        )
        
        return {
            "content": response,
            "num_questions": num_questions,
            "model_used": model or self.llm.default_model
        }
    
    def progressive_summarization(
        self,
        long_text: str,
        chunk_size: int = 2000,
        final_length: int = 300,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Progressively summarize very long texts by chunking
        
        Args:
            long_text: Very long text to summarize
            chunk_size: Size of each chunk (in characters)
            final_length: Target length for final summary
            model: Model to use
        
        Returns:
            Progressive summary results
        """
        # Split text into chunks
        chunks = [long_text[i:i+chunk_size] for i in range(0, len(long_text), chunk_size)]
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            summary = self.summarize_text(
                text=chunk,
                max_length=200,
                style="detailed",
                model=model
            )
            chunk_summaries.append(summary)
        
        # Combine and final summarization
        combined_summaries = "\n\n".join(chunk_summaries)
        
        final_summary = self.summarize_text(
            text=combined_summaries,
            max_length=final_length,
            style="comprehensive",
            model=model
        )
        
        return {
            "final_summary": final_summary,
            "num_chunks": len(chunks),
            "chunk_summaries": chunk_summaries,
            "original_length": len(long_text),
            "final_length": len(final_summary),
            "compression_ratio": len(final_summary) / len(long_text),
            "model_used": model or self.llm.default_model
        }

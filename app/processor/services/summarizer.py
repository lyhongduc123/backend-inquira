from typing import Dict, Any, Optional, List
from app.models.papers import DBPaper
from app.retriever.paper_schemas import Paper
from app.extensions.logger import create_logger
from app.llm import llm_service
from app.extensions.stream import get_stream_response_content

logger = create_logger(__name__)

class SummarizerService:
    def __init__(self, chunker):
        self.chunker = chunker
    
    async def generate_summary(self, paper: Paper, full_text: str) -> str:
        """
        Generate a 300-800 token summary of the paper
        
        Args:
            paper: Paper Pydantic model
            full_text: Full text of the paper
            
        Returns:
            Summary text
        """
        # Truncate full text if too long (use first ~4000 tokens)
        tokens = self.chunker.count_tokens(full_text)
        if tokens > 4000:
            # Rough approximation: 1 token ≈ 4 characters
            truncated_text = full_text[:16000]
        else:
            truncated_text = full_text
        
        # Extract title and authors safely
        title_str = str(paper.title)
        abstract_val = getattr(paper, 'abstract', None)
        abstract_str = str(abstract_val) if abstract_val is not None else 'No abstract available'
        
        # Handle authors safely
        authors_str = 'Unknown'
        authors_val = getattr(paper, 'authors', None)
        if authors_val:
            try:
                if isinstance(authors_val, list):
                    author_names = [a.get('name', 'Unknown') if isinstance(a, dict) else str(a) for a in authors_val]
                    authors_str = ', '.join(author_names)
            except:
                authors_str = 'Unknown'
        
        prompt = f"""Generate a concise summary (300-800 tokens) of the following research paper.

                    Title: {title_str}
                    Authors: {authors_str}

                    Abstract:
                    {abstract_str}

                    Full Text (truncated if necessary):
                    {truncated_text}

                    Summary should include:
                    1. Main research question and objectives
                    2. Key methodology and approach
                    3. Primary findings and results
                    4. Significance and implications
                    5. Limitations (if mentioned)

                    Generate the summary:"""

        try:
            # Use simple_prompt for summary (no streaming needed)
            summary = llm_service.llm_provider.simple_prompt(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            summary = summary.strip() if summary else ""
            
            # Verify summary length (should be 300-800 tokens)
            summary_tokens = self.chunker.count_tokens(summary)
            paper_id_val = getattr(paper, 'paper_id', None)
            paper_id_str = str(paper_id_val) if paper_id_val is not None else 'Unknown'
            logger.info(f"Generated summary with {summary_tokens} tokens for paper {paper_id_str}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Title: {title_str}\n\nAbstract: {abstract_str}"
"""
Response builder service for chat operations.
Handles context building, query enhancement, and response formatting.
"""

from multiprocessing import context
from typing import Dict, Any, Tuple, List, Optional
from app.rag_pipeline.schemas import RAGResult
from app.extensions.logger import create_logger

logger = create_logger(__name__)


class ChatResponseBuilder:
    """Builds context and formats responses for chat interactions"""
    
    @staticmethod
    def build_context_from_results(
        results: RAGResult
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build LLM-friendly context from RAG results.
        
        Args:
            results: RAG pipeline results with papers and chunks
            
        Returns:
            Tuple of (formatted_context_string, paper_id_to_dbpaper_mapping)
        """
        chunk_papers = {}
        
        # Build paper mapping from chunks
        for chunk in results.chunks:
            chunk_paper_id = str(chunk.paper_id)
            if chunk_paper_id not in chunk_papers:
                ranked_paper = next(
                    (
                        rp
                        for rp in results.papers
                        if str(rp.paper.paper_id) == chunk_paper_id
                    ),
                    None,
                )
                if ranked_paper:
                    chunk_papers[chunk_paper_id] = ranked_paper.paper
        
        # Build legend (paper reference list)
        legend_entries = []
        seen_ids = set()
        
        for rp in results.papers:
            p = rp.paper
            p_id = str(p.paper_id)
            if p_id not in seen_ids:
                year = p.publication_date.year if p.publication_date else "N/A"
                legend_entries.append(f"PAPER ID: {p_id} | Title: {p.title} ({year}) | DOI: {p.external_ids.get('doi', 'N/A')} | APA: {p.citation_styles.get('apa', 'N/A')}")
                seen_ids.add(p_id)
        
        # Build content entries (actual chunks)
        content_entries = []
        for chunk in results.chunks:
            content_entries.append(
                f"SOURCE_ID: {chunk.paper_id}\n"
                f"CHUNK_ID: {chunk.chunk_id}\n"
                f"SECTION: {chunk.section_title or 'Main text'}\n"
                f"CONTENT: {chunk.text}"
            )
        
        # Combine into final context
        context = "--- REFERENCE LEGEND ---\n" + "\n".join(legend_entries)
        context += "\n\n--- RESEARCH CONTENT ---\n" + "\n\n".join(content_entries)
        
        return context, chunk_papers
    
    @staticmethod
    def build_enhanced_query(
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context_string: Optional[str] = None
    ) -> str:
        """
        Enhance query with conversation history context.
        
        Args:
            query: Current user query
            conversation_history: Previous conversation messages
            
        Returns:
            Enhanced query string with history context
        """
        new_query = f"Current Question: {query}\n"
        if not conversation_history:
            return new_query
        
        # Format history for system context
        history_text = "\n".join(
            [
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in conversation_history
            ]
        )
        enhanced_query = f"""
<conversation_history>
Previous conversation:
{history_text}
</conversation_history>

<documents>
{context_string or "No retrieved documents."}
</documents>

{new_query}
"""
        
        return enhanced_query
    
    @staticmethod
    def extract_metadata_from_results(results: RAGResult) -> List[Dict[str, Any]]:
        """
        Extract paper metadata for client caching.
        
        Args:
            results: RAG pipeline results
            
        Returns:
            List of paper metadata dictionaries
        """
        from app.domain.papers.schemas import PaperMetadata
        papers_metadata = [
            PaperMetadata.from_ranked_paper(p) for p in results.papers
        ]
        return [p.model_dump(mode="json") for p in papers_metadata]
    
    @staticmethod
    def get_retrieved_paper_ids(results: RAGResult) -> List[str]:
        """Extract paper IDs from results."""
        return [str(p.paper_id) for p in results.papers]

    @staticmethod
    def extract_context_chunks_from_results(results: RAGResult) -> List[Dict[str, Any]]:
        """Extract exact chunk payload that is passed to context building/LLM."""
        context_chunks: List[Dict[str, Any]] = []

        for chunk in results.chunks:
            context_chunks.append(
                {
                    "paper_id": str(chunk.paper_id),
                    "chunk_id": str(chunk.chunk_id),
                    "section": chunk.section_title or "Main text",
                    "content": chunk.text,
                }
            )

        return context_chunks

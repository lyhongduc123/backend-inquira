"""
Response builder service for chat operations.
Handles context building, query enhancement, and response formatting.
"""

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
        from app.domain.papers.schemas import PaperMetadata  
        legend_entries = []
        
        paper_id_to_idx = {}
        for idx, rp in enumerate(results.papers):
            metadata = PaperMetadata.from_ranked_paper(rp)
            paper_id_to_idx[str(rp.paper_id)] = idx
            year = metadata.year if metadata.year is not None else "N/A"
            doi = (metadata.external_ids or {}).get("doi", "N/A")
            apa = (metadata.citation_styles or {}).get("apa", "N/A")
            
            legend_entries.append(
                f"PAPER ID: {idx} | Title: {metadata.title} ({year}) | DOI: {doi} | APA: {apa}"
            )
            idx += 1
            
        # Build content entries (actual chunks)
        content_entries = []
        for chunk in results.chunks:
            source_id = paper_id_to_idx.get(str(chunk.paper_id), "UNKNOWN")
            
            content_entries.append(
                f"SOURCE_ID: {source_id}\n"
                f"CHUNK_ID: {chunk.chunk_id}\n"
                f"SECTION: {chunk.section_title or 'Main text'}\n"
                f"CONTENT: {chunk.text}"
            )
        
        # Combine into final context
        context = "--- REFERENCE LEGEND ---\n" + "\n".join(legend_entries)
        context += "\n\n--- RESEARCH CONTENT ---\n" + "\n\n".join(content_entries)
        
        return context, paper_id_to_idx
    
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

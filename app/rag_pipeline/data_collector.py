"""
RAG Pipeline Data Collector

Collects and logs all intermediate data from RAG pipeline executions for
inspection, debugging, and evaluation purposes.

Captures:
- Query decomposition and intent
- Retrieved papers (with scores)
- Chunks (with relevance scores)
- Ranked papers (with ranking details)
- Filters applied
- Context building steps
- Timing information
- Pipeline configuration
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

from app.domain.chunks.schemas import ChunkRetrieved
from app.processor.schemas import RankedPaper
from app.models.papers import DBPaper
from app.llm.schemas import QuestionBreakdownResponse, QueryIntent
from app.extensions.logger import create_logger

logger = create_logger(__name__)


@dataclass
class PipelineExecutionData:
    """Complete data from a single pipeline execution."""
    
    # Request metadata
    execution_id: str
    timestamp: str
    pipeline_type: str  # "database", "hybrid", "standard"
    
    # Query information
    original_query: str
    decomposed_queries: List[str] = field(default_factory=list)
    intent: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    
    # Retrieved data
    retrieved_papers: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Ranking information
    ranked_papers: List[Dict[str, Any]] = field(default_factory=list)
    ranking_weights: Optional[Dict[str, float]] = None
    
    # Context information
    conversation_id: Optional[str] = None
    conversation_history_length: int = 0
    
    # Pipeline configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Timing information
    decomposition_time_ms: Optional[float] = None
    search_time_ms: Optional[float] = None
    ranking_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None
    
    # Errors
    errors: List[str] = field(default_factory=list)


class RAGDataCollector:
    """
    Collects comprehensive data from RAG pipeline executions.
    
    Usage:
        collector = RAGDataCollector(enabled=True)
        
        # Start collection
        collector.start_execution(query="What is quantum computing?", pipeline_type="database")
        
        # Record data at each step
        collector.record_decomposition(queries=["query1", "query2"], intent="comprehensive")
        collector.record_papers(papers, scores)
        collector.record_chunks(chunks)
        collector.record_ranking(ranked_papers, weights)
        
        # Finish and save
        collector.end_execution()
    """
    
    def __init__(
        self, 
        enabled: bool = True,
        output_dir: str = "app/rag_pipeline/logs",
        save_on_end: bool = True
    ):
        """
        Initialize the data collector.
        
        Args:
            enabled: Whether to actually collect data (can disable in production)
            output_dir: Directory to save collected data
            save_on_end: Whether to auto-save when execution ends
        """
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.save_on_end = save_on_end
        self.current_execution: Optional[PipelineExecutionData] = None
        self.start_time: Optional[float] = None
        self.step_times: Dict[str, float] = {}
        
        # Create output directory if it doesn't exist
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def start_execution(
        self, 
        query: str, 
        pipeline_type: str = "database",
        conversation_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new execution collection.
        
        Returns:
            execution_id: Unique identifier for this execution
        """
        if not self.enabled:
            return ""
        
        import time
        self.start_time = time.time() * 1000  # milliseconds
        
        execution_id = f"{pipeline_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self.current_execution = PipelineExecutionData(
            execution_id=execution_id,
            timestamp=datetime.now().isoformat(),
            pipeline_type=pipeline_type,
            original_query=query,
            conversation_id=conversation_id,
            filters=filters or {},
            config=config or {}
        )
        
        logger.info(f"Started data collection for execution: {execution_id}")
        return execution_id
    
    def _mark_time(self, step_name: str):
        """Mark the time for a specific step."""
        if not self.enabled or not self.start_time:
            return
        
        import time
        self.step_times[step_name] = time.time() * 1000
    
    def record_decomposition(
        self, 
        queries: List[str],
        intent: Optional[QueryIntent] = None,
    ):
        """Record query decomposition results."""
        if not self.enabled or not self.current_execution:
            return
        
        self._mark_time("decomposition")
        
        self.current_execution.decomposed_queries = queries
        self.current_execution.intent = intent.value if intent else None
        
        if self.start_time:
            self.current_execution.decomposition_time_ms = self.step_times["decomposition"] - self.start_time
    
    def record_papers(
        self, 
        papers: List[DBPaper], 
        scores: Optional[Dict[str, float]] = None,
        papers_with_scores: Optional[List[tuple]] = None
    ):
        """Record retrieved papers with their search scores."""
        if not self.enabled or not self.current_execution:
            return
        
        self._mark_time("search")
        
        # Handle both formats: list of papers + dict of scores, or list of tuples
        if papers_with_scores:
            retrieved = [
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "year": paper.year,
                    "citation_count": paper.citation_count,
                    "venue": paper.venue,
                    "authors": [],  # Authors not loaded in DBPaper by default
                    "search_score": float(score) if score else None,
                }
                for paper, score in papers_with_scores
            ]
        else:
            retrieved = [
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "year": paper.year,
                    "citation_count": paper.citation_count,
                    "venue": paper.venue,
                    "authors": [],  # Authors not loaded in DBPaper by default
                    "search_score": float(scores.get(paper.paper_id, 0)) if scores and paper.paper_id in scores else None,
                }
                for paper in papers
            ]
        
        self.current_execution.retrieved_papers = retrieved
        
        if self.start_time and "decomposition" in self.step_times:
            self.current_execution.search_time_ms = self.step_times["search"] - self.step_times.get("decomposition", self.start_time)
    
    def record_chunks(self, chunks: List[ChunkRetrieved]):
        """Record retrieved chunks with relevance scores."""
        if not self.enabled or not self.current_execution:
            return
        
        self.current_execution.retrieved_chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,  # Truncate for readability
                "full_text": chunk.text,
                "token_count": chunk.token_count,
                "chunk_index": chunk.chunk_index,
                "section_title": chunk.section_title,
                "relevance_score": float(chunk.relevance_score) if chunk.relevance_score else None,
            }
            for chunk in chunks
        ]
    
    def record_ranking(
        self, 
        ranked_papers: List[RankedPaper],
        weights: Optional[Dict[str, float]] = None
    ):
        """Record ranked papers with ranking scores."""
        if not self.enabled or not self.current_execution:
            return
        
        self._mark_time("ranking")
        
        self.current_execution.ranked_papers = [
            {
                "paper_id": rp.paper_id,
                "title": rp.paper.title if rp.paper else None,
                "year": rp.paper.year if rp.paper else None,
                "citation_count": rp.paper.citation_count if rp.paper else None,
                "relevance_score": float(rp.relevance_score) if rp.relevance_score else None,
                "ranking_scores": {k: float(v) for k, v in rp.ranking_scores.items()} if rp.ranking_scores else {},
                "rank": ranked_papers.index(rp) + 1,
            }
            for rp in ranked_papers
        ]
        
        self.current_execution.ranking_weights = weights
        
        if self.start_time and "search" in self.step_times:
            self.current_execution.ranking_time_ms = self.step_times["ranking"] - self.step_times.get("search", self.start_time)
    
    def record_conversation_context(self, history_length: int):
        """Record conversation context information."""
        if not self.enabled or not self.current_execution:
            return
        
        self.current_execution.conversation_history_length = history_length
    
    def record_error(self, error: str):
        """Record an error that occurred during execution."""
        if not self.enabled or not self.current_execution:
            return
        
        self.current_execution.errors.append(error)
        logger.error(f"Recorded error in pipeline execution: {error}")
    
    def end_execution(self) -> Optional[str]:
        """
        End the current execution and save data.
        
        Returns:
            filepath: Path to saved file, or None if not saved
        """
        if not self.enabled or not self.current_execution:
            return None
        
        import time
        if self.start_time:
            self.current_execution.total_time_ms = (time.time() * 1000) - self.start_time
        
        filepath = None
        if self.save_on_end:
            filepath = self.save()
        
        logger.info(f"Ended data collection for execution: {self.current_execution.execution_id}")
        
        # Clear current execution
        self.current_execution = None
        self.start_time = None
        self.step_times.clear()
        
        return filepath
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the current execution data to a JSON file.
        
        Args:
            filepath: Optional custom filepath. If not provided, auto-generates one.
        
        Returns:
            filepath: Path to the saved file
        """
        if not self.enabled or not self.current_execution:
            return ""
        
        if filepath is None:
            filename = f"{self.current_execution.execution_id}.json"
            filepath = str(self.output_dir / filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.current_execution), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved pipeline execution data to: {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"Failed to save pipeline execution data: {e}")
            return ""
    
    def get_current_data(self) -> Optional[Dict[str, Any]]:
        """Get the current execution data as a dictionary."""
        if not self.enabled or not self.current_execution:
            return None
        
        return asdict(self.current_execution)


# Global singleton instance (can be configured via environment variable)
_global_collector: Optional[RAGDataCollector] = None


def get_data_collector(enabled: Optional[bool] = None) -> RAGDataCollector:
    """
    Get or create the global data collector instance.
    
    Args:
        enabled: Override the default enabled state. If None, checks environment variable.
    
    Returns:
        RAGDataCollector instance
    """
    global _global_collector
    
    if _global_collector is None:
        # Check environment variable for default state
        if enabled is None:
            enabled = os.getenv("RAG_DATA_COLLECTION_ENABLED", "false").lower() == "true"
        
        _global_collector = RAGDataCollector(enabled=enabled)
    
    return _global_collector


def enable_data_collection():
    """Enable data collection globally."""
    collector = get_data_collector()
    collector.enabled = True
    logger.info("RAG data collection enabled")


def disable_data_collection():
    """Disable data collection globally."""
    collector = get_data_collector()
    collector.enabled = False
    logger.info("RAG data collection disabled")

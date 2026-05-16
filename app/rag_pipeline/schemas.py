from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from app.llm.schemas.chat import QueryIntent
from app.rag_pipeline.config import PipelineConfig
from app.rag_pipeline.types import (
    PipelineResult,
    RAGPipelineContext,
    RAGPipelineEvent,
    RAGResult,
)


class RAGEventType:
    RESULT = "result"
    RANKING = "ranking"
    SEARCHING = "search_queries"
    PROCESSING = "processing"


class SearchWorkflowConfig(BaseModel):
    query: str
    search_queries: List[str] = Field(default_factory=list, description="Hybrid retrieval queries (each query is used for BM25 and semantic search)")
    intent: Optional[QueryIntent] = None
    top_papers: int = Field(
        default=PipelineConfig.TOP_PAPERS_LIMIT, description="Papers to fetch from DB"
    )
    top_chunks: int = Field(
        default=PipelineConfig.TOP_CHUNKS_LIMIT, description="Chunks to keep after RRF"
    )
    relevance_threshold: float = 0.3

    enable_reranking: bool = True
    enable_paper_ranking: bool = True

    # Context
    filters: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None
    
class AgentWorkflowConfig(BaseModel):
    query: str
    search_queries: List[str] = Field(default_factory=list, description="Hybrid retrieval queries (each query is used for BM25 and semantic search)")
    intent: Optional[QueryIntent] = None
    top_papers: int = Field(
        default=PipelineConfig.TOP_PAPERS_LIMIT, description="Papers to fetch from DB"
    )
    top_chunks: int = Field(
        default=PipelineConfig.TOP_CHUNKS_LIMIT, description="Chunks to keep after RRF"
    )
    enable_reranking: bool = True
    enable_paper_ranking: bool = True

    relevance_threshold: float = 0.3

     # Context
    filters: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None


__all__ = [
    "AgentWorkflowConfig",
    "PipelineResult",
    "RAGEventType",
    "RAGPipelineContext",
    "RAGPipelineEvent",
    "RAGResult",
    "SearchWorkflowConfig",
]

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from app.core.dtos import PaperEnrichedDTO
from app.domain.chunks.schemas import ChunkRetrieved
from app.llm.schemas.chat import QueryIntent
from app.models.papers import DBPaper
from app.processor.schemas import RankedPaper
from app.retriever.schemas.openalex import OAAuthorResponse
from app.llm.schemas import QuestionBreakdownResponse

class RAGEventType:
    RESULT = "result"
    RANKING = "ranking"
    SEARCHING = "search_queries"
    PROCESSING = "processing"
    
class SearchWorkflowConfig(BaseModel):
    # Required core inputs
    query: str
    bm25_query: Optional[str] = None
    semantic_queries: Optional[List[str]] = None
    intent: Optional[QueryIntent] = None
    
    # Tuning parameters (with defaults)
    top_papers: int = Field(default=20, description="Papers to fetch from DB")
    top_chunks: int = Field(default=20, description="Chunks to keep after RRF")
    relevance_threshold: float = 0.3
    
    # Feature flags
    enable_reranking: bool = True
    enable_paper_ranking: bool = True
    
    # Context
    filters: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None

@dataclass
class RAGResult:
    """
    RAG pipeline result containing ranked papers and relevant chunks.
    
    Papers are RankedPaper instances with scores attached after ranking.
    """
    papers: List[RankedPaper]
    chunks: List[ChunkRetrieved]

@dataclass
class RAGPipelineEvent:
    type: str
    data: dict | str | RAGResult | None
    
@dataclass
class RAGPipelineContext:
    query: str
    search_queries: List[str] = field(default_factory=list)
    papers: List[PaperEnrichedDTO] = field(default_factory=list)
    filtered_papers: List[PaperEnrichedDTO] = field(default_factory=list)
    papers_with_hybrid_scores: List[tuple] = field(default_factory=list)  
    processed_paper_ids: List[str] = field(default_factory=list)
    result_papers: List[RankedPaper] = field(default_factory=list) 
    chunks: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class PipelineResult:
    author: Optional[OAAuthorResponse] = None
    papers: List[PaperEnrichedDTO] = field(default_factory=list)
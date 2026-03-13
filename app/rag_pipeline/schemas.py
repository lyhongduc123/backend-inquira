from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from app.core.dtos import PaperEnrichedDTO
from app.domain.chunks.schemas import ChunkRetrieved
from app.models.papers import DBPaper
from app.processor.schemas import RankedPaper
from app.retriever.schemas.openalex import OAAuthorResponse
from app.llm.schemas import QuestionBreakdownResponse

class RAGEventType:
    RESULT = "result"
    RANKING = "ranking"
    SEARCHING = "search_queries"
    PROCESSING = "processing"

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
    papers_with_hybrid_scores: List[tuple] = field(default_factory=list)  # New: (DBPaper, score)
    processed_paper_ids: List[str] = field(default_factory=list)
    result_papers: List[RankedPaper] = field(default_factory=list)  # Changed from DBPaper
    chunks: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    breakdown_response: QuestionBreakdownResponse | None = None
    
@dataclass
class PipelineResult:
    author: Optional[OAAuthorResponse] = None
    papers: List[PaperEnrichedDTO] = field(default_factory=list)
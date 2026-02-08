from dataclasses import dataclass
from typing import List
from app.models.papers import DBPaper
from app.models.papers import DBPaperChunk
from app.chunks.schemas import ChunkRetrieved
from app.retriever.paper_schemas import Paper

@dataclass
class RAGResult:
    papers: List[Paper]
    chunks: List[ChunkRetrieved]

@dataclass
class RAGPipelineEvent:
    type: str
    data: dict | str | RAGResult | None
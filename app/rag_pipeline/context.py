from dataclasses import dataclass, field
from app.models.papers import DBPaper, DBPaperChunk
from typing import Dict, List, Any


@dataclass
class RAGContext:
    """
    Represents the final context that will be supplied to the LLM.
    Contains paper metadata, relevant chunks, and fallback abstracts when necessary.
    """

    papers: List[DBPaper]
    chunks: List[DBPaperChunk]
    used_fallback: bool = False
        
    blocks: List[Dict[str, Any]] = field(default_factory=list)

    def build(self) -> "RAGContext":
        """
        Build context blocks that the LLM will use.
        Each block includes: title, authors, year, citationCount, text (chunk/abstract), section.
        """

        if self.chunks:
            self.blocks = self._build_chunk_blocks()
        else:
            self.blocks = self._build_fallback_blocks()
            self.used_fallback = True

        return self
        
    def _build_chunk_blocks(self) -> List[Dict]:
        paper_map = {str(p.paper_id): p for p in self.papers}

        blocks = []
        for chunk in self.chunks:
            paper = paper_map.get(str(chunk.paper_id))
            if not paper:
                continue

            blocks.append({
                "paper_id": str(paper.paper_id),
                "title": paper.title,
                "authors": paper.authors or [],
                "year": paper.publication_date,
                "abstract": paper.abstract,
                "url": paper.url,
                "pdf_url": paper.pdf_url,
                "citationCount": paper.citation_count or 0,
                "section": chunk.section_title or "Unknown Section",
                "chunk_text": chunk.text
            })

        return blocks

    def _build_fallback_blocks(self) -> List[Dict]:
        blocks = []
        for paper in self.papers[:5]:  # limit to top 5
            blocks.append({
                "paper_id": str(paper.paper_id),
                "title": paper.title,
                "authors": paper.authors or [],
                "year": paper.publication_date,
                "abstract": paper.abstract,
                "url": paper.url,
                "pdf_url": paper.pdf_url,
                "citationCount": paper.citation_count or 0,
                "content": paper.abstract,
                "section": "Abstract (fallback)"
            })
        return blocks

    def to_llm_context(self) -> List[Dict[str, Any]]:
        """LLM expects only blocks."""
        return self.blocks

    def to_sources(self) -> List[Dict]:
        """Frontend expects full paper metadata."""
        from app.chat.services import db_paper_to_dict
        return [db_paper_to_dict(p) for p in self.papers]
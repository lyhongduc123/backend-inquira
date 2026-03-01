from dataclasses import dataclass, field
from app.models.papers import DBPaper, DBPaperChunk
from app.processor.schemas import RankedPaper
from typing import Dict, List, Any, Union


@dataclass
class RAGContext:
    """
    Represents the final context that will be supplied to the LLM.
    Contains paper metadata, relevant chunks, and fallback abstracts when necessary.
    """

    papers: List[Union[DBPaper, RankedPaper]]  # Can accept both types
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

            # Extract authors - works for both DBPaper and RankedPaper
            authors = []
            if hasattr(paper, 'authors'):
                authors = paper.authors if isinstance(paper.authors, list) else []
            elif hasattr(paper, 'paper_authors') and paper.paper_authors:
                authors = [{
                    'name': pa.author.name,
                    'author_id': pa.author.author_id
                } for pa in paper.paper_authors if pa.author]

            blocks.append({
                "paper_id": str(paper.paper_id),
                "title": paper.title,
                "authors": authors,
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
            # Extract authors - works for both DBPaper and RankedPaper
            authors = []
            if hasattr(paper, 'authors'):
                authors = paper.authors if isinstance(paper.authors, list) else []
            elif hasattr(paper, 'paper_authors') and paper.paper_authors:
                authors = [{
                    'name': pa.author.name,
                    'author_id': pa.author.author_id
                } for pa in paper.paper_authors if pa.author]
            
            blocks.append({
                "paper_id": str(paper.paper_id),
                "title": paper.title,
                "authors": authors,
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
        sources = []
        for p in self.papers:
            # RankedPaper has dict() method, DBPaper needs conversion
            if hasattr(p, 'dict'):
                sources.append(p.dict())
            else:
                # DBPaper - convert to dict manually
                sources.append({
                    'paper_id': p.paper_id,
                    'title': p.title,
                    'abstract': p.abstract,
                    'url': p.url,
                    'citation_count': p.citation_count,
                    # Add other fields as needed
                })
        return sources
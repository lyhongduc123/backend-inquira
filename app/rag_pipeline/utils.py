from typing import List, Any
from app.models.papers import DBPaper
from app.retriever.paper_schemas import Paper

def deduplicate_papers(papers: List[Paper]) -> List[Paper]:
    """Deduplicate papers based on their paper_id."""
    seen_ids = set()
    unique_papers = []
    for paper in papers:
        if paper.paper_id not in seen_ids:
            seen_ids.add(paper.paper_id)
            unique_papers.append(paper)
    return unique_papers
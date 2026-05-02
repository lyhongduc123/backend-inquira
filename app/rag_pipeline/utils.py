from typing import List, Any, Dict, Tuple
from collections import Counter, defaultdict
from app.models.papers import DBPaper
from app.core.dtos import PaperDTO
from app.retriever.schemas import NormalizedPaperResult

def deduplicate_papers(papers: List[PaperDTO]) -> List[PaperDTO]:
    """Deduplicate papers based on their paper_id."""
    seen_ids = set()
    unique_papers = []
    for paper in papers:
        if paper.paper_id not in seen_ids:
            seen_ids.add(paper.paper_id)
            unique_papers.append(paper)
    return unique_papers


def deduplicate_papers_with_rrf(
    paper_rankings: List[List[PaperDTO]], 
    k: int = 60
) -> List[PaperDTO]:
    """
    Deduplicate papers using Reciprocal Rank Fusion (RRF).
    
    RRF combines multiple rankings by computing:
    RRF(d) = sum over all rankings r of 1 / (k + rank(d, r))
    
    Args:
        paper_rankings: List of paper lists, where each list is a ranking from one query
        k: RRF constant (default: 60, standard in literature)
    
    Returns:
        Deduplicated list of papers sorted by RRF score (descending)
    """
    rrf_scores = defaultdict(float)
    paper_map = {} 
    
    # Calculate RRF scores
    for ranking in paper_rankings:
        for rank, paper in enumerate(ranking, start=1):
            paper_id = paper.paper_id
            rrf_scores[paper_id] += 1.0 / (k + rank)
            if paper_id not in paper_map:
                paper_map[paper_id] = paper
    
    sorted_paper_ids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)
    
    return [paper_map[pid] for pid in sorted_paper_ids]


def count_and_rank_references(
    papers: List[NormalizedPaperResult], 
    top_k: int = 5
) -> List[Tuple[str, int, str]]:
    """
    Count references across all papers and return top K most referenced papers.
    
    Args:
        papers: List of normalized paper results with references
        top_k: Number of top referenced papers to return (default: 5)
    
    Returns:
        List of tuples (paper_id, count, title) for top K most referenced papers
        Sorted by count in descending order
    """
    reference_counter = Counter()
    reference_titles: Dict[str, str] = {}
    
    for paper in papers:
        paper_refs = getattr(paper, 'references', None) or (paper.get('references') if isinstance(paper, dict) else None)
        if paper_refs:
            for ref in paper_refs:
                paper_id = ref.get("paperId")
                if paper_id:
                    reference_counter[paper_id] += 1
                    if paper_id not in reference_titles:
                        reference_titles[paper_id] = ref.get("title", "")

    top_references = reference_counter.most_common(top_k)
    
    return [
        (paper_id, count, reference_titles.get(paper_id, ""))
        for paper_id, count in top_references
    ]

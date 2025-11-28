from typing import List

from app.models.papers import DBPaperChunk
from app.retriever.paper_schemas import Paper
from collections import defaultdict



class RankingService:
    def rank_papers(self, query: str, papers: List[Paper], chunks: List[DBPaperChunk]) -> List[Paper]:
        """
        Rank papers based on relevance to the query

        Args:
            query: The search query
            papers: List of Paper objects to rank

        Returns:
            List of Paper objects ranked by relevance
        """
        paper_chunk_scores = defaultdict(float)

        for chunk in chunks:
            paper_chunk_scores[chunk.paper_id] += chunk.relevance_score or 0

        ranked_papers = sorted(
            papers,
            key=lambda p: p.relevance_score if p.relevance_score is not None else 0,
            reverse=True
        )
        return ranked_papers
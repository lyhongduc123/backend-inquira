"""
Ranking schemas - DTOs for ranking/scoring results
"""
from typing import Dict, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict

if TYPE_CHECKING:
    from app.models.papers import DBPaper


class RankedPaper(BaseModel):
    """
    Wrapper for DBPaper with ranking scores attached.
    
    Returned by RankingService after scoring and ranking.
    Wraps the actual DBPaper object (with all relationships) and adds computed scores.
    
    Usage:
        - RAG pipeline after ranking papers
        - Search results with quality scores
        - Access paper data via .paper attribute (DBPaper with authors, journal, etc.)
    
    Benefits:
        - No data duplication
        - Preserves all SQLAlchemy relationships (authors, journal, institutions)
        - No need to refetch from DB
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)
    
    id: int = Field(description="Database int ID")
    paper_id: str = Field(description="Unique paper identifier (e.g., Semantic Scholar ID)")
    paper: "DBPaper" = Field(description="Full DBPaper object with relationships")
    
    # Computed ranking fields (not in DB)
    relevance_score: float = Field(
        description="Final blended relevance score (quality + semantic similarity)"
    )
    ranking_scores: Dict[str, float] = Field(
        description="Detailed score breakdown (citation_quality, venue_prestige, etc.)"
    )


# Import DBPaper after class definition and rebuild model to resolve forward reference
from app.models.papers import DBPaper  # noqa: E402
RankedPaper.model_rebuild()

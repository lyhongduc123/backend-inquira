"""Query routing for chat task execution."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class QueryRouteDecision:
    """Resolved route for submitted chat task."""

    route_type: str
    scoped_paper_ids: List[str]
    total_steps: int


def extract_scoped_paper_ids(filters: Optional[Dict[str, Any]]) -> List[str]:
    """Extract scoped paper IDs from filters using snake_case/camelCase keys."""
    if not filters:
        return []

    raw_ids = filters.get("paper_ids")
    if raw_ids is None:
        raw_ids = filters.get("paperIds")

    if not isinstance(raw_ids, list):
        return []

    return [str(pid).strip() for pid in raw_ids if str(pid).strip()]


def route_query(pipeline_type: str, filters: Optional[Dict[str, Any]]) -> QueryRouteDecision:
    """Resolve which pipeline route to execute for a submitted task."""
    scoped_paper_ids = extract_scoped_paper_ids(filters)
    if scoped_paper_ids:
        # Scoped flow skips decomposition and runs tighter retrieval/ranking.
        return QueryRouteDecision(
            route_type="scoped",
            scoped_paper_ids=scoped_paper_ids,
            total_steps=2,
        )

    normalized = (pipeline_type or "database").strip().lower()
    if normalized not in {"database", "hybrid", "standard"}:
        normalized = "database"

    return QueryRouteDecision(
        route_type=normalized,
        scoped_paper_ids=[],
        total_steps=3,
    )

"""Query routing for chat task execution."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class QueryRouteDecision:
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
    scoped_paper_ids = extract_scoped_paper_ids(filters)
    if scoped_paper_ids:
        return QueryRouteDecision(
            route_type="scoped",
            scoped_paper_ids=scoped_paper_ids,
            total_steps=2,
        )

    return QueryRouteDecision(
        route_type="database",
        scoped_paper_ids=[],
        total_steps=3,
    )

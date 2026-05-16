"""Ranking fusion helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

T = TypeVar("T")


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[Tuple[T, float]]],
    *,
    key: Callable[[T], Any],
    k: int = 60,
    limit: int = 100,
) -> List[Tuple[T, float]]:
    """Fuse ranked result lists with Reciprocal Rank Fusion."""
    rrf_scores: Dict[str, float] = {}
    item_map: Dict[str, T] = {}

    for ranking in rankings:
        for rank, (item, _score) in enumerate(ranking, start=1):
            item_key = str(key(item))
            rrf_scores[item_key] = rrf_scores.get(item_key, 0.0) + 1.0 / (k + rank)
            if item_key not in item_map:
                item_map[item_key] = item

    ranked_keys = sorted(
        rrf_scores.keys(),
        key=lambda item_key: rrf_scores[item_key],
        reverse=True,
    )[:limit]
    return [(item_map[item_key], rrf_scores[item_key]) for item_key in ranked_keys]


def weighted_hybrid_fusion(
    bm25_candidates: List[Tuple[T, float]],
    semantic_candidates: List[Tuple[T, float]],
    *,
    key: Callable[[T], Any],
    bm25_weight: float = 0.3,
    semantic_weight: float = 0.7,
    rrf_k: float = 60.0,
    rrf_only: bool = False,
    limit: int = 100,
) -> List[Tuple[T, float]]:
    """
    Fuse lexical and semantic result lists.

    The base score is RRF over both ranked lists. When ``rrf_only`` is false,
    normalized point scores from each source are added with the provided weights.
    """
    if not bm25_candidates and not semantic_candidates:
        return []

    def _to_rank_map(candidates: List[Tuple[T, float]]) -> Dict[str, int]:
        return {
            str(key(item)): rank
            for rank, (item, _score) in enumerate(candidates, start=1)
        }

    def _to_score_map(candidates: List[Tuple[T, float]]) -> Dict[str, float]:
        return {str(key(item)): float(score) for item, score in candidates}

    bm25_rank = _to_rank_map(bm25_candidates)
    semantic_rank = _to_rank_map(semantic_candidates)
    bm25_score_map = _to_score_map(bm25_candidates)
    semantic_score_map = _to_score_map(semantic_candidates)

    bm25_max = max(bm25_score_map.values()) if bm25_score_map else 0.0
    semantic_max = max(semantic_score_map.values()) if semantic_score_map else 0.0

    item_map: Dict[str, T] = {}
    for item, _score in bm25_candidates:
        item_map[str(key(item))] = item
    for item, _score in semantic_candidates:
        item_map[str(key(item))] = item

    fused_scores: List[Tuple[T, float]] = []
    for item_key, item in item_map.items():
        rank_bm25: Optional[int] = bm25_rank.get(item_key)
        rank_semantic: Optional[int] = semantic_rank.get(item_key)

        rrf_score = 0.0
        if rank_bm25 is not None:
            rrf_score += 1.0 / (rrf_k + rank_bm25)
        if rank_semantic is not None:
            rrf_score += 1.0 / (rrf_k + rank_semantic)

        final_score = rrf_score
        if not rrf_only:
            bm25_component = (
                bm25_score_map.get(item_key, 0.0) / bm25_max
                if bm25_max > 0
                else 0.0
            )
            semantic_component = (
                semantic_score_map.get(item_key, 0.0) / semantic_max
                if semantic_max > 0
                else 0.0
            )
            final_score += (bm25_component * bm25_weight) + (
                semantic_component * semantic_weight
            )

        fused_scores.append((item, float(final_score)))

    fused_scores.sort(key=lambda item: item[1], reverse=True)
    return fused_scores[:limit]


def weighted_rrf_fusion(
    bm25_candidates: List[Tuple[T, float]],
    semantic_candidates: List[Tuple[T, float]],
    *,
    key: Callable[[T], Any],
    bm25_weight: float = 0.4,
    semantic_weight: float = 0.6,
    rrf_k: float = 60.0,
    limit: int = 100,
) -> List[Tuple[T, float]]:
    """Fuse lexical and semantic results with weighted Reciprocal Rank Fusion."""
    if not bm25_candidates and not semantic_candidates:
        return []

    bm25_rank = {
        str(key(item)): rank
        for rank, (item, _score) in enumerate(bm25_candidates, start=1)
    }
    semantic_rank = {
        str(key(item)): rank
        for rank, (item, _score) in enumerate(semantic_candidates, start=1)
    }

    item_map: Dict[str, T] = {}
    for item, _score in bm25_candidates:
        item_map[str(key(item))] = item
    for item, _score in semantic_candidates:
        item_map[str(key(item))] = item

    fused_scores: List[Tuple[T, float]] = []
    for item_key, item in item_map.items():
        score = 0.0
        rank_bm25 = bm25_rank.get(item_key)
        rank_semantic = semantic_rank.get(item_key)
        if rank_bm25 is not None:
            score += bm25_weight * (1.0 / (rrf_k + rank_bm25))
        if rank_semantic is not None:
            score += semantic_weight * (1.0 / (rrf_k + rank_semantic))
        fused_scores.append((item, float(score)))

    fused_scores.sort(key=lambda item: item[1], reverse=True)
    return fused_scores[:limit]

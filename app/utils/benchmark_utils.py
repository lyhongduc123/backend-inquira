"""Shared benchmark helper utilities."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Set

import numpy as np


def mask_db_url(url: str) -> str:
    """Mask password in a database URL for logs/responses."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.password:
            return url.replace(parsed.password, "***")
        return url
    except Exception:
        return "<configured>"


def calculate_recall_at_k(retrieved: Iterable[str], relevant: Set[str]) -> float:
    """Calculate Recall@k for retrieved IDs and relevant set."""
    retrieved_set = set(retrieved)
    if not relevant:
        return 0.0
    return len(retrieved_set & relevant) / len(relevant)


def calculate_precision_at_k(retrieved: Iterable[str], relevant: Set[str]) -> float:
    """Calculate Precision@k for retrieved IDs and relevant set."""
    retrieved_list = list(retrieved)
    if not retrieved_list:
        return 0.0
    return len(set(retrieved_list) & relevant) / len(retrieved_list)


def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1 from precision and recall."""
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    return 2 * precision * recall / denominator


def calculate_mrr(retrieved: Iterable[str], relevant: Set[str]) -> float:
    """Calculate Mean Reciprocal Rank."""
    for idx, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / idx
    return 0.0


def calculate_ndcg_at_k(
    retrieved: Iterable[str],
    relevance: Mapping[str, float] | Set[str],
    k: int,
) -> float:
    """Calculate NDCG@k with graded relevance map or binary relevant set."""
    ranked = list(retrieved)[:k]
    if not ranked:
        return 0.0

    if isinstance(relevance, set):
        relevance_map: Dict[str, float] = {doc_id: 1.0 for doc_id in relevance}
    else:
        relevance_map = {doc_id: float(score) for doc_id, score in relevance.items()}

    if not relevance_map:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(ranked, start=1):
        rel = relevance_map.get(doc_id, 0.0)
        dcg += rel / np.log2(i + 1)

    ideal_rels = sorted(relevance_map.values(), reverse=True)[: len(ranked)]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

"""
Information Retrieval metrics for benchmarking RAG pipeline.

Implements standard IR metrics:
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- Semantic diversity
"""
import math
from typing import List, Set, Dict, Any
import numpy as np
from app.extensions.logger import create_logger

logger = create_logger(__name__)


def calculate_precision_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int
) -> float:
    """
    Calculate Precision@K: proportion of top-K retrieved items that are relevant.
    
    Precision@K = |retrieved[:k] ∩ relevant| / k
    
    Args:
        retrieved: List of retrieved item IDs in ranked order
        relevant: Set of relevant item IDs (ground truth)
        k: Cutoff position
        
    Returns:
        Precision at K (0.0 to 1.0)
        
    Example:
        retrieved = ["p1", "p2", "p3", "p4", "p5"]
        relevant = {"p1", "p3", "p6"}
        precision_at_k(retrieved, relevant, 5) = 2/5 = 0.4
    """
    if k <= 0 or not retrieved:
        return 0.0
    
    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for item in top_k if item in relevant)
    
    return relevant_in_top_k / k


def calculate_recall_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int
) -> float:
    """
    Calculate Recall@K: proportion of relevant items found in top-K results.
    
    Recall@K = |retrieved[:k] ∩ relevant| / |relevant|
    
    Args:
        retrieved: List of retrieved item IDs in ranked order
        relevant: Set of relevant item IDs (ground truth)
        k: Cutoff position
        
    Returns:
        Recall at K (0.0 to 1.0)
        
    Example:
        retrieved = ["p1", "p2", "p3", "p4", "p5"]
        relevant = {"p1", "p3", "p6", "p7"}
        recall_at_k(retrieved, relevant, 5) = 2/4 = 0.5
    """
    if not relevant or k <= 0 or not retrieved:
        return 0.0
    
    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for item in top_k if item in relevant)
    
    return relevant_in_top_k / len(relevant)


def calculate_f1_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int
) -> float:
    """
    Calculate F1@K: harmonic mean of Precision@K and Recall@K.
    
    F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)
    
    Args:
        retrieved: List of retrieved item IDs in ranked order
        relevant: Set of relevant item IDs (ground truth)
        k: Cutoff position
        
    Returns:
        F1 score at K (0.0 to 1.0)
    """
    precision = calculate_precision_at_k(retrieved, relevant, k)
    recall = calculate_recall_at_k(retrieved, relevant, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def calculate_ndcg_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int,
    relevance_scores: Dict[str, float] | None = None
) -> float:
    """
    Calculate NDCG@K (Normalized Discounted Cumulative Gain).
    
    Position-aware metric that rewards relevant items appearing earlier.
    Uses binary relevance (relevant=1, not relevant=0) if no scores provided.
    
    DCG@K = Σ(rel_i / log2(i+1)) for i=1 to k
    NDCG@K = DCG@K / IDCG@K
    
    Args:
        retrieved: List of retrieved item IDs in ranked order
        relevant: Set of relevant item IDs (ground truth)
        k: Cutoff position
        relevance_scores: Optional dict mapping item_id -> relevance score (0-1)
                         If not provided, uses binary relevance (1 if relevant, 0 otherwise)
        
    Returns:
        NDCG at K (0.0 to 1.0)
        
    Example:
        retrieved = ["p1", "p2", "p3", "p4"]
        relevant = {"p1", "p3"}
        # DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) + 0/log2(5)
        #     = 1.0 + 0.0 + 0.5 + 0.0 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.63 = 1.63
        # NDCG = 1.5 / 1.63 = 0.92
    """
    if k <= 0 or not retrieved or not relevant:
        return 0.0
    
    # Calculate DCG@K
    dcg = 0.0
    for i, item_id in enumerate(retrieved[:k], start=1):
        if relevance_scores and item_id in relevance_scores:
            rel = relevance_scores[item_id]
        else:
            rel = 1.0 if item_id in relevant else 0.0
        
        # Discounted gain: divide by log2(position + 1)
        dcg += rel / math.log2(i + 1)
    
    # Calculate IDCG@K (ideal DCG with perfect ranking)
    # Sort relevant items by score (or all 1.0 for binary)
    if relevance_scores:
        ideal_scores = sorted(
            [relevance_scores.get(item_id, 0.0) for item_id in relevant],
            reverse=True
        )[:k]
    else:
        ideal_scores = [1.0] * min(len(relevant), k)
    
    idcg = sum(
        score / math.log2(i + 1)
        for i, score in enumerate(ideal_scores, start=1)
    )
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_mrr(
    retrieved: List[str],
    relevant: Set[str]
) -> float:
    """
    Calculate MRR (Mean Reciprocal Rank).
    
    Measures how quickly the first relevant result appears.
    MRR = 1 / rank_of_first_relevant_item
    
    Args:
        retrieved: List of retrieved item IDs in ranked order
        relevant: Set of relevant item IDs (ground truth)
        
    Returns:
        MRR score (0.0 to 1.0)
        
    Example:
        retrieved = ["p1", "p2", "p3", "p4"]
        relevant = {"p3", "p5"}
        # First relevant item is p3 at position 3
        # MRR = 1/3 = 0.333
    """
    if not retrieved or not relevant:
        return 0.0
    
    for i, item_id in enumerate(retrieved, start=1):
        if item_id in relevant:
            return 1.0 / i
    
    # No relevant item found
    return 0.0


def calculate_average_precision(
    retrieved: List[str],
    relevant: Set[str]
) -> float:
    """
    Calculate Average Precision (AP).
    
    Average of precision values at each position where a relevant item appears.
    
    AP = (Σ P(k) × rel(k)) / |relevant|
    
    where:
    - P(k) is precision at position k
    - rel(k) is 1 if item at k is relevant, 0 otherwise
    
    Args:
        retrieved: List of retrieved item IDs in ranked order
        relevant: Set of relevant item IDs (ground truth)
        
    Returns:
        Average Precision (0.0 to 1.0)
    """
    if not relevant or not retrieved:
        return 0.0
    
    precision_sum = 0.0
    relevant_count = 0
    
    for i, item_id in enumerate(retrieved, start=1):
        if item_id in relevant:
            relevant_count += 1
            precision_at_i = relevant_count / i
            precision_sum += precision_at_i
    
    if relevant_count == 0:
        return 0.0
    
    return precision_sum / len(relevant)


def calculate_semantic_diversity(
    embeddings: List[List[float]] | np.ndarray,
    method: str = "cosine"
) -> float:
    """
    Calculate semantic diversity of a set of embeddings.
    
    Measures how diverse/spread out the embeddings are in semantic space.
    Higher score = more diverse content.
    
    Args:
        embeddings: List of embedding vectors or numpy array
        method: "cosine" (cosine distance) or "euclidean"
        
    Returns:
        Diversity score (0.0 to 1.0)
        1.0 = maximally diverse, 0.0 = all identical
    """
    if not embeddings or len(embeddings) < 2:
        return 0.0
    
    embeddings_array = np.array(embeddings)
    n = len(embeddings_array)
    
    if method == "cosine":
        # Calculate pairwise cosine similarities
        # Cosine similarity = dot(A, B) / (||A|| * ||B||)
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings_array)
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = similarities[np.triu_indices(n, k=1)]
        
        # Average similarity
        avg_similarity = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0.0
        
        # Diversity = 1 - similarity
        return 1.0 - avg_similarity
    
    elif method == "euclidean":
        # Calculate pairwise euclidean distances
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(embeddings_array)
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = distances[np.triu_indices(n, k=1)]
        
        # Normalize by max possible distance
        avg_distance = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0.0
        max_distance = np.max(upper_triangle) if len(upper_triangle) > 0 else 1.0
        
        return avg_distance / max_distance if max_distance > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'cosine' or 'euclidean'")


def analyze_retrieval_metrics(
    retrieved: List[str],
    relevant: Set[str],
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, Any]:
    """
    Calculate all retrieval metrics at once for multiple K values.
    
    Args:
        retrieved: List of retrieved item IDs in ranked order
        relevant: Set of relevant item IDs (ground truth)
        k_values: List of K values to calculate metrics for
        
    Returns:
        Dictionary with all metrics:
        {
            "precision": {5: 0.8, 10: 0.7, 20: 0.65},
            "recall": {5: 0.4, 10: 0.7, 20: 0.9},
            "f1": {5: 0.53, 10: 0.7, 20: 0.75},
            "ndcg": {5: 0.85, 10: 0.88, 20: 0.9},
            "mrr": 0.5,
            "average_precision": 0.75,
            "true_positives": ["p1", "p3"],
            "false_positives": ["p2", "p4"],
            "false_negatives": ["p5", "p6"],
        }
    """
    metrics = {
        "precision": {},
        "recall": {},
        "f1": {},
        "ndcg": {},
    }
    
    # Calculate metrics for each K
    for k in k_values:
        metrics["precision"][k] = calculate_precision_at_k(retrieved, relevant, k)
        metrics["recall"][k] = calculate_recall_at_k(retrieved, relevant, k)
        metrics["f1"][k] = calculate_f1_at_k(retrieved, relevant, k)
        metrics["ndcg"][k] = calculate_ndcg_at_k(retrieved, relevant, k)
    
    # Calculate single-value metrics
    metrics["mrr"] = calculate_mrr(retrieved, relevant)
    metrics["average_precision"] = calculate_average_precision(retrieved, relevant)
    
    # Analysis
    retrieved_set = set(retrieved)
    metrics["true_positives"] = list(retrieved_set & relevant)
    metrics["false_positives"] = list(retrieved_set - relevant)
    metrics["false_negatives"] = list(relevant - retrieved_set)
    
    return metrics


def calculate_query_diversity(queries: List[str], embeddings: List[List[float]]) -> float:
    """
    Calculate diversity of generated search queries.
    
    Args:
        queries: List of query strings
        embeddings: List of query embeddings
        
    Returns:
        Diversity score (0.0 to 1.0)
    """
    if len(queries) < 2:
        return 0.0
    
    # Semantic diversity
    semantic_div = calculate_semantic_diversity(embeddings)
    
    # Lexical diversity (unique words)
    all_words = set()
    total_words = 0
    for query in queries:
        words = query.lower().split()
        all_words.update(words)
        total_words += len(words)
    
    lexical_div = len(all_words) / total_words if total_words > 0 else 0.0
    
    # Combine both (weighted average)
    return 0.7 * semantic_div + 0.3 * lexical_div


def calculate_coverage_score(
    papers_covered: int,
    sections_covered: int,
    total_chunks: int
) -> float:
    """
    Calculate coverage score for retrieved chunks.
    
    Measures how well chunks cover different papers and sections.
    
    Args:
        papers_covered: Number of unique papers
        sections_covered: Number of unique sections
        total_chunks: Total number of chunks
        
    Returns:
        Coverage score (0.0 to 1.0)
    """
    if total_chunks == 0:
        return 0.0
    
    # Normalize by expected values
    # Ideal: 5-10 papers, 10-20 sections for 40 chunks
    paper_score = min(papers_covered / 10.0, 1.0)
    section_score = min(sections_covered / 20.0, 1.0)
    
    # Weighted combination
    return 0.6 * paper_score + 0.4 * section_score

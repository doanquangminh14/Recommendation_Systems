"""
Evaluation & Benchmarking Metrics Module for Recommender Systems.

Provides standard evaluation metrics:
- Rating Prediction Metrics: RMSE, MAE, Explained Variance.
- Ranking Quality Metrics: Precision@K, Recall@K, NDCG@K, HitRate@K, MAP@K.
- Diversity & Coverage Metrics: Intra-List Diversity (ILD), Catalog Coverage.
"""

from typing import List, Dict, Set, Any, Union, Optional
import numpy as np


# ============================================================================
# 1. Rating Prediction Error Metrics
# ============================================================================

def compute_rmse(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> float:
    """
    Computes Root Mean Squared Error (RMSE).
    
    RMSE = sqrt( (1 / N) * sum( (y_true - y_pred)^2 ) )
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    if len(y_true_arr) == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def compute_mae(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> float:
    """
    Computes Mean Absolute Error (MAE).
    
    MAE = (1 / N) * sum( |y_true - y_pred| )
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    if len(y_true_arr) == 0:
        return 0.0
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


# ============================================================================
# 2. Ranking Quality Metrics (Top-K)
# ============================================================================

def compute_precision_at_k(actual: Union[Set[str], List[str]], predicted: List[str], k: int = 10) -> float:
    """
    Computes Precision@K: Proportion of recommended items in Top-K that are relevant.
    
    Precision@K = |TopK(Rec) intersect Relevant| / K
    """
    if k <= 0:
        return 0.0
    top_k_pred = predicted[:k]
    actual_set = set(actual)
    hits = sum(1 for item in top_k_pred if item in actual_set)
    return float(hits / k)


def compute_recall_at_k(actual: Union[Set[str], List[str]], predicted: List[str], k: int = 10) -> float:
    """
    Computes Recall@K: Proportion of relevant items that are captured in Top-K.
    
    Recall@K = |TopK(Rec) intersect Relevant| / |Relevant|
    """
    actual_set = set(actual)
    if not actual_set:
        return 0.0
    top_k_pred = predicted[:k]
    hits = sum(1 for item in top_k_pred if item in actual_set)
    return float(hits / len(actual_set))


def compute_hit_rate_at_k(actual: Union[Set[str], List[str]], predicted: List[str], k: int = 10) -> float:
    """
    Computes Hit Rate@K: Returns 1.0 if at least one relevant item is in Top-K, else 0.0.
    """
    actual_set = set(actual)
    top_k_pred = predicted[:k]
    return 1.0 if any(item in actual_set for item in top_k_pred) else 0.0


def compute_ndcg_at_k(actual: Union[Set[str], List[str]], predicted: List[str], k: int = 10) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K (NDCG@K) with binary relevance.
    
    DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
    IDCG@K = sum_{i=1}^{min(|Rel|, K)} (1 / log2(i + 1))
    NDCG@K = DCG@K / IDCG@K
    """
    actual_set = set(actual)
    if not actual_set or k <= 0:
        return 0.0

    top_k_pred = predicted[:k]
    dcg = 0.0
    for i, item in enumerate(top_k_pred):
        if item in actual_set:
            dcg += 1.0 / np.log2(i + 2)  # i is 0-indexed, so rank = i+1, log2(rank+1) = log2(i+2)

    # Ideal DCG (all hits placed at the top ranks)
    ideal_hits = min(len(actual_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def compute_map_at_k(actual: Union[Set[str], List[str]], predicted: List[str], k: int = 10) -> float:
    """
    Computes Mean Average Precision at K (MAP@K / AP@K).
    """
    actual_set = set(actual)
    if not actual_set or k <= 0:
        return 0.0

    top_k_pred = predicted[:k]
    score = 0.0
    num_hits = 0.0

    for i, item in enumerate(top_k_pred):
        if item in actual_set:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return float(score / min(len(actual_set), k))


# ============================================================================
# 3. Diversity & Novelty Metrics
# ============================================================================

def compute_intra_list_diversity(item_ids: List[str], embeddings: np.ndarray, item2idx: Dict[str, int]) -> float:
    """
    Computes Intra-List Diversity (ILD) as average pairwise cosine distance (1 - CosineSim).
    
    ILD(R) = (2 / (|R| * (|R| - 1))) * sum_{i < j} (1 - cos(v_i, v_j))
    """
    valid_indices = [item2idx[iid] for iid in item_ids if iid in item2idx]
    if len(valid_indices) <= 1:
        return 0.0

    vecs = embeddings[valid_indices]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    norm_vecs = vecs / norms

    sim_matrix = np.dot(norm_vecs, norm_vecs.T)
    K = len(valid_indices)
    distances = 1.0 - sim_matrix
    upper_tri = distances[np.triu_indices(K, k=1)]
    return float(np.mean(upper_tri))


def compute_catalog_coverage(recommended_item_ids: Set[str], total_catalog_count: int) -> float:
    """
    Computes Catalog Coverage: Proportion of distinct items recommended relative to catalog size.
    
    Coverage = |Unique Recommended Items| / |Total Catalog Items|
    """
    if total_catalog_count <= 0:
        return 0.0
    return float(len(recommended_item_ids) / total_catalog_count)


# ============================================================================
# 4. Aggregated Benchmark Evaluator
# ============================================================================

def evaluate_ranking_predictions(
    user_actual_map: Dict[str, List[str]],
    user_predicted_map: Dict[str, List[str]],
    k_values: List[int] = [5, 10],
) -> Dict[str, float]:
    """
    Computes aggregated Precision@K, Recall@K, NDCG@K, and HitRate@K across all test users.
    
    Parameters
    ----------
    user_actual_map : dict
        Mapping of user_id to list of relevant/ground-truth item IDs.
    user_predicted_map : dict
        Mapping of user_id to list of recommended item IDs.
    k_values : list of int
        Cutoff ranks K (e.g. [5, 10]).
        
    Returns
    -------
    dict
        Dictionary of averaged evaluation metrics.
    """
    metrics: Dict[str, List[float]] = {}
    for k in k_values:
        metrics[f"precision@{k}"] = []
        metrics[f"recall@{k}"] = []
        metrics[f"ndcg@{k}"] = []
        metrics[f"hit_rate@{k}"] = []
        metrics[f"map@{k}"] = []

    for user_id, actual in user_actual_map.items():
        predicted = user_predicted_map.get(user_id, [])
        for k in k_values:
            metrics[f"precision@{k}"].append(compute_precision_at_k(actual, predicted, k=k))
            metrics[f"recall@{k}"].append(compute_recall_at_k(actual, predicted, k=k))
            metrics[f"ndcg@{k}"].append(compute_ndcg_at_k(actual, predicted, k=k))
            metrics[f"hit_rate@{k}"].append(compute_hit_rate_at_k(actual, predicted, k=k))
            metrics[f"map@{k}"].append(compute_map_at_k(actual, predicted, k=k))

    summary = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
    return summary

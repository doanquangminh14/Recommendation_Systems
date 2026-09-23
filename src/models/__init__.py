"""
Models package for Recommender Systems.
"""

from src.models.collaborative.matrix_factorization import SVDRecommender
from src.models.content_based.recommender import ContentBasedRecommender
from src.models.hybrid.hybrid_engine import HybridRecommender, HybridRecommendationResult, HybridRecommendationOutput
from src.models.hybrid.explainer import RecommendationExplainer, RecommendationExplanation
from src.models.ranking import DiversityRanker, DiversityMetrics
from src.models.evaluation import (
    compute_rmse,
    compute_mae,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_hit_rate_at_k,
    compute_ndcg_at_k,
    compute_map_at_k,
    compute_intra_list_diversity,
    compute_catalog_coverage,
    evaluate_ranking_predictions,
)

__all__ = [
    "SVDRecommender",
    "ContentBasedRecommender",
    "HybridRecommender",
    "HybridRecommendationResult",
    "HybridRecommendationOutput",
    "RecommendationExplainer",
    "RecommendationExplanation",
    "DiversityRanker",
    "DiversityMetrics",
    "compute_rmse",
    "compute_mae",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "compute_hit_rate_at_k",
    "compute_ndcg_at_k",
    "compute_map_at_k",
    "compute_intra_list_diversity",
    "compute_catalog_coverage",
    "evaluate_ranking_predictions",
]




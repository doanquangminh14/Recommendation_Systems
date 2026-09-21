"""
Hybrid Recommender Package.

Combines Collaborative Filtering (SVD), Content-Based Filtering (Semantic Embeddings),
and Sentiment Signals (VADER) with adaptive weighting and cold-start support.
"""

from src.models.hybrid.hybrid_engine import HybridRecommender, HybridRecommendationResult, HybridRecommendationOutput
from src.models.hybrid.explainer import RecommendationExplainer, RecommendationExplanation

__all__ = [
    "HybridRecommender",
    "HybridRecommendationResult",
    "HybridRecommendationOutput",
    "RecommendationExplainer",
    "RecommendationExplanation",
]


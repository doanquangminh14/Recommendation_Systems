"""
Models package for Recommender Systems.
"""

from src.models.collaborative.matrix_factorization import SVDRecommender
from src.models.content_based.recommender import ContentBasedRecommender
from src.models.hybrid.hybrid_engine import HybridRecommender

__all__ = [
    "SVDRecommender",
    "ContentBasedRecommender",
    "HybridRecommender",
]


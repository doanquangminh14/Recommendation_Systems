"""
Hybrid Recommender Engine.

Integrates Collaborative Filtering (TruncatedSVD), Content-Based Filtering (Semantic Embeddings Centroid),
and NLP Sentiment Signals (VADER item metrics) into a unified scoring pipeline with
multi-source score normalization and adaptive cold-start weights.
"""

from typing import List, Dict, Optional, Tuple, Any, Union
import os
import numpy as np
import polars as pl
from dataclasses import dataclass, field

from src.models.collaborative.matrix_factorization import SVDRecommender
from src.models.content_based.recommender import ContentBasedRecommender


@dataclass
class HybridRecommendationResult:
    """
    Data container for a single hybrid recommendation item.
    """
    parent_asin: str
    title: str
    category: str
    hybrid_score: float
    cf_score: float
    cb_score: float
    sentiment_score: float
    avg_rating: float
    rating_number: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_asin": self.parent_asin,
            "title": self.title,
            "category": self.category,
            "hybrid_score": round(self.hybrid_score, 4),
            "cf_score": round(self.cf_score, 4),
            "cb_score": round(self.cb_score, 4),
            "sentiment_score": round(self.sentiment_score, 4),
            "avg_rating": round(self.avg_rating, 2),
            "rating_number": self.rating_number,
            **self.extra,
        }


@dataclass
class HybridRecommendationOutput:
    """
    Data container for hybrid recommendation output batch and execution metadata.
    """
    items: List[HybridRecommendationResult]
    effective_weights: Dict[str, float]
    cf_available: bool
    user_liked_history: List[str]

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.items]


class HybridRecommender:
    """
    High-performance Hybrid Recommendation Engine combining:
    1. Collaborative Filtering (Latent space dot-products via TruncatedSVD).
    2. Content-Based Filtering (Semantic embedding dot-products with user preference centroid).
    3. Sentiment Scoring (VADER compound score and positive review ratio).
    """

    def __init__(
        self,
        cb_model: Optional[ContentBasedRecommender] = None,
        svd_model: Optional[SVDRecommender] = None,
        df_sentiment: Optional[pl.DataFrame] = None,
        df_interactions: Optional[pl.DataFrame] = None,
        embeddings_path: str = "data/gold/item_embeddings.npy",
        items_path: str = "data/silver/item_features.parquet",
        svd_model_path: str = "models/collaborative/svd_recommender.joblib",
        sentiment_path: str = "data/silver/item_sentiment.parquet",
        interactions_path: str = "data/silver/interactions.parquet",
    ):
        """
        Initialize the Hybrid Recommender with pre-loaded models or file paths.
        """
        # 1. Content-Based Model
        if cb_model is not None:
            self.cb_model = cb_model
        else:
            self.cb_model = ContentBasedRecommender(
                embeddings_path=embeddings_path,
                items_path=items_path,
            )

        # 2. Collaborative Filtering (SVD) Model
        if svd_model is not None:
            self.svd_model = svd_model
        else:
            if os.path.exists(svd_model_path):
                self.svd_model = SVDRecommender.load_model(svd_model_path)
            else:
                self.svd_model = None

        # 3. Sentiment Data
        if df_sentiment is not None:
            self.df_sentiment = df_sentiment
        elif os.path.exists(sentiment_path):
            self.df_sentiment = pl.read_parquet(sentiment_path)
        else:
            self.df_sentiment = pl.DataFrame()

        # 4. Interactions Data
        if df_interactions is not None:
            self.df_interactions = df_interactions
        elif os.path.exists(interactions_path):
            self.df_interactions = pl.read_parquet(interactions_path)
        else:
            self.df_interactions = pl.DataFrame()

        # Build fast lookup structures
        self.item_ids = self.cb_model.item_ids
        self.item2idx = self.cb_model.item2idx
        self.idx2item = self.cb_model.idx2item
        self.n_items = len(self.item_ids)

        # Initialize sentiment lookup vectors
        self._init_sentiment_scores()

        # Cache user interactions for fast filtering
        self._init_user_history_cache()

    def _init_sentiment_scores(self) -> None:
        """
        Precompute normalized sentiment scores aligned with catalog item indices.
        Formula: S_sent(i) = 0.6 * positive_ratio(i) + 0.4 * ((compound(i) + 1) / 2)
        """
        self.sent_info: Dict[str, Dict[str, Any]] = {}
        if not self.df_sentiment.is_empty():
            for row in self.df_sentiment.iter_rows(named=True):
                self.sent_info[row["parent_asin"]] = row

        self.item_sentiment_scores = np.zeros(self.n_items, dtype=np.float32)
        for i, iid in enumerate(self.item_ids):
            if iid in self.sent_info:
                info = self.sent_info[iid]
                pos = float(info.get("positive_review_ratio", 0.5) or 0.5)
                comp = float(info.get("avg_sentiment_compound", 0.0) or 0.0)
                norm_comp = (comp + 1.0) / 2.0  # Scale [-1, 1] to [0, 1]
                self.item_sentiment_scores[i] = 0.6 * pos + 0.4 * norm_comp
            else:
                self.item_sentiment_scores[i] = 0.5  # Neutral default

    def _init_user_history_cache(self) -> None:
        """
        Build an in-memory dictionary of user past interactions for quick masking.
        """
        self.user_history_dict: Dict[str, List[str]] = {}
        if not self.df_interactions.is_empty():
            # Group by user_id
            grouped = self.df_interactions.group_by("user_id").agg(pl.col("parent_asin").alias("items"))
            for row in grouped.iter_rows(named=True):
                self.user_history_dict[row["user_id"]] = row["items"]

    @staticmethod
    def _min_max_scale(arr: np.ndarray) -> np.ndarray:
        """
        Linearly scale array to range [0, 1].
        """
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        if max_v > min_v:
            return (arr - min_v) / (max_v - min_v)
        return np.zeros_like(arr)

    def recommend(
        self,
        user_id: Optional[str] = None,
        liked_item_ids: Optional[List[str]] = None,
        liked_weights: Optional[List[float]] = None,
        w_cf: float = 0.50,
        w_cb: float = 0.35,
        w_sent: float = 0.15,
        top_k: int = 10,
        exclude_interacted: bool = True,
        category_filter: Optional[str] = None,
        min_sentiment_score: Optional[float] = None,
    ) -> HybridRecommendationOutput:
        """
        Generate Top-K hybrid recommendations with adaptive cold-start weights.

        Parameters
        ----------
        user_id : str or None
            Target user identifier.
        liked_item_ids : list of str or None
            Explicitly liked items (e.g. cold-start seeds or dynamic query).
        liked_weights : list of float or None
            Custom affinity weights corresponding to liked_item_ids.
        w_cf : float
            Weight for Collaborative Filtering signal (default: 0.50).
        w_cb : float
            Weight for Content-Based Semantic signal (default: 0.35).
        w_sent : float
            Weight for NLP Sentiment signal (default: 0.15).
        top_k : int
            Number of top recommended items to return.
        exclude_interacted : bool
            Whether to mask out items already interacted by the user.
        category_filter : str or None
            Optional filter to restrict recommendations to a single main_category.
        min_sentiment_score : float or None
            Optional threshold to filter out items with sentiment score below this value.

        Returns
        -------
        HybridRecommendationOutput
            Dataclass containing list of recommended items and execution metadata.
        """
        interacted_asins = set()
        user_liked_history = []

        # -------------------------------------------------------------
        # 1. Collaborative Filtering Score Vector (SVD)
        # -------------------------------------------------------------
        cf_available = False
        scores_cf = np.zeros(self.n_items, dtype=np.float32)

        if user_id and self.svd_model is not None and user_id in self.svd_model.user2idx:
            u_idx = self.svd_model.user2idx[user_id]
            u_vec = self.svd_model.user_factors[u_idx]  # Shape: (n_factors,)
            
            # Predict scores for all SVD items via dot product
            svd_item_scores = np.dot(u_vec, self.svd_model.item_factors.T)  # Shape: (n_svd_items,)
            
            # Map SVD scores to catalog item order
            for i, iid in enumerate(self.item_ids):
                if iid in self.svd_model.item2idx:
                    scores_cf[i] = svd_item_scores[self.svd_model.item2idx[iid]]
                else:
                    scores_cf[i] = self.svd_model.global_mean

            scores_cf = self._min_max_scale(scores_cf)
            cf_available = True

            # Retrieve user history
            hist = self.user_history_dict.get(user_id, [])
            interacted_asins.update(hist)
            user_liked_history.extend(hist)

        # -------------------------------------------------------------
        # 2. Content-Based Semantic Score Vector
        # -------------------------------------------------------------
        scores_cb = np.zeros(self.n_items, dtype=np.float32)

        if liked_item_ids:
            interacted_asins.update(liked_item_ids)
            user_liked_history.extend(liked_item_ids)
            valid_indices = [self.item2idx[iid] for iid in liked_item_ids if iid in self.item2idx]
            
            if valid_indices:
                if liked_weights and len(liked_weights) == len(liked_item_ids):
                    valid_weights = [liked_weights[i] for i, iid in enumerate(liked_item_ids) if iid in self.item2idx]
                    w = np.array(valid_weights, dtype=np.float32)
                else:
                    w = np.ones(len(valid_indices), dtype=np.float32)

                item_vecs = self.cb_model.embeddings[valid_indices]
                user_centroid = np.sum(item_vecs * w.reshape(-1, 1), axis=0)
                norm = float(np.linalg.norm(user_centroid))
                if norm > 0:
                    user_centroid /= norm
                scores_cb = np.dot(self.cb_model.embeddings, user_centroid)
                scores_cb = self._min_max_scale(scores_cb)
        elif user_id and cf_available:
            # Fallback: construct profile from top-rated historical items of user
            user_top_items = self.user_history_dict.get(user_id, [])[:5]
            valid_indices = [self.item2idx[iid] for iid in user_top_items if iid in self.item2idx]
            if valid_indices:
                user_centroid = np.mean(self.cb_model.embeddings[valid_indices], axis=0)
                norm = float(np.linalg.norm(user_centroid))
                if norm > 0:
                    user_centroid /= norm
                scores_cb = np.dot(self.cb_model.embeddings, user_centroid)
                scores_cb = self._min_max_scale(scores_cb)

        # -------------------------------------------------------------
        # 3. NLP Sentiment Score Vector
        # -------------------------------------------------------------
        scores_sent = self.item_sentiment_scores.copy()

        # -------------------------------------------------------------
        # 4. Adaptive Weights Normalization
        # -------------------------------------------------------------
        if not cf_available:
            effective_w_cf = 0.0
            sum_w = w_cb + w_sent
            effective_w_cb = (w_cb / sum_w) if sum_w > 0 else 0.70
            effective_w_sent = (w_sent / sum_w) if sum_w > 0 else 0.30
        else:
            total_w = w_cf + w_cb + w_sent
            if total_w > 0:
                effective_w_cf = w_cf / total_w
                effective_w_cb = w_cb / total_w
                effective_w_sent = w_sent / total_w
            else:
                effective_w_cf, effective_w_cb, effective_w_sent = 0.50, 0.35, 0.15

        # -------------------------------------------------------------
        # 5. Hybrid Linear Fusion
        # -------------------------------------------------------------
        hybrid_scores = (
            effective_w_cf * scores_cf +
            effective_w_cb * scores_cb +
            effective_w_sent * scores_sent
        )

        # Mask interacted items
        if exclude_interacted and interacted_asins:
            for iid in interacted_asins:
                if iid in self.item2idx:
                    hybrid_scores[self.item2idx[iid]] = -np.inf

        # Category filter
        if category_filter:
            for i, iid in enumerate(self.item_ids):
                if self.cb_model.item_categories.get(iid) != category_filter:
                    hybrid_scores[i] = -np.inf

        # Sentiment threshold filter
        if min_sentiment_score is not None:
            for i in range(self.n_items):
                if scores_sent[i] < min_sentiment_score:
                    hybrid_scores[i] = -np.inf

        # -------------------------------------------------------------
        # 6. Top-K Candidates Selection
        # -------------------------------------------------------------
        top_k_clamped = min(top_k, self.n_items)
        top_indices = np.argpartition(hybrid_scores, -top_k_clamped)[-top_k_clamped:]
        top_indices = top_indices[np.argsort(-hybrid_scores[top_indices])]

        results: List[HybridRecommendationResult] = []
        for idx in top_indices:
            score = float(hybrid_scores[idx])
            if score == -np.inf:
                continue

            iid = self.idx2item[idx]
            rating_num = 0
            if iid in self.sent_info:
                rating_num = int(self.sent_info[iid].get("review_count", 0) or 0)

            results.append(
                HybridRecommendationResult(
                    parent_asin=iid,
                    title=self.cb_model.item_titles.get(iid, "Unknown Title"),
                    category=self.cb_model.item_categories.get(iid, "Unknown Category"),
                    hybrid_score=score,
                    cf_score=float(scores_cf[idx]) if cf_available else 0.0,
                    cb_score=float(scores_cb[idx]),
                    sentiment_score=float(scores_sent[idx]),
                    avg_rating=float(self.cb_model.item_ratings.get(iid, 0.0) or 0.0),
                    rating_number=rating_num,
                )
            )

        effective_weights_dict = {
            "w_cf": round(effective_w_cf, 4),
            "w_cb": round(effective_w_cb, 4),
            "w_sent": round(effective_w_sent, 4),
        }

        # Deduplicate user history
        user_liked_history = list(dict.fromkeys(user_liked_history))

        return HybridRecommendationOutput(
            items=results,
            effective_weights=effective_weights_dict,
            cf_available=cf_available,
            user_liked_history=user_liked_history,
        )

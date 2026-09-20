"""
Content-Based Filtering Recommender System.

Utilizes dense semantic item embeddings (384-D Sentence-Transformers) and metadata
to compute Item-to-Item similarity and User-Profile vector aggregations (solving Cold-Start).
"""

from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import polars as pl
import os


class ContentBasedRecommender:
    """
    Fast Content-Based Recommender utilizing L2-normalized dense embeddings and vector dot products.
    """

    def __init__(
        self,
        embeddings_path: str = "data/gold/item_embeddings.npy",
        items_path: str = "data/silver/item_features.parquet",
    ):
        """
        Initialize the recommender by loading embeddings matrix and item metadata.
        """
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings matrix not found at {embeddings_path}. Run embeddings pipeline first.")
        if not os.path.exists(items_path):
            raise FileNotFoundError(f"Item features dataset not found at {items_path}.")

        self.embeddings: np.ndarray = np.load(embeddings_path)  # Shape: (N, 384)
        self.df_items: pl.DataFrame = pl.read_parquet(items_path)
        
        self.item_ids: List[str] = self.df_items["parent_asin"].to_list()
        self.item2idx: Dict[str, int] = {iid: i for i, iid in enumerate(self.item_ids)}
        self.idx2item: Dict[int, str] = {i: iid for i, iid in enumerate(self.item_ids)}
        
        # Fast lookup dictionaries for metadata
        self.item_titles: Dict[str, str] = dict(zip(self.item_ids, self.df_items["title"].to_list()))
        self.item_categories: Dict[str, str] = dict(zip(self.item_ids, self.df_items["main_category"].to_list()))
        self.item_ratings: Dict[str, float] = dict(zip(self.item_ids, self.df_items["average_rating"].to_list()))

    def get_similar_items(
        self,
        item_id: str,
        top_k: int = 10,
        exclude_self: bool = True,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find Top-K most similar games to a given target game using Cosine Similarity.

        Parameters
        ----------
        item_id : str
            Parent ASIN of target game.
        top_k : int
            Number of similar items to retrieve.
        exclude_self : bool
            Whether to exclude the query game from results.
        category_filter : str or None
            Optional filter by main category.

        Returns
        -------
        list of dict
            Each dict contains 'parent_asin', 'title', 'main_category', 'similarity_score', 'average_rating'.
        """
        if item_id not in self.item2idx:
            raise KeyError(f"Item ID '{item_id}' not found in catalog.")

        idx = self.item2idx[item_id]
        query_vec = self.embeddings[idx]  # Shape: (384,)

        # Dot product with all normalized embeddings = Cosine Similarity
        sim_scores = np.dot(self.embeddings, query_vec)  # Shape: (N,)

        if exclude_self:
            sim_scores[idx] = -np.inf

        # Apply category filter if provided
        if category_filter:
            for i, iid in enumerate(self.item_ids):
                if self.item_categories.get(iid) != category_filter:
                    sim_scores[i] = -np.inf

        top_indices = np.argpartition(sim_scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-sim_scores[top_indices])]

        results = []
        for i in top_indices:
            score = float(sim_scores[i])
            if score == -np.inf:
                continue
            iid = self.idx2item[i]
            results.append({
                "parent_asin": iid,
                "title": self.item_titles.get(iid, "Unknown"),
                "main_category": self.item_categories.get(iid, "Unknown"),
                "similarity_score": round(score, 4),
                "average_rating": self.item_ratings.get(iid, 0.0),
            })
        return results

    def recommend_for_user_profile(
        self,
        liked_item_ids: List[str],
        weights: Optional[List[float]] = None,
        top_k: int = 10,
        exclude_interacted: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a user based on their liked items vector centroid.
        Provides zero-shot Cold-Start handling for any user profile.

        Parameters
        ----------
        liked_item_ids : list of str
            List of parent_asin that user previously enjoyed / rated highly.
        weights : list of float or None
            Weights for each liked item (e.g. user ratings or interaction strength).
        top_k : int
            Number of recommendations to return.
        exclude_interacted : bool
            Whether to exclude items the user already interacted with.

        Returns
        -------
        list of dict
            Recommended games with similarity scores and metadata.
        """
        valid_indices = []
        valid_weights = []

        for i, item_id in enumerate(liked_item_ids):
            if item_id in self.item2idx:
                valid_indices.append(self.item2idx[item_id])
                w = weights[i] if (weights and i < len(weights)) else 1.0
                valid_weights.append(w)

        if not valid_indices:
            return []

        # Weighted sum / centroid of item vectors
        item_vecs = self.embeddings[valid_indices]  # Shape: (M, 384)
        w_arr = np.array(valid_weights, dtype=np.float32).reshape(-1, 1)  # Shape: (M, 1)
        user_vector = np.sum(item_vecs * w_arr, axis=0)  # Shape: (384,)

        # L2 Normalize user vector
        norm = np.linalg.norm(user_vector)
        if norm > 0:
            user_vector = user_vector / norm

        # Compute similarity scores against all items in catalog
        sim_scores = np.dot(self.embeddings, user_vector)

        if exclude_interacted:
            for idx in valid_indices:
                sim_scores[idx] = -np.inf

        top_indices = np.argpartition(sim_scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-sim_scores[top_indices])]

        results = []
        for i in top_indices:
            score = float(sim_scores[i])
            if score == -np.inf:
                continue
            iid = self.idx2item[i]
            results.append({
                "parent_asin": iid,
                "title": self.item_titles.get(iid, "Unknown"),
                "main_category": self.item_categories.get(iid, "Unknown"),
                "similarity_score": round(score, 4),
                "average_rating": self.item_ratings.get(iid, 0.0),
            })
        return results

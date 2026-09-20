"""
Collaborative Filtering via TruncatedSVD Matrix Factorization.

Constructs sparse User-Item interaction matrices and computes latent representation
factors for users and items, predicting user preferences and generating Top-K recommendations.
"""

from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import joblib


class SVDRecommender:
    """
    SVD-based Collaborative Filtering Recommender using Scikit-Learn TruncatedSVD on sparse CSR matrices.
    """

    def __init__(self, n_factors: int = 64, random_state: int = 42):
        """
        Parameters
        ----------
        n_factors : int
            Number of latent dimensions (default: 64).
        random_state : int
            Random seed for reproducibility.
        """
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_factors, algorithm="randomized", random_state=random_state)
        
        self.user2idx: Dict[str, int] = {}
        self.idx2user: Dict[int, str] = {}
        self.item2idx: Dict[str, int] = {}
        self.idx2item: Dict[int, str] = {}
        
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.global_mean: float = 0.0

    def fit_transform_interactions(
        self,
        df_interactions: pl.DataFrame,
        user_col: str = "user_id",
        item_col: str = "parent_asin",
        rating_col: str = "rating"
    ) -> Tuple[csr_matrix, Dict[str, Any]]:
        """
        Build sparse matrix and fit TruncatedSVD latent factors.
        """
        unique_users = df_interactions[user_col].unique().to_list()
        unique_items = df_interactions[item_col].unique().to_list()

        self.user2idx = {uid: i for i, uid in enumerate(unique_users)}
        self.idx2user = {i: uid for uid, i in self.user2idx.items()}
        self.item2idx = {iid: i for i, iid in enumerate(unique_items)}
        self.idx2item = {i: iid for iid, i in self.item2idx.items()}

        user_indices = [self.user2idx[u] for u in df_interactions[user_col].to_list()]
        item_indices = [self.item2idx[i] for i in df_interactions[item_col].to_list()]
        ratings = df_interactions[rating_col].cast(pl.Float32).to_numpy()

        self.global_mean = float(np.mean(ratings))
        
        n_users = len(unique_users)
        n_items = len(unique_items)

        # Create sparse User-Item matrix
        R = csr_matrix((ratings, (user_indices, item_indices)), shape=(n_users, n_items), dtype=np.float32)

        # Fit SVD on User-Item matrix
        # Item factors: components_ of shape (n_factors, n_items) -> transpose to (n_items, n_factors)
        self.user_factors = self.svd.fit_transform(R)
        self.item_factors = self.svd.components_.T  # Shape: (n_items, n_factors)

        explained_variance_ratio = float(np.sum(self.svd.explained_variance_ratio_))

        stats = {
            "n_users": n_users,
            "n_items": n_items,
            "n_interactions": len(ratings),
            "global_mean_rating": self.global_mean,
            "explained_variance_sum": explained_variance_ratio,
        }
        return R, stats

    def predict_score(self, user_id: str, item_id: str) -> float:
        """
        Predict affinity score for a given user and item.
        """
        if user_id not in self.user2idx or item_id not in self.item2idx:
            return self.global_mean

        u_idx = self.user2idx[user_id]
        i_idx = self.item2idx[item_id]

        score = float(np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))
        return score

    def recommend_top_k(
        self,
        user_id: str,
        user_interacted_items: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Generate Top-K item recommendations for a given user.
        """
        if user_id not in self.user2idx:
            # Fallback for Cold-Start user: return empty or handled by Content-Based
            return []

        u_idx = self.user2idx[user_id]
        u_vector = self.user_factors[u_idx]

        # Compute dot product across all items: (1, n_factors) @ (n_factors, n_items) -> (n_items,)
        scores = np.dot(u_vector, self.item_factors.T)

        interacted_indices = set()
        if user_interacted_items:
            interacted_indices = {self.item2idx[item] for item in user_interacted_items if item in self.item2idx}

        # Mask out interacted items
        if interacted_indices:
            scores[list(interacted_indices)] = -np.inf

        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [(self.idx2item[i], float(scores[i])) for i in top_indices]

    def evaluate(self, df_test: pl.DataFrame, user_col: str = "user_id", item_col: str = "parent_asin", rating_col: str = "rating") -> Dict[str, float]:
        """
        Compute RMSE and MAE on test split.
        """
        y_true = []
        y_pred = []

        for row in df_test.iter_rows(named=True):
            u = row[user_col]
            i = row[item_col]
            actual = row[rating_col]
            pred = self.predict_score(u, i)

            y_true.append(actual)
            y_pred.append(pred)

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
        mae = float(mean_absolute_error(y_true_arr, y_pred_arr))

        return {"rmse": round(rmse, 4), "mae": round(mae, 4)}

    def save_model(self, output_dir: str = "models/collaborative") -> None:
        """
        Persist trained model artifacts.
        """
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self, os.path.join(output_dir, "svd_recommender.joblib"))
        print(f"[+] SVD model saved to {output_dir}/svd_recommender.joblib")

    @classmethod
    def load_model(cls, model_path: str = "models/collaborative/svd_recommender.joblib") -> "SVDRecommender":
        """
        Load trained model from disk.
        """
        return joblib.load(model_path)

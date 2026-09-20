"""
User & Item Behavioral Clustering using K-Means on latent factors and interaction features.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os


class CollaborativeClusterer:
    """
    Performs clustering on User and Item latent factors to identify game genres/archetypes
    and user personas.
    """

    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        self.scaler = StandardScaler()

    def fit_item_clusters(self, item_factors: np.ndarray, item_ids: List[str]) -> pl.DataFrame:
        """
        Cluster items based on their CF latent factors.
        """
        scaled_factors = self.scaler.fit_transform(item_factors)
        cluster_labels = self.kmeans.fit_predict(scaled_factors)

        return pl.DataFrame({
            "parent_asin": item_ids,
            "cf_cluster_id": cluster_labels
        })

    def save(self, path: str = "models/collaborative/item_clusterer.joblib") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

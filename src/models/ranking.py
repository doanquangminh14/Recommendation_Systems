"""
Diversity & Maximal Marginal Relevance (MMR) Ranking Module.

Provides algorithms to balance recommendation relevance and item diversity,
mitigating filter bubble effects and popularity bias.
"""

from typing import List, Dict, Optional, Tuple, Any, Union
import os
import numpy as np
import polars as pl
from dataclasses import dataclass

from src.models.hybrid.hybrid_engine import HybridRecommendationResult


@dataclass
class DiversityMetrics:
    """
    Data container for diversity evaluation metrics on a recommended list.
    """
    intra_list_diversity: float
    unique_categories_count: int
    category_distribution: Dict[str, int]
    mean_relevance_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intra_list_diversity": round(self.intra_list_diversity, 4),
            "unique_categories_count": self.unique_categories_count,
            "category_distribution": self.category_distribution,
            "mean_relevance_score": round(self.mean_relevance_score, 4),
        }


class DiversityRanker:
    """
    Re-ranks recommendation candidate pools using Maximal Marginal Relevance (MMR)
    and computes diversity metrics such as Intra-List Diversity (ILD) and Category Coverage.
    """

    def __init__(
        self,
        embeddings: Optional[np.ndarray] = None,
        item2idx: Optional[Dict[str, int]] = None,
        embeddings_path: str = "data/gold/item_embeddings.npy",
        items_path: str = "data/silver/item_features.parquet",
    ):
        """
        Initialize the ranker with dense embeddings and item index mapping.
        """
        if embeddings is not None and item2idx is not None:
            self.embeddings = embeddings
            self.item2idx = item2idx
        else:
            if not os.path.exists(embeddings_path):
                raise FileNotFoundError(f"Embeddings matrix not found at '{embeddings_path}'.")
            if not os.path.exists(items_path):
                raise FileNotFoundError(f"Item features dataset not found at '{items_path}'.")

            self.embeddings = np.load(embeddings_path)
            df_items = pl.read_parquet(items_path, columns=["parent_asin"])
            item_ids = df_items["parent_asin"].to_list()
            self.item2idx = {iid: i for i, iid in enumerate(item_ids)}

    def calculate_intra_list_diversity(self, item_ids: List[str]) -> float:
        """
        Calculate Intra-List Diversity (ILD) as the average pairwise cosine distance (1 - similarity).

        ILD(S) = (2 / (|S|(|S|-1))) * sum_{i < j} (1 - cos(e_i, e_j))

        Parameters
        ----------
        item_ids : list of str
            List of parent_asin identifiers in the recommended list.

        Returns
        -------
        float
            ILD score between 0.0 (identical items) and 2.0 (maximally diverse).
        """
        if len(item_ids) <= 1:
            return 0.0

        valid_indices = [self.item2idx[iid] for iid in item_ids if iid in self.item2idx]
        if len(valid_indices) <= 1:
            return 0.0

        vecs = self.embeddings[valid_indices]  # Shape: (K, 384)
        sim_matrix = np.dot(vecs, vecs.T)  # Pairwise Cosine Similarity matrix (K, K)
        
        K = len(valid_indices)
        distances = 1.0 - sim_matrix
        upper_tri = distances[np.triu_indices(K, k=1)]
        
        return float(np.mean(upper_tri))

    def calculate_diversity_metrics(
        self,
        items: Union[List[HybridRecommendationResult], List[Dict[str, Any]]],
        score_key: str = "hybrid_score"
    ) -> DiversityMetrics:
        """
        Compute comprehensive diversity and coverage metrics for a list of items.
        """
        if not items:
            return DiversityMetrics(
                intra_list_diversity=0.0,
                unique_categories_count=0,
                category_distribution={},
                mean_relevance_score=0.0,
            )

        item_ids: List[str] = []
        categories: List[str] = []
        scores: List[float] = []

        for it in items:
            if isinstance(it, HybridRecommendationResult):
                item_ids.append(it.parent_asin)
                categories.append(it.category)
                scores.append(it.hybrid_score)
            else:
                item_ids.append(it.get("parent_asin", ""))
                categories.append(it.get("category", "Unknown"))
                scores.append(float(it.get(score_key, 0.0)))

        ild = self.calculate_intra_list_diversity(item_ids)
        cat_dist: Dict[str, int] = {}
        for c in categories:
            cat_dist[c] = cat_dist.get(c, 0) + 1

        mean_score = float(np.mean(scores)) if scores else 0.0

        return DiversityMetrics(
            intra_list_diversity=ild,
            unique_categories_count=len(cat_dist),
            category_distribution=cat_dist,
            mean_relevance_score=mean_score,
        )

    def apply_mmr(
        self,
        candidate_items: Union[List[HybridRecommendationResult], List[Dict[str, Any]]],
        lambda_param: float = 0.65,
        top_k: int = 10,
        score_key: str = "hybrid_score"
    ) -> Union[List[HybridRecommendationResult], List[Dict[str, Any]]]:
        """
        Re-rank candidate items using Maximal Marginal Relevance (MMR).

        MMR(i) = argmax_{i in R \\ S} [ lambda * Score(i) - (1 - lambda) * max_{j in S} sim(e_i, e_j) ]

        Parameters
        ----------
        candidate_items : list of HybridRecommendationResult or list of dict
            Pre-ranked candidate items from hybrid engine.
        lambda_param : float
            Trade-off parameter:
            - 1.0 = Pure relevance (Original Hybrid Ranking).
            - 0.6 ~ 0.7 = Optimal balance between relevance and semantic diversity.
            - 0.0 = Maximal diversity / novelty.
        top_k : int
            Number of items to select after re-ranking.
        score_key : str
            Field name for relevance score when items are dicts.

        Returns
        -------
        list
            Re-ranked Top-K diverse items of the same type as input.
        """
        if not candidate_items:
            return []

        candidates = list(candidate_items)
        selected: List[Any] = []
        selected_indices: List[int] = []

        def get_asin(it: Any) -> str:
            return it.parent_asin if isinstance(it, HybridRecommendationResult) else it.get("parent_asin", "")

        def get_score(it: Any) -> float:
            return it.hybrid_score if isinstance(it, HybridRecommendationResult) else float(it.get(score_key, 0.0))

        # Select first item having highest relevance score
        first_item = candidates.pop(0)
        selected.append(first_item)
        first_asin = get_asin(first_item)
        if first_asin in self.item2idx:
            selected_indices.append(self.item2idx[first_asin])

        while len(selected) < top_k and candidates:
            if not selected_indices:
                # Fallback if embeddings missing
                selected.append(candidates.pop(0))
                continue

            selected_vecs = self.embeddings[selected_indices]  # Shape: (|S|, 384)
            best_mmr_score = -np.inf
            best_cand_idx = 0

            for c_idx, cand in enumerate(candidates):
                cand_asin = get_asin(cand)
                rel_score = get_score(cand)

                if cand_asin in self.item2idx:
                    cand_vec = self.embeddings[self.item2idx[cand_asin]]
                    # Compute max cosine similarity against all already selected items
                    max_sim_to_selected = float(np.max(np.dot(selected_vecs, cand_vec)))
                else:
                    max_sim_to_selected = 0.0

                mmr_val = (lambda_param * rel_score) - ((1.0 - lambda_param) * max_sim_to_selected)

                if mmr_val > best_mmr_score:
                    best_mmr_score = mmr_val
                    best_cand_idx = c_idx

            best_cand = candidates.pop(best_cand_idx)
            selected.append(best_cand)
            best_asin = get_asin(best_cand)
            if best_asin in self.item2idx:
                selected_indices.append(self.item2idx[best_asin])

        return selected

    def evaluate_diversity_tradeoff(
        self,
        candidate_items: Union[List[HybridRecommendationResult], List[Dict[str, Any]]],
        lambda_values: Optional[List[float]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Evaluate and compare Relevance vs Diversity metrics across different lambda settings.
        """
        if lambda_values is None:
            lambda_values = [1.0, 0.85, 0.70, 0.65, 0.50, 0.30, 0.0]

        tradeoff_results = []
        for lmb in lambda_values:
            reranked = self.apply_mmr(candidate_items=candidate_items, lambda_param=lmb, top_k=top_k)
            metrics = self.calculate_diversity_metrics(reranked)
            tradeoff_results.append({
                "lambda": round(lmb, 2),
                "ild_diversity": metrics.intra_list_diversity,
                "mean_relevance": metrics.mean_relevance_score,
                "unique_categories": metrics.unique_categories_count,
            })

        return tradeoff_results

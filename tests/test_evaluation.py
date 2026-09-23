"""
Unit tests for evaluation metrics module (src/models/evaluation.py).
"""

import unittest
import numpy as np

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


class TestEvaluationMetrics(unittest.TestCase):
    """Test suite for recommender systems evaluation metrics."""

    def test_rmse_and_mae(self):
        y_true = [4.0, 5.0, 3.0, 2.0]
        y_pred = [4.0, 4.0, 3.0, 1.0]

        rmse = compute_rmse(y_true, y_pred)
        mae = compute_mae(y_true, y_pred)

        # differences: [0, 1, 0, 1], squared: [0, 1, 0, 1], mean = 0.5, sqrt = ~0.7071
        self.assertAlmostEqual(rmse, np.sqrt(0.5), places=4)
        self.assertAlmostEqual(mae, 0.5, places=4)

    def test_precision_recall_hit_rate(self):
        actual = {"game_1", "game_2", "game_3"}
        predicted = ["game_1", "game_4", "game_2", "game_5", "game_6"]

        # Top-3: ["game_1", "game_4", "game_2"] -> 2 hits
        prec_3 = compute_precision_at_k(actual, predicted, k=3)
        rec_3 = compute_recall_at_k(actual, predicted, k=3)
        hr_3 = compute_hit_rate_at_k(actual, predicted, k=3)

        self.assertAlmostEqual(prec_3, 2 / 3, places=4)
        self.assertAlmostEqual(rec_3, 2 / 3, places=4)
        self.assertEqual(hr_3, 1.0)

        # Empty actual check
        self.assertEqual(compute_recall_at_k([], predicted, k=5), 0.0)
        self.assertEqual(compute_hit_rate_at_k([], predicted, k=5), 0.0)

    def test_ndcg_at_k(self):
        actual = {"A", "B"}
        
        # Perfect ranking: A, B at positions 1, 2
        perfect_pred = ["A", "B", "C", "D"]
        self.assertAlmostEqual(compute_ndcg_at_k(actual, perfect_pred, k=4), 1.0, places=4)

        # Imperfect ranking: hits at positions 2, 4
        imperfect_pred = ["C", "A", "D", "B"]
        ndcg = compute_ndcg_at_k(actual, imperfect_pred, k=4)
        self.assertTrue(0.0 < ndcg < 1.0)

    def test_intra_list_diversity(self):
        embeddings = np.array([
            [1.0, 0.0],  # item 0
            [0.0, 1.0],  # item 1 (orthogonal -> distance 1.0)
            [1.0, 0.0],  # item 2 (identical to item 0 -> distance 0.0)
        ], dtype=np.float32)
        item2idx = {"item_0": 0, "item_1": 1, "item_2": 2}

        # Items 0 & 1 -> orthogonal, cosine distance = 1.0
        ild_diff = compute_intra_list_diversity(["item_0", "item_1"], embeddings, item2idx)
        self.assertAlmostEqual(ild_diff, 1.0, places=4)

        # Items 0 & 2 -> identical, cosine distance = 0.0
        ild_same = compute_intra_list_diversity(["item_0", "item_2"], embeddings, item2idx)
        self.assertAlmostEqual(ild_same, 0.0, places=4)

    def test_catalog_coverage(self):
        rec_items = {"g1", "g2", "g3"}
        cov = compute_catalog_coverage(rec_items, total_catalog_count=10)
        self.assertAlmostEqual(cov, 0.3, places=4)

    def test_evaluate_ranking_predictions(self):
        user_actual = {
            "u1": ["g1", "g2"],
            "u2": ["g3"],
        }
        user_pred = {
            "u1": ["g1", "g4", "g2"],
            "u2": ["g5", "g6", "g3"],
        }
        summary = evaluate_ranking_predictions(user_actual, user_pred, k_values=[3])
        self.assertIn("precision@3", summary)
        self.assertIn("recall@3", summary)
        self.assertIn("ndcg@3", summary)
        self.assertIn("hit_rate@3", summary)
        self.assertGreater(summary["precision@3"], 0.0)


if __name__ == "__main__":
    unittest.main()

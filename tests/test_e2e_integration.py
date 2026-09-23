"""
Comprehensive End-to-End (E2E) Integration Test Suite.

Validates the full system lifecycle:
Silver Layer -> Gold Vectors -> Hybrid ML Models -> Explainer & MMR Ranker -> AI Agent -> FastAPI -> UI Components.
"""

import os
import sys
import unittest
import numpy as np
import polars as pl

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.agent_runner import GameAgentRunner
from src.ui.components import (
    render_rating_stars,
    render_badge,
)


class TestEndToEndSystem(unittest.TestCase):
    """Full End-to-End System Integration Tests."""

    @classmethod
    def setUpClass(cls):
        """Warm up the complete AI Agent Runner and ML engines once."""
        print("\n[E2E] Initializing GameAgentRunner with real project data...")
        cls.runner = GameAgentRunner(
            embeddings_path="data/gold/item_embeddings.npy",
            items_path="data/silver/item_features.parquet",
            svd_model_path="models/collaborative/svd_recommender.joblib",
            sentiment_path="data/silver/item_sentiment.parquet",
            reviews_path="data/silver/review_sentiment.parquet",
            interactions_path="data/silver/interactions.parquet",
        )
        cls.test_user_id = "AHJRJCJMK3XVV4BSPBRAHIYEODWA"  # Active user with 473 interactions

    def test_01_silver_and_gold_data_integrity(self):
        """Verify presence and dimensionality of Silver Parquet files and Gold vector embeddings."""
        self.assertTrue(os.path.exists("data/silver/interactions.parquet"))
        self.assertTrue(os.path.exists("data/silver/item_features.parquet"))
        self.assertTrue(os.path.exists("data/silver/item_images.parquet"))
        self.assertTrue(os.path.exists("data/gold/item_embeddings.npy"))

        df_items = pl.read_parquet("data/silver/item_features.parquet")
        self.assertEqual(len(df_items), 25612)

        embeddings = np.load("data/gold/item_embeddings.npy")
        self.assertEqual(embeddings.shape, (25612, 384))

    def test_02_personalized_hybrid_recommendation(self):
        """Verify Personalized Hybrid Recommendation with MMR Diversity and Explainability."""
        rec_res = self.runner.recommend_tool.run(
            user_id=self.test_user_id,
            top_k=5,
            diversity_weight=0.30,
            include_explanation=True,
        )
        self.assertEqual(rec_res.status, "success")
        self.assertGreater(len(rec_res.recommendations), 0)
        self.assertLessEqual(len(rec_res.recommendations), 5)

        # Check top item attributes
        top_item = rec_res.recommendations[0]
        self.assertTrue(len(top_item.parent_asin) > 0)
        self.assertTrue(len(top_item.title) > 0)
        self.assertGreater(top_item.hybrid_score, 0.0)

        # Check explanation attached
        self.assertIsNotNone(top_item.explanation)
        self.assertTrue("reasons" in top_item.explanation or "key_reasons" in top_item.explanation)

    def test_03_zero_shot_semantic_cold_start(self):
        """Verify Semantic Natural Language Search via MiniLM-L6 vector embeddings."""
        query = "Open-world dark fantasy RPG with atmospheric lore"
        sem_res = self.runner.recommend_tool.run(
            query=query,
            top_k=4,
            include_explanation=True,
        )
        self.assertEqual(sem_res.status, "success")
        self.assertGreater(len(sem_res.recommendations), 0)

    def test_04_item_to_item_similarity(self):
        """Verify Item-to-Item content-based similarity calculation."""
        sample_asin = self.runner.recommend_tool.cb_model.item_ids[0]
        sim_res = self.runner.recommend_tool.hybrid_engine.recommend(
            liked_item_ids=[sample_asin],
            top_k=3,
        )
        self.assertGreater(len(sim_res.items), 0)

    def test_05_gamer_persona_analytics(self):
        """Verify Gamer Persona detection, rating histogram, and category profiling."""
        analytics = self.runner.analytics_tool.run(user_id=self.test_user_id)
        self.assertEqual(analytics.user_id, self.test_user_id)
        self.assertGreater(analytics.total_interactions, 0)
        self.assertTrue(len(analytics.gamer_persona) > 0)
        self.assertTrue(len(analytics.favorite_categories) > 0)

    def test_06_recommendation_explainer(self):
        """Verify multi-signal reasoning and social proof review quote extraction."""
        sample_asin = self.runner.recommend_tool.cb_model.item_ids[0]
        expl = self.runner.explain_tool.run(
            item_asin_or_title=sample_asin,
            user_id=self.test_user_id,
        )
        self.assertEqual(expl.parent_asin, sample_asin)
        self.assertGreater(len(expl.key_reasons), 0)

    def test_07_ai_agent_multi_turn_dialogue(self):
        """Verify multi-turn session memory and intent classification in AgentRunner."""
        session_id = "test_e2e_session"
        self.runner.reset_session(session_id)

        # Turn 1: Casual greeting
        t1 = self.runner.run_dialogue(
            message="Xin chào!",
            user_id=self.test_user_id,
            session_id=session_id,
        )
        self.assertEqual(t1.intent.upper(), "CHAT")
        self.assertTrue(len(t1.response_text) > 0)

        # Turn 2: Recommendation query
        t2 = self.runner.run_dialogue(
            message="Gợi ý game hay cho tôi",
            user_id=self.test_user_id,
            session_id=session_id,
        )
        self.assertEqual(t2.intent.upper(), "RECOMMEND")
        self.assertEqual(t2.tool_used, "RecommendTool")
        self.assertIsNotNone(t2.tool_output)

        # Turn 3: Analytics query
        t3 = self.runner.run_dialogue(
            message="Phân tích lịch sử chơi game của tôi",
            user_id=self.test_user_id,
            session_id=session_id,
        )
        self.assertEqual(t3.intent.upper(), "ANALYZE")
        self.assertEqual(t3.tool_used, "AnalyticsTool")

    def test_08_ui_components_rendering(self):
        """Verify that UI formatting helpers render valid HTML."""
        stars_html = render_rating_stars(4.8)
        self.assertIn("★", stars_html)
        self.assertIn("4.8", stars_html)

        badge_html = render_badge("Action", "cyan", "🎮")
        self.assertIn("badge-cyan", badge_html)


if __name__ == "__main__":
    unittest.main()

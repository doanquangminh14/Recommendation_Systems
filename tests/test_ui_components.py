"""
Unit and Integration Tests for Streamlit UI Components (src/ui/components.py).
"""

import unittest
from unittest.mock import MagicMock, patch
from src.ui.components import (
    apply_custom_css,
    render_rating_stars,
    render_badge,
    render_header,
    render_metric_card,
    render_game_card,
    render_games_grid,
    render_explanation_panel,
    render_gamer_persona_card,
    render_chat_message,
    render_empty_state,
)
from src.api.schemas import RecommendedGameItem, GameExplanationDetail, UserAnalyticsResponse


class TestUIComponents(unittest.TestCase):

    def test_render_rating_stars(self):
        stars_html = render_rating_stars(4.5)
        self.assertIn("4.5", stars_html)
        self.assertIn("★", stars_html)

        stars_zero = render_rating_stars(0.0)
        self.assertIn("0.0", stars_zero)

    def test_render_badge(self):
        badge_html = render_badge("Action", color_type="cyan", icon="🗡️")
        self.assertIn("badge-cyan", badge_html)
        self.assertIn("Action", badge_html)
        self.assertIn("🗡️", badge_html)

    @patch("streamlit.markdown")
    def test_render_header(self, mock_st_markdown):
        render_header("My Title", "My Subtitle", stats={"total_games": 25612, "status": "Online"})
        self.assertTrue(mock_st_markdown.called)
        args, _ = mock_st_markdown.call_args
        self.assertIn("My Title", args[0])
        self.assertIn("25,612 Games Active", args[0])

    @patch("streamlit.markdown")
    def test_render_metric_card(self, mock_st_markdown):
        render_metric_card("ILD Diversity", 0.85, subtitle="Intra-list diversity", icon="🎯", delta="+5%")
        self.assertTrue(mock_st_markdown.called)
        args, _ = mock_st_markdown.call_args
        self.assertIn("ILD Diversity", args[0])
        self.assertIn("0.85", args[0])

    @patch("streamlit.markdown")
    def test_render_game_card_dict(self, mock_st_markdown):
        game_dict = {
            "parent_asin": "B0000001",
            "title": "Super Mario Odyssey",
            "category": "Platformer",
            "avg_rating": 4.8,
            "rating_number": 1250,
            "image_url": "https://example.com/poster.jpg",
            "price": 59.99,
            "hybrid_score": 0.92,
            "cf_score": 0.88,
            "cb_score": 0.95,
            "sentiment_score": 0.85,
            "explanation": {
                "anchor_game": "Super Mario Galaxy",
                "anchor_similarity_pct": 91.5,
                "key_reasons": ["Rich 3D exploration", "Highly reviewed platformer"],
                "social_proof_quote": "One of the best Mario games ever created!",
            },
        }
        render_game_card(game_dict, show_explanation=True, rank=1)
        self.assertTrue(mock_st_markdown.called)
        args, _ = mock_st_markdown.call_args
        self.assertIn("Super Mario Odyssey", args[0])
        self.assertIn("Super Mario Galaxy", args[0])
        self.assertIn("92%", args[0])

    @patch("streamlit.markdown")
    def test_render_game_card_dto(self, mock_st_markdown):
        game_dto = RecommendedGameItem(
            parent_asin="B0000002",
            title="The Legend of Zelda: Breath of the Wild",
            category="Action-Adventure",
            avg_rating=4.9,
            rating_number=5400,
            image_url="https://example.com/botw.jpg",
            price=59.99,
            hybrid_score=0.96,
            cf_score=0.92,
            cb_score=0.98,
            sentiment_score=0.90,
            rank=1,
            explanation=GameExplanationDetail(
                anchor_game="Elden Ring",
                anchor_similarity_pct=88.0,
                key_reasons=["Vast open world", "Award-winning gameplay"],
                social_proof_quote="Masterpiece of game design.",
            ),
        )
        render_game_card(game_dto, show_explanation=True)
        self.assertTrue(mock_st_markdown.called)
        args, _ = mock_st_markdown.call_args
        self.assertIn("The Legend of Zelda", args[0])
        self.assertIn("Elden Ring", args[0])

    @patch("streamlit.columns")
    def test_render_games_grid(self, mock_st_columns):
        mock_cols = [MagicMock(), MagicMock(), MagicMock()]
        mock_st_columns.return_value = mock_cols

        games = [
            {"parent_asin": f"B000000{i}", "title": f"Game {i}", "avg_rating": 4.5}
            for i in range(6)
        ]
        render_games_grid(games, num_columns=3)
        self.assertEqual(mock_st_columns.call_count, 2)

    @patch("streamlit.markdown")
    def test_render_explanation_panel(self, mock_st_markdown):
        explanation_data = {
            "title": "God of War Ragnarok",
            "anchor_game": "God of War (2018)",
            "anchor_similarity_pct": 95.0,
            "key_reasons": ["Epic combat", "Immersive storyline"],
            "social_proof_quote": "Incredible narrative and stunning visuals.",
        }
        render_explanation_panel(explanation_data)
        self.assertTrue(mock_st_markdown.called)
        args, _ = mock_st_markdown.call_args
        self.assertIn("God of War Ragnarok", args[0])
        self.assertIn("95.0%", args[0])

    @patch("streamlit.markdown")
    def test_render_gamer_persona_card(self, mock_st_markdown):
        analytics = UserAnalyticsResponse(
            user_id="A100TESTUSER",
            total_interactions=42,
            average_rating=4.76,
            gamer_persona="RPG & Strategy Master",
            top_categories=["RPG", "Strategy", "Adventure"],
            rating_distribution={5: 35, 4: 5, 3: 2, 2: 0, 1: 0},
        )
        render_gamer_persona_card(analytics)
        self.assertTrue(mock_st_markdown.called)
        args, _ = mock_st_markdown.call_args
        self.assertTrue("RPG &amp; Strategy Master" in args[0] or "RPG & Strategy Master" in args[0])
        self.assertIn("A100TESTUSER", args[0])

    @patch("streamlit.markdown")
    def test_render_chat_message(self, mock_st_markdown):
        render_chat_message(role="user", content="Recommend me some RPG games")
        self.assertTrue(mock_st_markdown.called)

        render_chat_message(
            role="assistant",
            content="Here are top RPG games for you!",
            intent="recommend",
            tool_used="RecommendTool",
        )
        args, _ = mock_st_markdown.call_args
        self.assertIn("AI Gaming Concierge", args[0])
        self.assertIn("RecommendTool", args[0])


if __name__ == "__main__":
    unittest.main()

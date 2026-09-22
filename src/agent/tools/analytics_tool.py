"""
User Profile & Behavioral Analytics Tool for AI Agent.

Aggregates interaction history, computes rating distributions, identifies favorite genres,
and infers personalized Gamer Persona profiles.
"""

from typing import List, Dict, Optional, Any, Union
import os
import polars as pl
from dataclasses import dataclass, field

from src.models.content_based.recommender import ContentBasedRecommender


@dataclass
class UserInteractionRecord:
    """
    Data container for a single historical game interaction of a user.
    """
    parent_asin: str
    title: str
    category: str
    user_rating: float
    timestamp: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_asin": self.parent_asin,
            "title": self.title,
            "category": self.category,
            "user_rating": round(self.user_rating, 2),
            "timestamp": self.timestamp,
        }


@dataclass
class AnalyticsToolOutput:
    """
    Data container for user profile analytics results.
    """
    status: str
    user_id: str
    total_interactions: int = 0
    average_rating: float = 0.0
    rating_distribution: Dict[int, int] = field(default_factory=dict)
    favorite_categories: Dict[str, int] = field(default_factory=dict)
    top_categories: List[str] = field(default_factory=list)
    gamer_persona: str = "General Gamer"
    favorite_games: List[UserInteractionRecord] = field(default_factory=list)
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        if self.status == "not_found":
            return {
                "status": "not_found",
                "user_id": self.user_id,
                "message": self.message or f"User '{self.user_id}' has no interaction history.",
            }
        return {
            "status": "success",
            "user_id": self.user_id,
            "total_interactions": self.total_interactions,
            "average_rating": round(self.average_rating, 2),
            "rating_distribution": self.rating_distribution,
            "favorite_categories": self.favorite_categories,
            "top_categories": self.top_categories,
            "gamer_persona": self.gamer_persona,
            "favorite_games": [g.to_dict() for g in self.favorite_games],
        }


class AnalyticsTool:
    """
    Agent Tool for extracting comprehensive user statistics, interaction history, and gamer persona.
    """

    name: str = "analyze_user_profile"
    description: str = (
        "Phân tích chi tiết lịch sử chơi game, phân bố điểm đánh giá (1-5 sao), "
        "các thể loại yêu thích nhất, và xây dựng Gamer Persona của người dùng."
    )

    def __init__(
        self,
        df_interactions: Optional[pl.DataFrame] = None,
        cb_model: Optional[ContentBasedRecommender] = None,
        interactions_path: str = "data/silver/interactions.parquet",
        items_path: str = "data/silver/item_features.parquet",
        embeddings_path: str = "data/gold/item_embeddings.npy",
    ):
        """
        Initialize the AnalyticsTool with pre-loaded interactions and metadata models.
        """
        if df_interactions is not None:
            self.df_interactions = df_interactions
        elif os.path.exists(interactions_path):
            self.df_interactions = pl.read_parquet(interactions_path)
        else:
            self.df_interactions = pl.DataFrame()

        if cb_model is not None:
            self.cb_model = cb_model
        else:
            self.cb_model = ContentBasedRecommender(
                embeddings_path=embeddings_path,
                items_path=items_path,
            )

    @staticmethod
    def _infer_gamer_persona(sorted_categories: List[tuple], avg_rating: float) -> str:
        """
        Infer a descriptive gamer persona title from primary genres and rating behavior.
        """
        if not sorted_categories:
            return "Casual Gaming Enthusiast"

        primary_cat = sorted_categories[0][0].lower()
        if any(w in primary_cat for w in ["role playing", "rpg", "adventure"]):
            persona = "RPG Enthusiast & Lore Explorer"
        elif any(w in primary_cat for w in ["action", "shooter", "fps"]):
            persona = "Action & High-Adrenaline Fighter"
        elif any(w in primary_cat for w in ["strategy", "simulation", "builder"]):
            persona = "Tactical Strategist & World Builder"
        elif any(w in primary_cat for w in ["retro", "classic", "arcade"]):
            persona = "Retro Gaming Collector"
        elif any(w in primary_cat for w in ["sports", "racing"]):
            persona = "Competitive Esports & Sports Fanatic"
        else:
            persona = f"{sorted_categories[0][0]} Explorer"

        if avg_rating >= 4.5:
            persona += " (Critical Master)"
        elif avg_rating <= 3.0:
            persona += " (Demanding Critic)"

        return persona

    def run(self, user_id: str) -> AnalyticsToolOutput:
        """
        Execute user behavioral analysis and extract gamer persona.

        Parameters
        ----------
        user_id : str
            Target user identifier.

        Returns
        -------
        AnalyticsToolOutput
            Structured gamer profile analysis output.
        """
        if not user_id or not user_id.strip():
            return AnalyticsToolOutput(
                status="error",
                user_id="",
                message="Mã người dùng (user_id) không được để trống.",
            )

        clean_uid = user_id.strip()
        user_df = self.df_interactions.filter(pl.col("user_id") == clean_uid)

        if user_df.is_empty():
            return AnalyticsToolOutput(
                status="not_found",
                user_id=clean_uid,
                message=f"Không tìm thấy lịch sử tương tác cho người dùng '{clean_uid}'. Đây là Cold-Start User.",
            )

        total_reviews = len(user_df)
        avg_rating = float(user_df.select(pl.col("rating").mean()).item())

        # Rating distribution breakdown (1-5 stars)
        rating_counts = user_df.group_by("rating").len().sort("rating", descending=True)
        rating_dist = {
            int(row["rating"]): int(row["len"])
            for row in rating_counts.iter_rows(named=True)
        }

        # Item interaction history & metadata resolution
        item_asins = user_df.select("parent_asin").to_series().to_list()
        ratings = user_df.select("rating").to_series().to_list()
        timestamps = (
            user_df.select("timestamp").to_series().to_list()
            if "timestamp" in user_df.columns
            else [None] * len(item_asins)
        )

        history_items: List[UserInteractionRecord] = []
        category_counts: Dict[str, int] = {}

        for asin, rating, ts in zip(item_asins, ratings, timestamps):
            title = self.cb_model.item_titles.get(asin, "Unknown Title")
            cat = self.cb_model.item_categories.get(asin, "Video Games")

            category_counts[cat] = category_counts.get(cat, 0) + 1
            history_items.append(
                UserInteractionRecord(
                    parent_asin=asin,
                    title=title,
                    category=cat,
                    user_rating=float(rating),
                    timestamp=ts,
                )
            )

        # Ranked categories
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        top_categories = [cat for cat, _ in sorted_categories[:3]]

        # Infer Gamer Persona
        persona = self._infer_gamer_persona(sorted_categories, avg_rating)

        # Top 5 highest rated games in user history
        top_rated_games = sorted(history_items, key=lambda x: x.user_rating, reverse=True)[:5]

        return AnalyticsToolOutput(
            status="success",
            user_id=clean_uid,
            total_interactions=total_reviews,
            average_rating=avg_rating,
            rating_distribution=rating_dist,
            favorite_categories=dict(sorted_categories[:5]),
            top_categories=top_categories,
            gamer_persona=persona,
            favorite_games=top_rated_games,
        )

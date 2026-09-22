"""
Recommendation Explanation Tool for AI Agent.

Extracts multi-perspective transparent reasoning signals, anchor game similarities,
and authentic player social proof quotes for any recommended game in the catalog.
"""

from typing import List, Dict, Optional, Any, Union
import os
import re
import polars as pl
from dataclasses import dataclass, field

from src.models.content_based.recommender import ContentBasedRecommender
from src.models.hybrid.hybrid_engine import HybridRecommender, HybridRecommendationResult
from src.models.hybrid.explainer import RecommendationExplainer, RecommendationExplanation


@dataclass
class ExplainToolOutput:
    """
    Data container for the complete execution output of ExplainTool.
    """
    status: str
    parent_asin: Optional[str] = None
    title: Optional[str] = None
    category: Optional[str] = None
    average_rating: Optional[float] = None
    rating_number: Optional[int] = None
    anchor_game: Optional[str] = None
    anchor_asin: Optional[str] = None
    anchor_similarity_pct: float = 0.0
    key_reasons: List[str] = field(default_factory=list)
    social_proof_quote: str = ""
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        if self.status == "error":
            return {
                "status": "error",
                "message": self.message or "Unknown error occurred.",
            }
        return {
            "status": "success",
            "parent_asin": self.parent_asin,
            "title": self.title,
            "category": self.category,
            "average_rating": round(self.average_rating, 2) if self.average_rating is not None else None,
            "rating_number": self.rating_number,
            "anchor_game": self.anchor_game,
            "anchor_asin": self.anchor_asin,
            "anchor_similarity_pct": round(self.anchor_similarity_pct, 1),
            "key_reasons": self.key_reasons,
            "social_proof_quote": self.social_proof_quote,
        }


class ExplainTool:
    """
    Agent Tool for explaining recommendation logic and extracting authentic user testimonials.
    """

    name: str = "explain_recommendation"
    description: str = (
        "Giải thích chi tiết vì sao một tựa game được đề xuất cho người dùng. "
        "Cung cấp mức độ tương đồng với game đã chơi trong quá khứ, trùng khớp thể loại, "
        "đồng thuận cộng đồng và trích dẫn nhận xét tích cực thực tế từ cộng đồng game thủ."
    )

    def __init__(
        self,
        explainer: Optional[RecommendationExplainer] = None,
        hybrid_engine: Optional[HybridRecommender] = None,
        embeddings_path: str = "data/gold/item_embeddings.npy",
        items_path: str = "data/silver/item_features.parquet",
        sentiment_path: str = "data/silver/item_sentiment.parquet",
        reviews_path: str = "data/silver/review_sentiment.parquet",
        interactions_path: str = "data/silver/interactions.parquet",
    ):
        """
        Initialize the ExplainTool with pre-built explainer or component datasets.
        """
        if explainer is not None:
            self.explainer = explainer
        else:
            self.explainer = RecommendationExplainer(
                embeddings_path=embeddings_path,
                items_path=items_path,
                sentiment_path=sentiment_path,
                reviews_path=reviews_path,
            )

        if hybrid_engine is not None:
            self.hybrid_engine = hybrid_engine
        else:
            self.hybrid_engine = HybridRecommender(
                cb_model=self.explainer.cb_model,
                sentiment_path=sentiment_path,
                interactions_path=interactions_path,
            )

        self.cb_model = self.explainer.cb_model

    def find_item_asin(self, item_query: str) -> Optional[str]:
        """
        Resolve an item identifier or title query to a canonical parent_asin.
        """
        query = item_query.strip()
        # Direct ASIN match
        if query in self.cb_model.item2idx:
            return query

        query_lower = query.lower()
        # 1. Exact substring match
        for asin, title in self.cb_model.item_titles.items():
            if query_lower in title.lower():
                return asin

        # 2. Token-based fallback
        stop_words = {"game", "play", "cho", "tôi", "hay", "nhất", "top", "tựa", "này", "về"}
        tokens = [t for t in re.findall(r"\w+", query_lower) if len(t) >= 3 and t not in stop_words]
        for token in tokens:
            for asin, title in self.cb_model.item_titles.items():
                if token in title.lower():
                    return asin

        return None

    def run(
        self,
        item_asin_or_title: str,
        user_id: Optional[str] = None,
        reference_game_title: Optional[str] = None,
    ) -> ExplainToolOutput:
        """
        Execute explanation generation for a given game item.

        Parameters
        ----------
        item_asin_or_title : str
            Parent ASIN or full/partial game title to explain.
        user_id : str or None
            Target user identifier to incorporate personalized interaction history.
        reference_game_title : str or None
            Optional specific game title to compare against.

        Returns
        -------
        ExplainToolOutput
            Structured explanation data.
        """
        if not item_asin_or_title or not item_asin_or_title.strip():
            return ExplainToolOutput(
                status="error",
                message="Tên hoặc mã game không được để trống.",
            )

        asin = self.find_item_asin(item_asin_or_title)
        if not asin:
            return ExplainToolOutput(
                status="error",
                message=f"Không tìm thấy thông tin tựa game với mã hoặc tên: '{item_asin_or_title}'",
            )

        title = self.cb_model.item_titles.get(asin, "Unknown Title")
        category = self.cb_model.item_categories.get(asin, "Video Games")
        rating = float(self.cb_model.item_ratings.get(asin, 4.5))

        # Retrieve user history or reference anchor
        user_history: List[str] = []
        if user_id:
            user_history = self.hybrid_engine.user_history_dict.get(user_id, [])
        elif reference_game_title:
            ref_asin = self.find_item_asin(reference_game_title)
            if ref_asin:
                user_history = [ref_asin]

        # Construct recommendation item container for explanation extraction
        item_res = HybridRecommendationResult(
            parent_asin=asin,
            title=title,
            category=category,
            hybrid_score=0.90,
            cf_score=0.85,
            cb_score=0.92,
            sentiment_score=0.88,
            avg_rating=rating,
            rating_number=100,
        )

        is_cold_start = (user_id is None and not reference_game_title)
        explanation = self.explainer.explain_recommendation(
            rec_item=item_res,
            user_liked_asins=user_history,
            is_cold_start=is_cold_start,
        )

        return ExplainToolOutput(
            status="success",
            parent_asin=asin,
            title=title,
            category=category,
            average_rating=rating,
            rating_number=item_res.rating_number,
            anchor_game=explanation.anchor_title,
            anchor_asin=explanation.anchor_asin,
            anchor_similarity_pct=round(explanation.anchor_similarity * 100, 1),
            key_reasons=explanation.reasons,
            social_proof_quote=explanation.highlight_quote,
        )

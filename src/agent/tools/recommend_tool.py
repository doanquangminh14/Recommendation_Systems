"""
Smart Game Recommendation Tool for AI Agent.

Combines Multi-Modal Hybrid Filtering (Collaborative Filtering + Semantic Text Embeddings + VADER Sentiment)
with Maximal Marginal Relevance (MMR) Diversity Re-Ranking and Multi-Signal Explanations.
Supports both personalized recommendations for existing users and zero-shot semantic query recommendations
for cold-start users.
"""

from typing import List, Dict, Optional, Any, Union
import os
import re
import numpy as np
import polars as pl
from dataclasses import dataclass, field

from src.models.content_based.recommender import ContentBasedRecommender
from src.models.collaborative.matrix_factorization import SVDRecommender
from src.models.hybrid.hybrid_engine import HybridRecommender, HybridRecommendationResult
from src.models.hybrid.explainer import RecommendationExplainer, RecommendationExplanation
from src.models.ranking import DiversityRanker


@dataclass
class RecommendToolItem:
    """
    Data container for an individual recommended game item returned by RecommendTool.
    """
    parent_asin: str
    title: str
    category: str
    avg_rating: float
    rating_number: int
    hybrid_score: float
    cf_score: float
    cb_score: float
    sentiment_score: float
    explanation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "parent_asin": self.parent_asin,
            "title": self.title,
            "category": self.category,
            "avg_rating": round(self.avg_rating, 2),
            "rating_number": self.rating_number,
            "hybrid_score": round(self.hybrid_score, 4),
            "cf_score": round(self.cf_score, 4),
            "cb_score": round(self.cb_score, 4),
            "sentiment_score": round(self.sentiment_score, 4),
        }
        if self.explanation is not None:
            data["explanation"] = self.explanation
        return data


@dataclass
class RecommendToolOutput:
    """
    Data container for the complete execution output of RecommendTool.
    """
    status: str
    mode: str
    recommendations: List[RecommendToolItem]
    user_id: Optional[str] = None
    query: Optional[str] = None
    matched_anchor: Optional[str] = None
    category_filter: Optional[str] = None
    total_recommendations: int = 0
    cf_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "mode": self.mode,
            "user_id": self.user_id,
            "query": self.query,
            "matched_anchor": self.matched_anchor,
            "category_filter": self.category_filter,
            "total_recommendations": len(self.recommendations),
            "cf_available": self.cf_available,
            "metadata": self.metadata,
            "recommendations": [item.to_dict() for item in self.recommendations],
        }


class RecommendTool:
    """
    Agent Recommender Tool combining Hybrid Recommender, MMR Diversity Re-ranking, and Explainer.
    """

    name: str = "recommend_games"
    description: str = (
        "Gợi ý danh sách game phù hợp nhất dựa trên mã người dùng (user_id), "
        "hoặc tìm kiếm ngữ nghĩa theo từ khóa/thể loại/tựa game yêu thích. "
        "Tự động áp dụng MMR Re-ranking để tối ưu hóa sự cân bằng giữa độ liên quan và tính đa dạng danh mục."
    )

    def __init__(
        self,
        hybrid_engine: Optional[HybridRecommender] = None,
        ranker: Optional[DiversityRanker] = None,
        explainer: Optional[RecommendationExplainer] = None,
        embeddings_path: str = "data/gold/item_embeddings.npy",
        items_path: str = "data/silver/item_features.parquet",
        svd_model_path: str = "models/collaborative/svd_recommender.joblib",
        sentiment_path: str = "data/silver/item_sentiment.parquet",
        reviews_path: str = "data/silver/review_sentiment.parquet",
        interactions_path: str = "data/silver/interactions.parquet",
    ):
        """
        Initialize the RecommendTool with pre-built models or dataset file paths.
        """
        # 1. Hybrid Engine
        if hybrid_engine is not None:
            self.hybrid_engine = hybrid_engine
        else:
            self.hybrid_engine = HybridRecommender(
                embeddings_path=embeddings_path,
                items_path=items_path,
                svd_model_path=svd_model_path,
                sentiment_path=sentiment_path,
                interactions_path=interactions_path,
            )

        # 2. Diversity Ranker
        if ranker is not None:
            self.ranker = ranker
        else:
            self.ranker = DiversityRanker(
                embeddings=self.hybrid_engine.cb_model.embeddings,
                item2idx=self.hybrid_engine.cb_model.item2idx,
            )

        # 3. Recommendation Explainer
        if explainer is not None:
            self.explainer = explainer
        else:
            self.explainer = RecommendationExplainer(
                cb_model=self.hybrid_engine.cb_model,
                sentiment_path=sentiment_path,
                reviews_path=reviews_path,
            )

        self.cb_model = self.hybrid_engine.cb_model

    def find_items_by_keyword(self, keyword: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for games matching a query string using exact substring and token fallback.
        """
        kw = keyword.lower().strip()
        matches = []

        # 1. Exact substring match in title
        for asin, title in self.cb_model.item_titles.items():
            if kw in title.lower():
                matches.append({
                    "parent_asin": asin,
                    "title": title,
                    "category": self.cb_model.item_categories.get(asin, "Video Games"),
                })
                if len(matches) >= top_k:
                    return matches

        # 2. Token-based fallback for multi-word queries
        stop_words = {"game", "play", "cho", "tôi", "hay", "nhất", "top", "muốn", "tìm", "thích", "về", "các", "những"}
        tokens = [t for t in re.findall(r"\w+", kw) if len(t) >= 3 and t not in stop_words]

        for token in tokens:
            for asin, title in self.cb_model.item_titles.items():
                if token in title.lower() and not any(m["parent_asin"] == asin for m in matches):
                    matches.append({
                        "parent_asin": asin,
                        "title": title,
                        "category": self.cb_model.item_categories.get(asin, "Video Games"),
                    })
                    if len(matches) >= top_k:
                        return matches

        return matches

    def run(
        self,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 5,
        diversity_weight: float = 0.30,
        include_explanation: bool = True,
    ) -> RecommendToolOutput:
        """
        Execute smart game recommendations based on user identifier or natural language query.

        Parameters
        ----------
        user_id : str or None
            Target user identifier for personalized hybrid recommendations.
        query : str or None
            Keywords / game title for cold-start semantic search.
        category : str or None
            Optional filter to restrict recommendations to a specific genre/category.
        top_k : int
            Number of top recommended games to return (default: 5).
        diversity_weight : float
            Weight for diversity in MMR re-ranking between 0.0 (pure relevance) and 1.0 (pure diversity).
        include_explanation : bool
            Whether to attach multi-faceted explanation metadata to each result item.

        Returns
        -------
        RecommendToolOutput
            Structured recommendation output.
        """
        lambda_param = max(0.0, min(1.0, 1.0 - diversity_weight))

        # -------------------------------------------------------------
        # 1. Cold-Start Semantic / Keyword Search Flow
        # -------------------------------------------------------------
        if query and not user_id:
            matched_items = self.find_items_by_keyword(query, top_k=3)
            if matched_items:
                anchor_asin = matched_items[0]["parent_asin"]
                anchor_title = matched_items[0]["title"]
                raw_recs = self.hybrid_engine.recommend(
                    liked_item_ids=[anchor_asin],
                    top_k=max(top_k * 3, 20),
                    category_filter=category,
                )
            else:
                raw_recs = self.hybrid_engine.recommend(
                    liked_item_ids=None,
                    top_k=max(top_k * 3, 20),
                    category_filter=category,
                )
                anchor_asin, anchor_title = None, None

            # Apply MMR Diversity Re-ranking
            ranked_items = self.ranker.apply_mmr(
                raw_recs.items,
                lambda_param=lambda_param,
                top_k=top_k,
            )

            results: List[RecommendToolItem] = []
            for item in ranked_items:
                exp_dict = None
                if include_explanation:
                    exp = self.explainer.explain_recommendation(
                        item,
                        user_liked_asins=[anchor_asin] if anchor_asin else [],
                        is_cold_start=True,
                    )
                    exp_dict = exp.to_dict()

                results.append(
                    RecommendToolItem(
                        parent_asin=item.parent_asin,
                        title=item.title,
                        category=item.category,
                        avg_rating=item.avg_rating,
                        rating_number=item.rating_number,
                        hybrid_score=item.hybrid_score,
                        cf_score=item.cf_score,
                        cb_score=item.cb_score,
                        sentiment_score=item.sentiment_score,
                        explanation=exp_dict,
                    )
                )

            return RecommendToolOutput(
                status="success",
                mode="semantic_query",
                user_id=None,
                query=query,
                matched_anchor=anchor_title,
                category_filter=category,
                total_recommendations=len(results),
                cf_available=False,
                recommendations=results,
                metadata={"effective_weights": raw_recs.effective_weights},
            )

        # -------------------------------------------------------------
        # 2. Personalized Hybrid Recommendation Flow (User ID)
        # -------------------------------------------------------------
        elif user_id:
            user_history = self.hybrid_engine.user_history_dict.get(user_id, [])
            raw_output = self.hybrid_engine.recommend(
                user_id=user_id,
                top_k=max(top_k * 3, 20),
                category_filter=category,
            )

            # Apply MMR Diversity Re-ranking
            ranked_items = self.ranker.apply_mmr(
                raw_output.items,
                lambda_param=lambda_param,
                top_k=top_k,
            )

            results = []
            for item in ranked_items:
                exp_dict = None
                if include_explanation:
                    exp = self.explainer.explain_recommendation(
                        item,
                        user_liked_asins=user_history,
                        is_cold_start=not raw_output.cf_available,
                    )
                    exp_dict = exp.to_dict()

                results.append(
                    RecommendToolItem(
                        parent_asin=item.parent_asin,
                        title=item.title,
                        category=item.category,
                        avg_rating=item.avg_rating,
                        rating_number=item.rating_number,
                        hybrid_score=item.hybrid_score,
                        cf_score=item.cf_score,
                        cb_score=item.cb_score,
                        sentiment_score=item.sentiment_score,
                        explanation=exp_dict,
                    )
                )

            return RecommendToolOutput(
                status="success",
                mode="personalized_user",
                user_id=user_id,
                query=None,
                matched_anchor=None,
                category_filter=category,
                total_recommendations=len(results),
                cf_available=raw_output.cf_available,
                recommendations=results,
                metadata={
                    "user_history_count": len(user_history),
                    "effective_weights": raw_output.effective_weights,
                },
            )

        # -------------------------------------------------------------
        # 3. Global Popular / High-Sentiment Fallback Flow
        # -------------------------------------------------------------
        else:
            raw_output = self.hybrid_engine.recommend(
                top_k=top_k,
                category_filter=category,
            )
            results = [
                RecommendToolItem(
                    parent_asin=item.parent_asin,
                    title=item.title,
                    category=item.category,
                    avg_rating=item.avg_rating,
                    rating_number=item.rating_number,
                    hybrid_score=item.hybrid_score,
                    cf_score=item.cf_score,
                    cb_score=item.cb_score,
                    sentiment_score=item.sentiment_score,
                )
                for item in raw_output.items
            ]

            return RecommendToolOutput(
                status="success",
                mode="popular_top_rated",
                user_id=None,
                query=None,
                matched_anchor=None,
                category_filter=category,
                total_recommendations=len(results),
                cf_available=False,
                recommendations=results,
                metadata={"effective_weights": raw_output.effective_weights},
            )

"""
Multi-Faceted Recommendation Explainer Module.

Extracts human-understandable explanation signals and real player social proof quotes
for recommended items, explaining:
1. Anchor Item Similarity (Semantic Content alignment with user's past favorite games).
2. Thematic / Genre Overlap.
3. Collaborative Community Consensus (Gu đồng điệu).
4. Sentiment Metrics & Social Proof Quotes (Authentic positive player reviews).
"""

from typing import List, Dict, Optional, Tuple, Any, Union
import os
import re
import html
import numpy as np
import polars as pl
from dataclasses import dataclass, field

from src.models.content_based.recommender import ContentBasedRecommender
from src.models.hybrid.hybrid_engine import HybridRecommendationResult


@dataclass
class RecommendationExplanation:
    """
    Data container for explanation signals of a single recommended item.
    """
    parent_asin: str
    title: str
    hybrid_score: float
    reasons: List[str]
    highlight_quote: str
    anchor_asin: Optional[str] = None
    anchor_title: Optional[str] = None
    anchor_similarity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_asin": self.parent_asin,
            "title": self.title,
            "hybrid_score": round(self.hybrid_score, 4),
            "reasons": self.reasons,
            "highlight_quote": self.highlight_quote,
            "anchor_asin": self.anchor_asin,
            "anchor_title": self.anchor_title,
            "anchor_similarity": round(self.anchor_similarity, 4),
        }


class RecommendationExplainer:
    """
    Generates transparent, multi-perspective explanations and authentic user review highlights.
    """

    DEFAULT_QUOTE = "Cộng đồng game thủ đánh giá cao lối chơi cuốn hút, đồ họa ấn tượng và trải nghiệm xuất sắc."

    def __init__(
        self,
        cb_model: Optional[ContentBasedRecommender] = None,
        df_sentiment: Optional[pl.DataFrame] = None,
        df_reviews: Optional[pl.DataFrame] = None,
        embeddings_path: str = "data/gold/item_embeddings.npy",
        items_path: str = "data/silver/item_features.parquet",
        sentiment_path: str = "data/silver/item_sentiment.parquet",
        reviews_path: str = "data/silver/review_sentiment.parquet",
    ):
        """
        Initialize the explainer with models and review datasets.
        """
        # 1. Content-Based Model / Embeddings
        if cb_model is not None:
            self.cb_model = cb_model
        else:
            self.cb_model = ContentBasedRecommender(
                embeddings_path=embeddings_path,
                items_path=items_path,
            )

        # 2. Sentiment Metrics
        if df_sentiment is not None:
            self.df_sentiment = df_sentiment
        elif os.path.exists(sentiment_path):
            self.df_sentiment = pl.read_parquet(sentiment_path)
        else:
            self.df_sentiment = pl.DataFrame()

        # Build sentiment lookup dictionary
        self.sent_dict: Dict[str, Dict[str, Any]] = {}
        if not self.df_sentiment.is_empty():
            for row in self.df_sentiment.iter_rows(named=True):
                self.sent_dict[row["parent_asin"]] = row

        # 3. Authentic Review Text Quotes
        if df_reviews is not None:
            self.df_reviews = df_reviews
        elif os.path.exists(reviews_path):
            # Read relevant columns to save memory
            self.df_reviews = pl.read_parquet(
                reviews_path,
                columns=["parent_asin", "title", "text", "sentiment_label", "sentiment_compound"]
            )
        else:
            self.df_reviews = pl.DataFrame()

        # Cache high-quality positive quotes per item
        self._init_quote_cache()

    def _init_quote_cache(self) -> None:
        """
        Index positive reviews per game ASIN for instant social proof quote retrieval.
        """
        self.cached_quotes: Dict[str, str] = {}
        if self.df_reviews.is_empty():
            return

        # Filter positive reviews with high compound score
        pos_reviews = (
            self.df_reviews
            .filter((pl.col("sentiment_label") == "positive") & (pl.col("sentiment_compound") >= 0.50))
            .sort("sentiment_compound", descending=True)
        )

        for row in pos_reviews.iter_rows(named=True):
            asin = row["parent_asin"]
            if asin in self.cached_quotes:
                continue

            text = row.get("text") or ""
            title = row.get("title") or ""
            full_text = f"{title}: {text}" if title else text
            
            # Clean text: remove HTML tags, unescape entities, collapse whitespaces
            clean = html.unescape(full_text)
            clean = re.sub(r"<[^>]+>", " ", clean)
            clean = re.sub(r"\s+", " ", clean).strip()

            if 25 <= len(clean) <= 180:
                self.cached_quotes[asin] = clean

    def find_anchor_game(
        self,
        rec_asin: str,
        user_liked_asins: List[str]
    ) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Find the anchor game in user's history with highest cosine similarity to target game.

        Returns
        -------
        tuple : (anchor_asin, similarity_score, anchor_title)
        """
        if not user_liked_asins or rec_asin not in self.cb_model.item2idx:
            return None, 0.0, None

        rec_idx = self.cb_model.item2idx[rec_asin]
        rec_vec = self.cb_model.embeddings[rec_idx]

        best_asin = None
        best_sim = -1.0

        for past_asin in user_liked_asins:
            if past_asin in self.cb_model.item2idx and past_asin != rec_asin:
                past_idx = self.cb_model.item2idx[past_asin]
                sim = float(np.dot(rec_vec, self.cb_model.embeddings[past_idx]))
                if sim > best_sim:
                    best_sim = sim
                    best_asin = past_asin

        if best_asin is not None and best_sim > 0.0:
            anchor_title = self.cb_model.item_titles.get(best_asin, "một tựa game bạn từng chơi")
            return best_asin, best_sim, anchor_title

        return None, 0.0, None

    def get_social_proof_quote(self, rec_asin: str) -> str:
        """
        Retrieve cached authentic player quote or fallback message.
        """
        return self.cached_quotes.get(rec_asin, self.DEFAULT_QUOTE)

    def explain_recommendation(
        self,
        rec_item: Union[HybridRecommendationResult, Dict[str, Any]],
        user_liked_asins: Optional[List[str]] = None,
        is_cold_start: bool = False,
    ) -> RecommendationExplanation:
        """
        Generate comprehensive explanation signals for a single recommendation.

        Parameters
        ----------
        rec_item : HybridRecommendationResult or dict
            The recommendation item to explain.
        user_liked_asins : list of str or None
            List of game ASINs the user has interacted with or explicitly liked.
        is_cold_start : bool
            Whether the recommendation was generated under cold-start conditions.

        Returns
        -------
        RecommendationExplanation
            Structured explanation object.
        """
        if isinstance(rec_item, HybridRecommendationResult):
            rec_asin = rec_item.parent_asin
            rec_title = rec_item.title
            rec_category = rec_item.category
            rec_score = rec_item.hybrid_score
            rec_cf_score = rec_item.cf_score
            rec_avg_rating = rec_item.avg_rating
        else:
            rec_asin = rec_item.get("parent_asin", "")
            rec_title = rec_item.get("title", "Unknown Title")
            rec_category = rec_item.get("category", "")
            rec_score = float(rec_item.get("hybrid_score", 0.0))
            rec_cf_score = float(rec_item.get("cf_score", 0.0))
            rec_avg_rating = float(rec_item.get("avg_rating", 0.0))

        reasons: List[str] = []
        user_liked = user_liked_asins or []

        # 1. Anchor Game Semantic Similarity
        anchor_asin, anchor_sim, anchor_title = self.find_anchor_game(rec_asin, user_liked)
        if anchor_asin and anchor_sim >= 0.40:
            reasons.append(
                f"🎯 Tương đồng {anchor_sim * 100:.1f}% về nội dung & lối chơi với game bạn yêu thích: '{anchor_title}'"
            )

        # 2. Genre / Category Overlap
        if rec_category and rec_category not in ["Unknown", "Unknown Category"]:
            reasons.append(f"🎮 Trùng khớp thể loại phù hợp: {rec_category}")

        # 3. Collaborative Community Consensus
        if not is_cold_start and rec_cf_score >= 0.40:
            reasons.append("👥 Được cộng đồng người chơi có cùng gu sở thích với bạn đánh giá rất cao")

        # 4. Sentiment & Community Score
        if rec_asin in self.sent_dict:
            s_info = self.sent_dict[rec_asin]
            pos_pct = float(s_info.get("positive_review_ratio", 0.0) or 0.0) * 100
            compound = float(s_info.get("avg_sentiment_compound", 0.0) or 0.0)
            sign = "+" if compound >= 0 else ""
            reasons.append(
                f"⭐ {pos_pct:.1f}% đánh giá tích cực trên toàn cộng đồng (Điểm cảm xúc: {sign}{compound:.2f})"
            )
        elif rec_avg_rating > 0:
            reasons.append(f"⭐ Điểm đánh giá trung bình từ người chơi: {rec_avg_rating:.1f}/5.0")

        # 5. Authentic Social Proof Quote
        quote = self.get_social_proof_quote(rec_asin)

        return RecommendationExplanation(
            parent_asin=rec_asin,
            title=rec_title,
            hybrid_score=rec_score,
            reasons=reasons,
            highlight_quote=quote,
            anchor_asin=anchor_asin,
            anchor_title=anchor_title,
            anchor_similarity=anchor_sim,
        )

    def explain_batch(
        self,
        recommendations: Union[List[HybridRecommendationResult], List[Dict[str, Any]]],
        user_liked_asins: Optional[List[str]] = None,
        is_cold_start: bool = False,
    ) -> List[RecommendationExplanation]:
        """
        Generate explanations for a batch of recommendations.
        """
        return [
            self.explain_recommendation(
                rec_item=item,
                user_liked_asins=user_liked_asins,
                is_cold_start=is_cold_start,
            )
            for item in recommendations
        ]

"""
Sentiment Analysis Module for Video Game Reviews.

Provides high-performance sentiment score extraction (using VADER sentiment analyzer)
and item-level sentiment aggregation for hybrid recommendation and explanation signals.
"""

from typing import Optional, Dict, Any, List
import polars as pl
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import os


class ReviewSentimentAnalyzer:
    """
    Analyzes sentiment in review text using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    Optimized for batch processing on Polars DataFrames.
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(self, text: Optional[str]) -> Dict[str, float]:
        """
        Calculate VADER polarity scores for a single text.

        Parameters
        ----------
        text : str or None
            Input review text.

        Returns
        -------
        dict
            Contains 'compound', 'pos', 'neg', 'neu'.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        return self.analyzer.polarity_scores(text)

    def classify_compound_score(self, compound: float, pos_threshold: float = 0.05, neg_threshold: float = -0.05) -> str:
        """
        Classify sentiment into 'positive', 'neutral', or 'negative' based on compound score.
        """
        if compound >= pos_threshold:
            return "positive"
        elif compound <= neg_threshold:
            return "negative"
        else:
            return "neutral"

    def process_reviews_dataframe(
        self,
        df: pl.DataFrame,
        text_column: str = "text",
        title_column: Optional[str] = "title",
        batch_size: int = 20000,
        sample_size: Optional[int] = None
    ) -> pl.DataFrame:
        """
        Extract sentiment scores for all reviews in a Polars DataFrame.
        Combines title and text if title is provided for richer sentiment signals.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame containing review interactions and text.
        text_column : str
            Column name containing review text.
        title_column : str or None
            Column name containing review title/summary.
        batch_size : int
            Batch size for iteration.
        sample_size : int or None
            Optional sample size for quick evaluation.

        Returns
        -------
        pl.DataFrame
            DataFrame augmented with 'sentiment_compound', 'sentiment_pos', 'sentiment_neg', 'sentiment_label'.
        """
        if sample_size and sample_size < len(df):
            df_target = df.sample(n=sample_size, seed=42)
        else:
            df_target = df

        # Prepare combined text for sentiment if title exists
        if title_column and title_column in df_target.columns:
            combined_texts = [
                f"{t or ''}. {x or ''}".strip(". ")
                for t, x in zip(df_target[title_column].to_list(), df_target[text_column].to_list())
            ]
        else:
            combined_texts = [x or "" for x in df_target[text_column].to_list()]

        compounds: List[float] = []
        pos_scores: List[float] = []
        neg_scores: List[float] = []
        neu_scores: List[float] = []
        labels: List[str] = []

        for text in tqdm(combined_texts, desc="Processing Sentiment", total=len(combined_texts)):
            if not text:
                compounds.append(0.0)
                pos_scores.append(0.0)
                neg_scores.append(0.0)
                neu_scores.append(1.0)
                labels.append("neutral")
            else:
                scores = self.analyzer.polarity_scores(text)
                c = scores["compound"]
                compounds.append(round(c, 4))
                pos_scores.append(round(scores["pos"], 4))
                neg_scores.append(round(scores["neg"], 4))
                neu_scores.append(round(scores["neu"], 4))
                labels.append(self.classify_compound_score(c))

        return df_target.with_columns([
            pl.Series("sentiment_compound", compounds, dtype=pl.Float32),
            pl.Series("sentiment_pos", pos_scores, dtype=pl.Float32),
            pl.Series("sentiment_neg", neg_scores, dtype=pl.Float32),
            pl.Series("sentiment_neu", neu_scores, dtype=pl.Float32),
            pl.Series("sentiment_label", labels, dtype=pl.String),
        ])

    def aggregate_item_sentiments(self, df_with_sentiment: pl.DataFrame) -> pl.DataFrame:
        """
        Aggregate sentiment statistics per item (`parent_asin`) to produce item sentiment profiles.

        Parameters
        ----------
        df_with_sentiment : pl.DataFrame
            DataFrame containing 'parent_asin', 'sentiment_compound', 'sentiment_label', 'rating'.

        Returns
        -------
        pl.DataFrame
            Item-level sentiment features (mean_compound, pos_ratio, neg_ratio, sentiment_consistency).
        """
        item_sentiment = (
            df_with_sentiment.group_by("parent_asin")
            .agg([
                pl.len().alias("review_count"),
                pl.col("sentiment_compound").mean().round(4).alias("avg_sentiment_compound"),
                pl.col("sentiment_compound").std().round(4).alias("std_sentiment_compound"),
                (pl.col("sentiment_label") == "positive").mean().round(4).alias("positive_review_ratio"),
                (pl.col("sentiment_label") == "negative").mean().round(4).alias("negative_review_ratio"),
                (pl.col("sentiment_label") == "neutral").mean().round(4).alias("neutral_review_ratio"),
                pl.col("rating").mean().round(3).alias("avg_user_rating"),
            ])
            .sort("review_count", descending=True)
        )
        return item_sentiment


def run_sentiment_pipeline(
    review_path: str = "data/silver/review_text.parquet",
    output_review_sentiment_path: str = "data/silver/review_sentiment.parquet",
    output_item_sentiment_path: str = "data/silver/item_sentiment.parquet",
    sample_size: Optional[int] = None
) -> None:
    """
    End-to-end execution of Sentiment Analysis on Silver review dataset.
    """
    print(f"[*] Loading review text dataset from {review_path}...")
    df_reviews = pl.read_parquet(review_path)
    print(f"    Loaded {len(df_reviews):,} reviews.")

    analyzer = ReviewSentimentAnalyzer()
    print(f"[*] Extracting sentiment scores (sample_size={sample_size})...")
    df_scored = analyzer.process_reviews_dataframe(df_reviews, sample_size=sample_size)

    print(f"[*] Aggregating item-level sentiment profiles...")
    df_item_sentiment = analyzer.aggregate_item_sentiments(df_scored)

    # Save to Parquet
    os.makedirs(os.path.dirname(output_review_sentiment_path), exist_ok=True)
    print(f"[*] Saving review sentiment to {output_review_sentiment_path}...")
    df_scored.write_parquet(output_review_sentiment_path, compression="zstd")

    print(f"[*] Saving item sentiment profiles to {output_item_sentiment_path}...")
    df_item_sentiment.write_parquet(output_item_sentiment_path, compression="zstd")

    print(f"[+] Sentiment analysis pipeline completed successfully!")
    print(f"    - Processed reviews: {len(df_scored):,}")
    print(f"    - Unique items with sentiment: {len(df_item_sentiment):,}")


if __name__ == "__main__":
    # Test run on sample or full dataset
    run_sentiment_pipeline(sample_size=50000)

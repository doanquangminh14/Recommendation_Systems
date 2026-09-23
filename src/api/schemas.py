"""
Pydantic Schemas and Data Transfer Objects (DTOs) for the Video Games Recommender API.

Provides robust validation, type hinting, default values, and serialization
for all API endpoints (Recommendations, Explanation, Analytics, AI Agent, Catalog Search, System Health).
"""

from typing import List, Dict, Optional, Any, Union
from enum import Enum
import datetime
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# Enums
# ============================================================================

class ResponseStatusEnum(str, Enum):
    """Standardized API response statuses."""
    SUCCESS = "success"
    ERROR = "error"
    NOT_FOUND = "not_found"


class RecommendationStrategyEnum(str, Enum):
    """Supported recommendation strategies."""
    HYBRID = "hybrid"
    CONTENT_BASED = "content_based"
    COLLABORATIVE = "collaborative"
    SEMANTIC_COLD_START = "semantic_cold_start"
    SIMILAR_ITEM = "similar_item"
    POPULARITY = "popularity"


class GamerPersonaEnum(str, Enum):
    """Standardized Gamer Personas detected by Analytics."""
    ACTION_ADVENTURE = "Action-Adventure Enthusiast"
    RPG_MASTER = "RPG & Strategy Master"
    RETRO_CLASSIC = "Retro & Classic Collector"
    CASUAL_FAMILY = "Casual & Family Gamer"
    SHOOTER_PRO = "Shooter & Competitive Gamer"
    SIMULATION_SPORTS = "Simulation & Sports Fan"
    GENERAL = "General Gamer"


# ============================================================================
# Base & Utility Models
# ============================================================================

class BaseAPIResponse(BaseModel):
    """Base response model with timestamp and status."""
    status: ResponseStatusEnum = Field(default=ResponseStatusEnum.SUCCESS, description="Status of API operation")
    message: Optional[str] = Field(default=None, description="Optional informational or error message")
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description="Response generation timestamp"
    )

    model_config = ConfigDict(extra="ignore")


class ErrorResponse(BaseModel):
    """Detailed error payload returned on API exceptions."""
    status: ResponseStatusEnum = Field(default=ResponseStatusEnum.ERROR, description="Always 'error'")
    error_code: str = Field(..., description="Short error code identifier (e.g., USER_NOT_FOUND)")
    message: str = Field(..., description="Human-readable error description")
    detail: Optional[Any] = Field(default=None, description="Additional context or validation traceback")
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


# ============================================================================
# Game & Item Models
# ============================================================================

class GameExplanationDetail(BaseModel):
    """Structured reasoning and social proof signals explaining why a game is recommended."""
    anchor_game: Optional[str] = Field(default=None, description="Title of the high-affinity game in user history that anchored this suggestion")
    anchor_asin: Optional[str] = Field(default=None, description="ASIN of the anchor game")
    anchor_similarity_pct: float = Field(default=0.0, description="Semantic similarity percentage to the anchor game (0-100%)")
    key_reasons: List[str] = Field(default_factory=list, description="Bullet points explaining why this recommendation fits")
    social_proof_quote: str = Field(default="", description="Exemplary quote extracted from authentic player reviews")
    category_overlap: Optional[bool] = Field(default=None, description="Whether the game shares categories with user favorites")
    cf_consensus: Optional[bool] = Field(default=None, description="Whether users with similar taste also favored this game")


class GameItemBase(BaseModel):
    """Core game catalog representation."""
    parent_asin: str = Field(..., description="Unique Amazon Standard Identification Number (ASIN)")
    title: str = Field(..., description="Cleaned title of the video game")
    category: str = Field(..., description="Main or sub-genre category")
    avg_rating: float = Field(default=0.0, description="Global average star rating (1.0 - 5.0)")
    rating_number: int = Field(default=0, description="Total number of ratings received")
    image_url: Optional[str] = Field(default=None, description="Direct URL to high-resolution game poster/cover image")
    price: Optional[float] = Field(default=None, description="Game price if available in metadata")


class RecommendedGameItem(GameItemBase):
    """Detailed recommendation item including multi-modal scores and explanation."""
    hybrid_score: float = Field(default=0.0, description="Weighted composite recommendation score")
    cf_score: float = Field(default=0.0, description="Collaborative filtering predicted score")
    cb_score: float = Field(default=0.0, description="Semantic content-based affinity score")
    sentiment_score: float = Field(default=0.0, description="Aggregated player review sentiment score")
    rank: Optional[int] = Field(default=None, description="Position in top-K result list")
    explanation: Optional[GameExplanationDetail] = Field(default=None, description="Transparent multi-signal explanation")


class GameItemDetail(GameItemBase):
    """Comprehensive game metadata for item detail view."""
    description: Optional[str] = Field(default=None, description="Long-form game description")
    features: Optional[List[str]] = Field(default_factory=list, description="Bullet points of key gameplay features")
    store: Optional[str] = Field(default=None, description="Store or publisher name")
    bought_together: Optional[List[str]] = Field(default_factory=list, description="List of frequently bought together ASINs")
    avg_sentiment_compound: Optional[float] = Field(default=None, description="Mean VADER compound sentiment score (-1.0 to +1.0)")
    positive_review_ratio: Optional[float] = Field(default=None, description="Fraction of reviews with positive sentiment (0.0 to 1.0)")


# ============================================================================
# Recommendation Endpoints DTOs
# ============================================================================

class PersonalizedRecommendRequest(BaseModel):
    """Request payload for user personalized hybrid recommendation."""
    user_id: str = Field(..., min_length=1, description="Amazon User ID (e.g. 'A100WO06OIG7KW')")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of games to recommend (1 to 50)")
    use_mmr: bool = Field(default=True, description="Whether to apply Maximal Marginal Relevance for diversity")
    diversity_lambda: float = Field(default=0.7, ge=0.0, le=1.0, description="MMR trade-off factor (1.0 = pure relevance, 0.0 = pure diversity)")
    filter_category: Optional[str] = Field(default=None, description="Optional category filter (case-insensitive)")
    min_rating: Optional[float] = Field(default=None, ge=1.0, le=5.0, description="Minimum average star rating threshold")
    include_explanations: bool = Field(default=True, description="Whether to compute explainability signals for each item")


class SemanticRecommendRequest(BaseModel):
    """Request payload for zero-shot natural language game search & recommendation."""
    query: str = Field(..., min_length=1, description="Gaming preference or theme query (e.g. 'open-world soulslike RPG with dark atmosphere')")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of items to recommend")
    filter_category: Optional[str] = Field(default=None, description="Optional category filter")
    min_rating: Optional[float] = Field(default=None, ge=1.0, le=5.0, description="Minimum average star rating threshold")


class SimilarGamesRequest(BaseModel):
    """Request payload for Item-to-Item content-based similarity recommendation."""
    item_id: str = Field(..., min_length=1, description="Target game parent ASIN")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of similar games to find")
    filter_category: Optional[str] = Field(default=None, description="Optional category filter")


class RecommendResponse(BaseModel):
    """Standardized response payload for all recommendation endpoints."""
    status: ResponseStatusEnum = Field(default=ResponseStatusEnum.SUCCESS)
    strategy: RecommendationStrategyEnum = Field(..., description="The algorithm strategy used to generate results")
    user_id: Optional[str] = Field(default=None, description="User ID if personalized")
    query: Optional[str] = Field(default=None, description="Search query if semantic search")
    anchor_item_id: Optional[str] = Field(default=None, description="Anchor ASIN if item-to-item search")
    count: int = Field(default=0, description="Total items returned in recommendations list")
    recommendations: List[RecommendedGameItem] = Field(default_factory=list, description="Ranked list of recommended games")
    diversity_ild: Optional[float] = Field(default=None, description="Intra-List Diversity score of recommended list (0.0 to 1.0)")
    execution_time_ms: Optional[float] = Field(default=None, description="Inference execution time in milliseconds")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ============================================================================
# Explanation Endpoints DTOs
# ============================================================================

class ExplainGameRequest(BaseModel):
    """Request payload to explain why a specific game fits a user."""
    user_id: str = Field(..., min_length=1, description="Target User ID")
    item_id: str = Field(..., min_length=1, description="Target Game ASIN to explain")


class ExplainGameResponse(BaseModel):
    """Response payload containing full explanation for a specific user and game."""
    status: ResponseStatusEnum = Field(default=ResponseStatusEnum.SUCCESS)
    user_id: Optional[str] = Field(default=None)
    item_id: Optional[str] = Field(default=None)
    title: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)
    average_rating: Optional[float] = Field(default=None)
    rating_number: Optional[int] = Field(default=None)
    anchor_game: Optional[str] = Field(default=None)
    anchor_asin: Optional[str] = Field(default=None)
    anchor_similarity_pct: float = Field(default=0.0)
    key_reasons: List[str] = Field(default_factory=list)
    social_proof_quote: str = Field(default="")
    message: Optional[str] = Field(default=None)
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ============================================================================
# User Analytics Endpoints DTOs
# ============================================================================

class UserInteractionDTO(BaseModel):
    """Historical interaction record of a user."""
    parent_asin: str = Field(..., description="Interacted game ASIN")
    title: str = Field(..., description="Game title")
    category: str = Field(..., description="Game category")
    user_rating: float = Field(..., description="Rating given by the user")
    timestamp: Optional[int] = Field(default=None, description="Unix timestamp of review")


class UserAnalyticsRequest(BaseModel):
    """Request payload to fetch user profile analytics."""
    user_id: str = Field(..., min_length=1, description="Target User ID")
    limit_favorites: int = Field(default=10, ge=1, le=50, description="Max favorite games to return")


class UserAnalyticsResponse(BaseModel):
    """Response payload with behavioral statistics and gamer persona."""
    status: ResponseStatusEnum = Field(default=ResponseStatusEnum.SUCCESS)
    user_id: str = Field(..., description="Queried User ID")
    total_interactions: int = Field(default=0, description="Total reviews submitted by user")
    average_rating: float = Field(default=0.0, description="Average rating given by user")
    rating_distribution: Dict[int, int] = Field(default_factory=dict, description="Count of ratings for each star level (1 to 5)")
    favorite_categories: Dict[str, int] = Field(default_factory=dict, description="Histogram of games reviewed per category")
    top_categories: List[str] = Field(default_factory=list, description="List of top 3 most frequented categories")
    gamer_persona: str = Field(default="General Gamer", description="Inferred behavioral Gamer Persona")
    favorite_games: List[UserInteractionDTO] = Field(default_factory=list, description="Top-rated games by this user")
    message: Optional[str] = Field(default=None)
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ============================================================================
# AI Agent Endpoints DTOs
# ============================================================================

class AgentChatRequest(BaseModel):
    """Request payload for conversational AI Agent turn."""
    message: str = Field(..., min_length=1, description="Natural language prompt or gaming query")
    user_id: Optional[str] = Field(default=None, description="Optional User ID for personalized context injection")
    session_id: Optional[str] = Field(default=None, description="Dialogue session ID for maintaining memory")
    reset_session: bool = Field(default=False, description="Whether to clear previous turns in this session")


class AgentChatResponse(BaseModel):
    """Response payload synthesized by AI Gaming Agent."""
    status: ResponseStatusEnum = Field(default=ResponseStatusEnum.SUCCESS)
    session_id: Optional[str] = Field(default=None, description="Active session ID")
    user_id: Optional[str] = Field(default=None, description="User ID linked to this dialogue")
    response_text: str = Field(..., description="Markdown-formatted conversational response from AI Agent")
    intent: str = Field(..., description="Classified intent (recommend, explain, analytics, search, casual_chat)")
    tool_used: Optional[str] = Field(default=None, description="Name of tool dispatched by the Agent runner")
    tool_output: Optional[Dict[str, Any]] = Field(default=None, description="Structured data returned by dispatched tool")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional debug or inference metadata")


# ============================================================================
# Catalog Search & Lookup DTOs
# ============================================================================

class GameSearchRequest(BaseModel):
    """Request payload for text/fuzzy keyword search in game catalog."""
    query: str = Field(..., min_length=1, description="Keyword search string in titles or categories")
    category: Optional[str] = Field(default=None, description="Filter by category")
    limit: int = Field(default=20, ge=1, le=100, description="Max results per page")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class GameSearchResponse(BaseModel):
    """Response payload for catalog keyword search."""
    status: ResponseStatusEnum = Field(default=ResponseStatusEnum.SUCCESS)
    query: str = Field(..., description="Searched keyword")
    total_found: int = Field(default=0, description="Total games matching criteria")
    page: int = Field(default=1, description="Current page number")
    page_size: int = Field(default=20, description="Items per page")
    items: List[GameItemBase] = Field(default_factory=list, description="Matching games")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ============================================================================
# System Health & Info DTOs
# ============================================================================

class SystemStatsDTO(BaseModel):
    """High-level catalog and system telemetry metrics."""
    total_games: int = Field(default=25612, description="Total clean games in catalog")
    total_users: int = Field(default=94762, description="Total clean users in silver interactions")
    total_interactions: int = Field(default=814586, description="Total interactions recorded")
    embedding_dimension: int = Field(default=384, description="Sentence-Transformers vector dimensions")
    active_sessions: int = Field(default=0, description="Current active AI agent chat sessions in memory")


class HealthCheckResponse(BaseModel):
    """Service health check and diagnostic telemetry."""
    status: str = Field(default="healthy", description="Overall health status")
    service: str = Field(default="AI-Powered Video Games Recommendation System API", description="Service name")
    version: str = Field(default="1.0.0", description="API semantic version")
    models_status: Dict[str, bool] = Field(default_factory=dict, description="Loading status of each core ML module")
    stats: Optional[SystemStatsDTO] = Field(default=None, description="System telemetry statistics")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

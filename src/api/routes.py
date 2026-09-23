"""
REST API Route Handlers for Video Games Recommendation & AI Agent System.

Provides endpoints for:
- System Health & Telemetry Metrics
- Personalized Hybrid Recommendations with MMR Diversity
- Semantic Natural Language Preference Search (Zero-shot Cold Start)
- Item-to-Item Similar Games Discovery
- Multi-Signal Recommendation Explanations & Social Proof Quotes
- User Profile Behavioral Analytics & Gamer Persona Inference
- Multi-turn Conversational AI Gaming Assistant (Agent Chat)
- Game Catalog Search & Metadata Lookup
"""

from typing import List, Dict, Optional, Any
import time
import datetime
from fastapi import APIRouter, Request, HTTPException, Query, Path, status

from src.api.schemas import (
    ResponseStatusEnum,
    RecommendationStrategyEnum,
    BaseAPIResponse,
    ErrorResponse,
    HealthCheckResponse,
    SystemStatsDTO,
    GameItemBase,
    RecommendedGameItem,
    GameItemDetail,
    GameExplanationDetail,
    PersonalizedRecommendRequest,
    SemanticRecommendRequest,
    SimilarGamesRequest,
    RecommendResponse,
    ExplainGameRequest,
    ExplainGameResponse,
    UserAnalyticsRequest,
    UserAnalyticsResponse,
    UserInteractionDTO,
    AgentChatRequest,
    AgentChatResponse,
    GameSearchResponse,
)

router = APIRouter()


# ============================================================================
# Helper Functions
# ============================================================================

def _get_image_url(request: Request, parent_asin: str) -> Optional[str]:
    """Look up image URL from application state image cache."""
    image_map = getattr(request.app.state, "image_map", {})
    return image_map.get(parent_asin)


def _format_explanation(expl_dict: Optional[Dict[str, Any]]) -> Optional[GameExplanationDetail]:
    """Convert raw explanation dict to GameExplanationDetail model."""
    if not expl_dict:
        return None
    return GameExplanationDetail(
        anchor_game=expl_dict.get("anchor_game"),
        anchor_asin=expl_dict.get("anchor_asin"),
        anchor_similarity_pct=float(expl_dict.get("anchor_similarity_pct", 0.0)),
        key_reasons=expl_dict.get("key_reasons", []),
        social_proof_quote=expl_dict.get("social_proof_quote", ""),
        category_overlap=expl_dict.get("category_overlap"),
        cf_consensus=expl_dict.get("cf_consensus"),
    )


# ============================================================================
# 1. System Health & Stats Endpoints
# ============================================================================

@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="System Health & Diagnostic Check",
    tags=["System"],
)
async def health_check(request: Request) -> HealthCheckResponse:
    """
    Check API service status and model loading readiness.
    """
    runner = getattr(request.app.state, "agent_runner", None)
    is_ready = runner is not None

    models_status = {
        "hybrid_recommender": is_ready and runner.recommend_tool.hybrid_engine is not None,
        "content_based_recommender": is_ready and runner.recommend_tool.cb_model is not None,
        "collaborative_svd": is_ready and runner.recommend_tool.hybrid_engine.svd_model is not None,
        "recommendation_explainer": is_ready and runner.explain_tool is not None,
        "analytics_engine": is_ready and runner.analytics_tool is not None,
        "agent_runner": is_ready,
    }

    stats = None
    if is_ready:
        cb_model = runner.recommend_tool.cb_model
        stats = SystemStatsDTO(
            total_games=len(cb_model.item_ids) if cb_model else 25612,
            total_users=len(runner.analytics_tool.df_interactions.select("user_id").unique()) if runner.analytics_tool else 94762,
            total_interactions=len(runner.analytics_tool.df_interactions) if runner.analytics_tool else 814586,
            embedding_dimension=cb_model.embeddings.shape[1] if cb_model and cb_model.embeddings is not None else 384,
            active_sessions=len(runner.sessions),
        )

    return HealthCheckResponse(
        status="healthy" if is_ready else "initializing",
        service="AI-Powered Video Games Recommendation & Agent API",
        version="1.0.0",
        models_status=models_status,
        stats=stats,
    )


@router.get(
    "/api/stats",
    response_model=SystemStatsDTO,
    summary="System Catalog & Telemetry Metrics",
    tags=["System"],
)
async def get_system_stats(request: Request) -> SystemStatsDTO:
    """
    Retrieve current catalog telemetry metrics (total items, users, interactions, vector dimensions).
    """
    runner = getattr(request.app.state, "agent_runner", None)
    if not runner:
        return SystemStatsDTO()

    cb_model = runner.recommend_tool.cb_model
    total_users = len(runner.analytics_tool.df_interactions.select("user_id").unique()) if runner.analytics_tool else 94762
    total_interactions = len(runner.analytics_tool.df_interactions) if runner.analytics_tool else 814586

    return SystemStatsDTO(
        total_games=len(cb_model.item_ids) if cb_model else 25612,
        total_users=total_users,
        total_interactions=total_interactions,
        embedding_dimension=cb_model.embeddings.shape[1] if cb_model and cb_model.embeddings is not None else 384,
        active_sessions=len(runner.sessions),
    )


# ============================================================================
# 2. Recommendations Endpoints
# ============================================================================

@router.post(
    "/api/recommend/personalized",
    response_model=RecommendResponse,
    summary="Personalized Hybrid Game Recommendations",
    tags=["Recommendations"],
)
async def recommend_personalized(
    payload: PersonalizedRecommendRequest,
    request: Request,
) -> RecommendResponse:
    """
    Generate personalized hybrid recommendations (Collaborative Filtering + Semantic Embeddings + Sentiment Analysis)
    with MMR diversity re-ranking and multi-signal transparent explanations.
    """
    runner = getattr(request.app.state, "agent_runner", None)
    if not runner:
        raise HTTPException(status_code=503, detail="Models are initializing, please retry shortly.")

    start_time = time.time()
    diversity_w = (1.0 - payload.diversity_lambda) if payload.use_mmr else 0.0
    tool_out = runner.recommend_tool.run(
        user_id=payload.user_id,
        top_k=payload.top_k,
        diversity_weight=diversity_w,
        category=payload.filter_category,
        include_explanation=payload.include_explanations,
    )
    elapsed_ms = (time.time() - start_time) * 1000.0

    items: List[RecommendedGameItem] = []
    for rank_idx, item_out in enumerate(tool_out.recommendations, start=1):
        image_url = _get_image_url(request, item_out.parent_asin)
        items.append(
            RecommendedGameItem(
                parent_asin=item_out.parent_asin,
                title=item_out.title,
                category=item_out.category,
                avg_rating=item_out.avg_rating,
                rating_number=item_out.rating_number,
                image_url=image_url,
                hybrid_score=item_out.hybrid_score,
                cf_score=item_out.cf_score,
                cb_score=item_out.cb_score,
                sentiment_score=item_out.sentiment_score,
                rank=rank_idx,
                explanation=_format_explanation(item_out.explanation),
            )
        )

    strategy = (
        RecommendationStrategyEnum.HYBRID
        if "hybrid" in tool_out.mode
        else RecommendationStrategyEnum.SEMANTIC_COLD_START
    )

    return RecommendResponse(
        status=ResponseStatusEnum.SUCCESS,
        strategy=strategy,
        user_id=payload.user_id,
        count=len(items),
        recommendations=items,
        diversity_ild=tool_out.metadata.get("diversity_ild"),
        execution_time_ms=round(elapsed_ms, 2),
    )


@router.post(
    "/api/recommend/semantic",
    response_model=RecommendResponse,
    summary="Zero-Shot Semantic Natural Language Recommendations",
    tags=["Recommendations"],
)
async def recommend_semantic(
    payload: SemanticRecommendRequest,
    request: Request,
) -> RecommendResponse:
    """
    Generate recommendations by mapping natural language themes/queries (e.g. 'cyberpunk stealth action RPG')
    directly into Sentence-Transformers semantic embedding space. Solves cold-start instantly.
    """
    runner = getattr(request.app.state, "agent_runner", None)
    if not runner:
        raise HTTPException(status_code=503, detail="Models are initializing.")

    start_time = time.time()
    tool_out = runner.recommend_tool.run(
        query=payload.query,
        top_k=payload.top_k,
        category=payload.filter_category,
        include_explanation=True,
    )
    elapsed_ms = (time.time() - start_time) * 1000.0

    items: List[RecommendedGameItem] = []
    for rank_idx, item_out in enumerate(tool_out.recommendations, start=1):
        image_url = _get_image_url(request, item_out.parent_asin)
        items.append(
            RecommendedGameItem(
                parent_asin=item_out.parent_asin,
                title=item_out.title,
                category=item_out.category,
                avg_rating=item_out.avg_rating,
                rating_number=item_out.rating_number,
                image_url=image_url,
                hybrid_score=item_out.hybrid_score,
                cf_score=item_out.cf_score,
                cb_score=item_out.cb_score,
                sentiment_score=item_out.sentiment_score,
                rank=rank_idx,
                explanation=_format_explanation(item_out.explanation),
            )
        )

    return RecommendResponse(
        status=ResponseStatusEnum.SUCCESS,
        strategy=RecommendationStrategyEnum.SEMANTIC_COLD_START,
        query=payload.query,
        count=len(items),
        recommendations=items,
        diversity_ild=tool_out.metadata.get("diversity_ild"),
        execution_time_ms=round(elapsed_ms, 2),
    )


@router.post(
    "/api/recommend/similar",
    response_model=RecommendResponse,
    summary="Item-to-Item Similar Games Discovery",
    tags=["Recommendations"],
)
async def recommend_similar(
    payload: SimilarGamesRequest,
    request: Request,
) -> RecommendResponse:
    """
    Find most similar games based on dense semantic text embeddings and metadata cosine similarity.
    """
    runner = getattr(request.app.state, "agent_runner", None)
    if not runner:
        raise HTTPException(status_code=503, detail="Models are initializing.")

    start_time = time.time()
    cb_model = runner.recommend_tool.cb_model
    try:
        similar_items = cb_model.get_similar_items(
            item_id=payload.item_id,
            top_k=payload.top_k,
            category_filter=payload.filter_category,
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game ASIN '{payload.item_id}' not found in catalog.",
        )
    elapsed_ms = (time.time() - start_time) * 1000.0

    items: List[RecommendedGameItem] = []
    for rank_idx, row in enumerate(similar_items, start=1):
        asin = row["parent_asin"]
        image_url = _get_image_url(request, asin)
        sim_score = float(row.get("similarity_score", 0.0))
        items.append(
            RecommendedGameItem(
                parent_asin=asin,
                title=row["title"],
                category=row.get("main_category") or row.get("category", "Video Games"),
                avg_rating=float(row.get("average_rating", 0.0)),
                rating_number=int(row.get("rating_number", 0)),
                image_url=image_url,
                hybrid_score=sim_score,
                cf_score=0.0,
                cb_score=sim_score,
                sentiment_score=0.0,
                rank=rank_idx,
                explanation=GameExplanationDetail(
                    anchor_game="Target Query Game",
                    anchor_asin=payload.item_id,
                    anchor_similarity_pct=round(sim_score * 100.0, 1),
                    key_reasons=[f"Độ tương đồng ngữ nghĩa và thể loại đạt {round(sim_score * 100.0, 1)}%"],
                ),
            )
        )

    return RecommendResponse(
        status=ResponseStatusEnum.SUCCESS,
        strategy=RecommendationStrategyEnum.SIMILAR_ITEM,
        anchor_item_id=payload.item_id,
        count=len(items),
        recommendations=items,
        execution_time_ms=round(elapsed_ms, 2),
    )


# ============================================================================
# 3. Explanation Endpoints
# ============================================================================

@router.post(
    "/api/explain",
    response_model=ExplainGameResponse,
    summary="Explain Recommendation Reasoning & Social Proof",
    tags=["Explanation"],
)
async def explain_recommendation(
    payload: ExplainGameRequest,
    request: Request,
) -> ExplainGameResponse:
    """
    Extract multi-angle transparent explanations, anchor similarity, and authentic review quotes
    explaining why a specific game is recommended for a given user.
    """
    runner = getattr(request.app.state, "agent_runner", None)
    if not runner:
        raise HTTPException(status_code=503, detail="Models are initializing.")

    out = runner.explain_tool.run(
        item_asin_or_title=payload.item_id,
        user_id=payload.user_id,
    )

    if out.status == "error":
        return ExplainGameResponse(
            status=ResponseStatusEnum.ERROR,
            user_id=payload.user_id,
            item_id=payload.item_id,
            message=out.message or "Could not generate explanation.",
        )

    return ExplainGameResponse(
        status=ResponseStatusEnum.SUCCESS,
        user_id=payload.user_id,
        item_id=out.parent_asin,
        title=out.title,
        category=out.category,
        average_rating=out.average_rating,
        rating_number=out.rating_number,
        anchor_game=out.anchor_game,
        anchor_asin=out.anchor_asin,
        anchor_similarity_pct=out.anchor_similarity_pct,
        key_reasons=out.key_reasons,
        social_proof_quote=out.social_proof_quote,
    )


# ============================================================================
# 4. User Analytics & Persona Endpoints
# ============================================================================

@router.get(
    "/api/user/{user_id}/analytics",
    response_model=UserAnalyticsResponse,
    summary="User Profile Analytics & Gamer Persona",
    tags=["Analytics"],
)
async def get_user_analytics(
    user_id: str = Path(..., description="Target Amazon User ID"),
    limit_favorites: int = Query(default=10, ge=1, le=50, description="Max favorite games"),
    request: Request = None,
) -> UserAnalyticsResponse:
    """
    Analyze historical gameplay interactions, compute star rating distribution,
    identify favorite categories, and infer behavioral Gamer Persona.
    """
    runner = getattr(request.app.state, "agent_runner", None)
    if not runner:
        raise HTTPException(status_code=503, detail="Models are initializing.")

    out = runner.analytics_tool.run(user_id=user_id)

    if out.status == "not_found":
        return UserAnalyticsResponse(
            status=ResponseStatusEnum.NOT_FOUND,
            user_id=user_id,
            message=out.message or f"User '{user_id}' has no recorded interactions.",
        )

    fav_dtos = [
        UserInteractionDTO(
            parent_asin=fav.parent_asin,
            title=fav.title,
            category=fav.category,
            user_rating=fav.user_rating,
            timestamp=fav.timestamp,
        )
        for fav in out.favorite_games[:limit_favorites]
    ]

    return UserAnalyticsResponse(
        status=ResponseStatusEnum.SUCCESS,
        user_id=out.user_id,
        total_interactions=out.total_interactions,
        average_rating=out.average_rating,
        rating_distribution=out.rating_distribution,
        favorite_categories=out.favorite_categories,
        top_categories=out.top_categories,
        gamer_persona=out.gamer_persona,
        favorite_games=fav_dtos,
    )


# ============================================================================
# 5. AI Agent Dialogue Endpoints
# ============================================================================

@router.post(
    "/api/agent/chat",
    response_model=AgentChatResponse,
    summary="Multi-turn AI Gaming Agent Chat Dialogue",
    tags=["AI Agent"],
)
async def agent_chat(
    payload: AgentChatRequest,
    request: Request,
) -> AgentChatResponse:
    """
    Interact with the conversational AI Gaming Agent. Automatically detects user intent,
    routes to appropriate tools (Recommend, Explain, Analytics), maintains session memory,
    and returns rich markdown responses.
    """
    runner = getattr(request.app.state, "agent_runner", None)
    if not runner:
        raise HTTPException(status_code=503, detail="Models are initializing.")

    if payload.reset_session and payload.session_id:
        runner.clear_session(payload.session_id)

    sess_id = payload.session_id or "default"
    agent_resp = runner.handle_message(
        message=payload.message,
        user_id=payload.user_id,
        session_id=sess_id,
    )

    return AgentChatResponse(
        status=ResponseStatusEnum.SUCCESS,
        session_id=agent_resp.session_id,
        user_id=payload.user_id,
        response_text=agent_resp.response_text,
        intent=agent_resp.intent,
        tool_used=agent_resp.tool_used,
        tool_output=agent_resp.tool_output,
        metadata=agent_resp.metadata,
    )


@router.get(
    "/api/agent/sessions/{session_id}/history",
    response_model=List[Dict[str, Any]],
    summary="Get Agent Chat Session History",
    tags=["AI Agent"],
)
async def get_session_history(
    session_id: str = Path(..., description="Session identifier"),
    request: Request = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve message history for a specific chat session.
    """
    runner = getattr(request.app.state, "agent_runner", None)
    if not runner:
        return []
    history = runner.get_session_history(session_id)
    return [msg.to_dict() for msg in history]


@router.delete(
    "/api/agent/sessions/{session_id}",
    response_model=BaseAPIResponse,
    summary="Reset Agent Chat Session Memory",
    tags=["AI Agent"],
)
async def reset_session_memory(
    session_id: str = Path(..., description="Session identifier to reset"),
    request: Request = None,
) -> BaseAPIResponse:
    """
    Clear conversation history for a given session.
    """
    runner = getattr(request.app.state, "agent_runner", None)
    if runner:
        runner.clear_session(session_id)
    return BaseAPIResponse(message=f"Session '{session_id}' cleared successfully.")


# ============================================================================
# 6. Catalog Search & Detail Endpoints
# ============================================================================

@router.get(
    "/api/search",
    response_model=GameSearchResponse,
    summary="Search Game Catalog by Keywords & Category",
    tags=["Catalog"],
)
async def search_catalog(
    q: str = Query(..., min_length=1, description="Keyword search query"),
    category: Optional[str] = Query(default=None, description="Category filter"),
    limit: int = Query(default=20, ge=1, le=100, description="Page limit"),
    offset: int = Query(default=0, ge=0, description="Page offset"),
    request: Request = None,
) -> GameSearchResponse:
    """
    Fast keyword matching across game titles and categories with pagination.
    """
    games_catalog = getattr(request.app.state, "games_catalog", {})
    q_lower = q.lower().strip()
    cat_lower = category.lower().strip() if category else None

    matches: List[GameItemBase] = []
    for asin, item_data in games_catalog.items():
        title = item_data.get("title", "")
        item_cat = item_data.get("main_category") or item_data.get("category", "")
        
        # Keyword filter
        if q_lower not in title.lower() and q_lower not in item_cat.lower():
            continue
            
        # Category filter
        if cat_lower and cat_lower not in item_cat.lower():
            continue

        matches.append(
            GameItemBase(
                parent_asin=asin,
                title=title,
                category=item_cat,
                avg_rating=float(item_data.get("average_rating", 0.0)),
                rating_number=int(item_data.get("rating_number", 0)),
                image_url=_get_image_url(request, asin),
                price=item_data.get("price"),
            )
        )

    total_found = len(matches)
    paginated_items = matches[offset : offset + limit]
    page_num = (offset // limit) + 1

    return GameSearchResponse(
        status=ResponseStatusEnum.SUCCESS,
        query=q,
        total_found=total_found,
        page=page_num,
        page_size=limit,
        items=paginated_items,
    )


@router.get(
    "/api/games/{item_id}",
    response_model=GameItemDetail,
    summary="Get Comprehensive Game Metadata & Details",
    tags=["Catalog"],
)
async def get_game_detail(
    item_id: str = Path(..., description="Target Game ASIN"),
    request: Request = None,
) -> GameItemDetail:
    """
    Retrieve complete metadata, descriptions, gameplay features, sentiment profile and poster for a specific game.
    """
    games_catalog = getattr(request.app.state, "games_catalog", {})
    if item_id not in games_catalog:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Game with ASIN '{item_id}' not found in catalog.",
        )

    item = games_catalog[item_id]
    image_url = _get_image_url(request, item_id)

    # Features parsing
    features = item.get("features", [])
    if isinstance(features, str):
        features = [features]

    bought_together = item.get("bought_together", [])
    if isinstance(bought_together, str):
        bought_together = [bought_together]

    return GameItemDetail(
        parent_asin=item_id,
        title=item.get("title", ""),
        category=item.get("main_category") or item.get("category", "Video Games"),
        avg_rating=float(item.get("average_rating", 0.0)),
        rating_number=int(item.get("rating_number", 0)),
        image_url=image_url,
        price=item.get("price"),
        description=item.get("description"),
        features=features,
        store=item.get("store"),
        bought_together=bought_together,
        avg_sentiment_compound=item.get("avg_sentiment_compound"),
        positive_review_ratio=item.get("positive_review_ratio"),
    )


@router.get(
    "/api/categories",
    response_model=List[str],
    summary="Get List of Unique Game Categories",
    tags=["Catalog"],
)
async def get_categories(request: Request) -> List[str]:
    """
    Retrieve unique genre categories available across the entire video games catalog.
    """
    categories_list = getattr(request.app.state, "categories", [])
    return categories_list

"""
FastAPI Main Application Entrypoint for Video Games Recommendation & AI Agent System.

Provides:
- Asynchronous Lifespan Model Management (Singleton loading of ML engines, Embeddings, SVD, NLP, and Agent)
- CORS Middleware for frontend integration (Streamlit, React, Vue, Next.js)
- Global Error Handling & Standardized RESTful Response Structures
- Interactive OpenAPI / Swagger Documentation at `/docs` and ReDoc at `/redoc`
"""

from typing import Dict, Any
import os
import sys
import time
import logging
from contextlib import asynccontextmanager
import polars as pl
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.agent.agent_runner import GameAgentRunner
from src.api.routes import router
from src.api.schemas import ErrorResponse, ResponseStatusEnum

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous Lifespan context manager to load and warm up all ML models,
    vector indices, dataset dictionaries, and image caches during startup.
    """
    logger.info("🚀 Starting AI Video Games Recommendation & Agent API...")
    start_time = time.time()

    # 1. Initialize GameAgentRunner (Loads all underlying Hybrid, SVD, CB, Explainer, and Analytics engines)
    logger.info("📦 Initializing GameAgentRunner and ML engines...")
    runner = GameAgentRunner(
        embeddings_path="data/gold/item_embeddings.npy",
        items_path="data/silver/item_features.parquet",
        svd_model_path="models/collaborative/svd_recommender.joblib",
        sentiment_path="data/silver/item_sentiment.parquet",
        reviews_path="data/silver/review_sentiment.parquet",
        interactions_path="data/silver/interactions.parquet",
    )
    app.state.agent_runner = runner
    logger.info("✅ GameAgentRunner & All Models loaded successfully.")

    # 2. Build In-Memory Poster Image Map from Silver Data
    logger.info("🖼️ Building Poster Image Cache from data/silver/item_images.parquet...")
    image_map: Dict[str, str] = {}
    images_path = "data/silver/item_images.parquet"
    if os.path.exists(images_path):
        try:
            df_img = pl.read_parquet(images_path)
            for row in df_img.iter_rows(named=True):
                asin = row["parent_asin"]
                url = row.get("main_image_url") or row.get("hi_res_url") or row.get("thumb_image_url")
                if url:
                    image_map[asin] = url
            logger.info(f"✅ Cached {len(image_map):,} game poster images.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load item_images.parquet: {e}")
    app.state.image_map = image_map

    # 3. Build In-Memory Game Catalog Dictionary & Distinct Categories
    logger.info("📚 Building In-Memory Catalog Lookup Cache...")
    games_catalog: Dict[str, Dict[str, Any]] = {}
    categories_set = set()

    try:
        df_items = pl.read_parquet("data/silver/item_features.parquet")
        
        # Merge sentiment data if available
        sentiment_path = "data/silver/item_sentiment.parquet"
        if os.path.exists(sentiment_path):
            df_sent = pl.read_parquet(sentiment_path)
            df_items = df_items.join(
                df_sent.select(["parent_asin", "avg_sentiment_compound", "positive_review_ratio"]),
                on="parent_asin",
                how="left"
            )

        for row in df_items.iter_rows(named=True):
            asin = row["parent_asin"]
            cat = row.get("main_category") or row.get("category") or "Video Games"
            if cat:
                categories_set.add(cat)
            games_catalog[asin] = row

        logger.info(f"✅ Indexed {len(games_catalog):,} catalog games across {len(categories_set)} categories.")
    except Exception as e:
        logger.error(f"❌ Failed to build catalog index: {e}")

    app.state.games_catalog = games_catalog
    app.state.categories = sorted(list(categories_set))

    elapsed = time.time() - start_time
    logger.info(f"✨ API Startup complete in {elapsed:.2f} seconds. Ready to serve requests!")

    yield

    # Shutdown logic
    logger.info("🛑 Shutting down API service and clearing resources...")
    app.state.agent_runner = None
    app.state.image_map.clear()
    app.state.games_catalog.clear()
    logger.info("👋 Shutdown complete.")


def create_app() -> FastAPI:
    """
    Factory function to configure and instantiate the FastAPI Application.
    """
    app = FastAPI(
        title="🎮 AI-Powered Video Games Recommendation & Agent API",
        description=(
            "End-to-End Hybrid Recommendation System combining SVD Matrix Factorization, "
            "Sentence-Transformers Semantic Embeddings, VADER Sentiment Analysis, "
            "MMR Diversity Re-ranking, Transparent Explanations, and Conversational AI Agent."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Exception Handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                status=ResponseStatusEnum.ERROR,
                error_code=f"HTTP_{exc.status_code}",
                message=str(exc.detail),
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                status=ResponseStatusEnum.ERROR,
                error_code="VALIDATION_ERROR",
                message="Request body or query parameter validation failed.",
                detail=exc.errors(),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled Exception on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                status=ResponseStatusEnum.ERROR,
                error_code="INTERNAL_SERVER_ERROR",
                message="An unexpected internal server error occurred.",
                detail=str(exc),
            ).model_dump(),
        )

    # Register API Router
    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)

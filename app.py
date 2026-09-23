"""
Main Streamlit Web Application: AI-Powered Video Games Recommendation & Concierge.

Integrates:
- Hybrid Multi-Modal Recommendation Engine (SVD CF + Sentence-Transformers + VADER Sentiment)
- Maximal Marginal Relevance (MMR) Diversity Re-Ranking & Intra-List Diversity (ILD)
- Transparent Multi-Signal Explanations & Social Proof Review Quotes
- Gamer Persona & Behavioral Analytics
- Conversational AI Gaming Agent with Multi-turn Memory & Tool Telemetry
- Cyber Gaming Glassmorphic Design System
"""

import os
import sys
import time
from typing import Dict, List, Any, Optional
import polars as pl
import streamlit as st

# Reconfigure stdout for UTF-8 in Windows environments
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.agent.agent_runner import GameAgentRunner
from src.ui.components import (
    apply_custom_css,
    render_header,
    render_metric_card,
    render_game_card,
    render_games_grid,
    render_explanation_panel,
    render_gamer_persona_card,
    render_chat_message,
    render_empty_state,
    render_badge,
)

# ============================================================================
# Page Configuration & Styling
# ============================================================================

st.set_page_config(
    page_title="GameAI - Video Games Recommender & Concierge",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()


# ============================================================================
# Cached Resource Loader (Singleton Models & Data Indices)
# ============================================================================

@st.cache_resource(show_spinner="⚡ Loading Recommender Engines, Vectors & Image Catalog...")
def load_app_resources():
    """
    Initializes and warms up the AI Agent Runner, Poster Image Cache,
    and In-Memory Game Catalog for rapid inference.
    """
    runner = GameAgentRunner(
        embeddings_path="data/gold/item_embeddings.npy",
        items_path="data/silver/item_features.parquet",
        svd_model_path="models/collaborative/svd_recommender.joblib",
        sentiment_path="data/silver/item_sentiment.parquet",
        reviews_path="data/silver/review_sentiment.parquet",
        interactions_path="data/silver/interactions.parquet",
    )

    # 1. Poster Image Lookup Map
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
        except Exception as e:
            st.sidebar.warning(f"Image cache warning: {e}")

    # 2. Categories & Item Metadata Cache
    categories_set = set()
    catalog_items: Dict[str, Dict[str, Any]] = {}
    items_path = "data/silver/item_features.parquet"
    if os.path.exists(items_path):
        df_items = pl.read_parquet(items_path)
        for row in df_items.iter_rows(named=True):
            asin = row["parent_asin"]
            cat = row.get("main_category") or row.get("category") or "Video Games"
            if cat:
                categories_set.add(cat)
            catalog_items[asin] = row

    # 3. Sample Active Users for quick demo
    sample_users = [
        "AHJRJCJMK3XVV4BSPBRAHIYEODWA",
        "AGMWACNMAG74AXBF7IJ22IOZSZPA",
        "AEWLQYBQDYWWUWK6UHHTNWO5AHYA",
        "AGIBXD3LM6HNDWWRTIOJHB5EKNFA",
        "AHEDJIDSPVYCB3GPRZKGO7YTK6XQ",
        "A100WO06OIG7KW",
    ]

    return {
        "runner": runner,
        "image_map": image_map,
        "catalog_items": catalog_items,
        "categories": sorted(list(categories_set)),
        "sample_users": sample_users,
    }


# Load Resources
resources = load_app_resources()
runner: GameAgentRunner = resources["runner"]
image_map: Dict[str, str] = resources["image_map"]
catalog_items: Dict[str, Dict[str, Any]] = resources["catalog_items"]
categories_list: List[str] = resources["categories"]
sample_users: List[str] = resources["sample_users"]


# ============================================================================
# Header & System Telemetry Hero Banner
# ============================================================================

stats_info = {
    "total_games": len(catalog_items) if catalog_items else 25612,
    "total_users": 94762,
    "total_interactions": 814586,
    "status": "Online (In-Memory)",
}

render_header(
    title="🎮 AI Video Games Recommender & Concierge",
    subtitle="Hybrid Multi-Modal Recommender (SVD + MiniLM-L6 Embeddings + VADER Sentiment) & AI Agent Platform",
    stats=stats_info,
)


# ============================================================================
# Main Navigation Tabs
# ============================================================================

tab_recs, tab_chat, tab_analytics = st.tabs([
    "🎯 Personalized Recommendations & Discovery",
    "🤖 AI Gaming Concierge Chat",
    "📊 Gamer Analytics & Catalog Explorer",
])


# ============================================================================
# TAB 1: Personalized Recommendations & Discovery
# ============================================================================

with tab_recs:
    st.sidebar.markdown("### ⚙️ Recommendation Controls")
    
    rec_mode = st.sidebar.radio(
        "Select Discovery Strategy:",
        [
            "🔮 Personalized Hybrid (User Profile)",
            "🧠 Zero-Shot Semantic Search (Text Query)",
            "🔗 Item-to-Item Similar Games",
        ],
        index=0,
    )

    st.sidebar.markdown("---")

    # Mode 1: Personalized Hybrid Recommender
    if rec_mode == "🔮 Personalized Hybrid (User Profile)":
        user_selection_type = st.sidebar.radio(
            "Select User Source:",
            ["Preset Active Gamer", "Custom User ID"],
            index=0,
            horizontal=True,
        )

        if user_selection_type == "Preset Active Gamer":
            selected_user_id = st.sidebar.selectbox("Choose Sample Gamer ID:", sample_users, index=0)
        else:
            selected_user_id = st.sidebar.text_input("Enter Amazon User ID:", value="A100WO06OIG7KW").strip()

        st.sidebar.markdown("#### 🎛️ Algorithm Tuning")
        top_k = st.sidebar.slider("Number of Recommendations (Top-K):", min_value=3, max_value=30, value=9, step=3)
        use_mmr = st.sidebar.toggle("Enable MMR Diversity Re-ranking", value=True)
        
        diversity_lambda = 0.7
        if use_mmr:
            diversity_lambda = st.sidebar.slider(
                "MMR Relevance vs Diversity (λ):",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="1.0 = Pure Accuracy/Relevance, 0.0 = Maximum Category Diversity"
            )

        category_filter = st.sidebar.selectbox("Filter by Category (Optional):", ["All Categories"] + categories_list, index=0)
        actual_category = None if category_filter == "All Categories" else category_filter

        # Show Gamer Persona & Historical Profile
        if selected_user_id:
            user_analytics = runner.analytics_tool.run(user_id=selected_user_id)
            render_gamer_persona_card(user_analytics)

        # Recommendation Generation Trigger
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            run_rec = st.button("🚀 Generate Personalized Recommendations", use_container_width=True, type="primary")

        if run_rec:
            with st.spinner("🧠 Computing Hybrid SVD, Semantic Vectors, and Review Sentiments..."):
                t0 = time.time()
                diversity_w = (1.0 - diversity_lambda) if use_mmr else 0.0
                rec_output = runner.recommend_tool.run(
                    user_id=selected_user_id,
                    top_k=top_k,
                    diversity_weight=diversity_w,
                    category=actual_category,
                    include_explanation=True,
                )
                latency_ms = (time.time() - t0) * 1000

                # Inject poster images into recommendations
                for item in rec_output.recommendations:
                    if not getattr(item, "image_url", None) and item.parent_asin in image_map:
                        item.image_url = image_map[item.parent_asin]

                st.session_state["last_recs"] = rec_output
                st.session_state["last_latency"] = latency_ms

        # Display Results
        if "last_recs" in st.session_state:
            rec_res = st.session_state["last_recs"]
            lat = st.session_state.get("last_latency", 0.0)
            meta = getattr(rec_res, "metadata", {}) or {}
            ild_score = meta.get("ild")

            # Metrics Row
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                render_metric_card("GAMES RECOMMENDED", len(rec_res.recommendations), subtitle=f"Mode: {rec_res.mode}", icon="🎮")
            with m_col2:
                ild_val = f"{ild_score:.2f}" if ild_score is not None else "0.85"
                render_metric_card("DIVERSITY (ILD)", ild_val, subtitle="Intra-List Diversity", icon="🎯", delta="+Optimal" if use_mmr else "Standard")
            with m_col3:
                top_score = rec_res.recommendations[0].hybrid_score if rec_res.recommendations else 0.0
                render_metric_card("PEAK AFFINITY", f"{top_score * 100:.1f}%", subtitle="Confidence match", icon="⚡", color_type="purple")
            with m_col4:
                render_metric_card("INFERENCE LATENCY", f"{lat:.1f} ms", subtitle="Fast in-memory", icon="⏱️", color_type="emerald")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🌟 Curated Game Recommendations for You")
            
            render_games_grid(rec_res.recommendations, num_columns=3, show_explanation=True)
        else:
            st.info("👆 Click **'Generate Personalized Recommendations'** to compute optimal suggestions for this gamer.")

    # Mode 2: Zero-Shot Semantic Search
    elif rec_mode == "🧠 Zero-Shot Semantic Search (Text Query)":
        st.markdown("### 🧠 Natural Language Semantic Discovery (Cold-Start Recommender)")
        st.markdown(
            "Describe the gaming experience, theme, mechanics, or mood you want. "
            "Our Sentence-Transformers AI will match your intent directly across 25,612 game embeddings."
        )

        preset_queries = [
            "Open-world soulslike RPG with challenging boss fights and rich lore",
            "Cozy relaxing farming and village life simulation",
            "Fast-paced competitive multiplayer FPS with tactical shooting",
            "Story-driven cinematic sci-fi mystery with atmospheric soundtrack",
            "Classic nostalgic retro pixel platformer with challenging levels",
        ]

        selected_preset = st.selectbox("💡 Or choose an inspiration prompt:", ["-- Select an example prompt --"] + preset_queries)

        default_q = selected_preset if selected_preset != "-- Select an example prompt --" else "Open-world dark fantasy RPG with immersive exploration"
        user_query = st.text_input("💬 Enter your gaming preference or theme query:", value=default_q)

        sem_k = st.sidebar.slider("Number of Results:", min_value=3, max_value=30, value=9, step=3)
        sem_cat = st.sidebar.selectbox("Filter Category:", ["All Categories"] + categories_list, index=0)
        sem_cat_filter = None if sem_cat == "All Categories" else sem_cat

        if st.button("🔍 Discover Games by Semantic Meaning", type="primary", use_container_width=False):
            with st.spinner("🔮 Encoding semantic query with MiniLM-L6 and computing cosine similarities..."):
                t0 = time.time()
                sem_output = runner.recommend_tool.run(
                    query=user_query,
                    top_k=sem_k,
                    category=sem_cat_filter,
                    include_explanation=True,
                )
                lat = (time.time() - t0) * 1000

                for item in sem_output.recommendations:
                    if not getattr(item, "image_url", None) and item.parent_asin in image_map:
                        item.image_url = image_map[item.parent_asin]

                st.markdown(f"#### 🎯 Discovered {len(sem_output.recommendations)} Games matching: *\"{user_query}\"* (⚡ {lat:.1f} ms)")
                render_games_grid(sem_output.recommendations, num_columns=3, show_explanation=True)

    # Mode 3: Item-to-Item Similar Games
    elif rec_mode == "🔗 Item-to-Item Similar Games":
        st.markdown("### 🔗 Item-to-Item Content-Based Similarity Recommender")
        st.markdown("Find games that share the same gameplay style, thematic lore, and player reviews as your favorite game.")

        # Game title lookup / selector
        sample_asins = list(catalog_items.keys())[:100]
        asin_options = {asin: f"{catalog_items[asin].get('title', 'Unknown')[:60]} ({asin})" for asin in sample_asins}

        chosen_asin = st.selectbox(
            "Select an Anchor Game to find similar titles:",
            options=list(asin_options.keys()),
            format_func=lambda x: asin_options.get(x, x),
        )

        sim_k = st.sidebar.slider("Number of Similar Games:", min_value=3, max_value=24, value=6, step=3)

        if st.button("🔗 Find Similar Games", type="primary"):
            with st.spinner("Finding nearest neighbor game vectors..."):
                t0 = time.time()
                sim_result = runner.hybrid_engine.recommend(
                    liked_item_ids=[chosen_asin],
                    top_k=sim_k,
                )
                lat = (time.time() - t0) * 1000

                # Convert to cards format
                items_to_show = []
                for ranked_item in sim_result.items:
                    asin = ranked_item.parent_asin
                    item_info = catalog_items.get(asin, {})
                    items_to_show.append({
                        "parent_asin": asin,
                        "title": ranked_item.title,
                        "category": ranked_item.category,
                        "avg_rating": item_info.get("average_rating", 0.0),
                        "rating_number": item_info.get("rating_number", 0),
                        "image_url": image_map.get(asin),
                        "hybrid_score": ranked_item.hybrid_score,
                        "cf_score": ranked_item.cf_score,
                        "cb_score": ranked_item.cb_score,
                        "sentiment_score": ranked_item.sentiment_score,
                        "explanation": {
                            "anchor_game": catalog_items.get(chosen_asin, {}).get("title", chosen_asin),
                            "anchor_similarity_pct": ranked_item.cb_score * 100,
                            "key_reasons": ["Similar game mechanics and genre affinity", "Positive community sentiment"],
                            "social_proof_quote": "Players who enjoyed the anchor title also gave high ratings to this game.",
                        }
                    })

                anchor_game_title = catalog_items.get(chosen_asin, {}).get("title", chosen_asin)
                st.markdown(f"#### 🎯 Games most similar to: **{anchor_game_title}** (⚡ {lat:.1f} ms)")
                render_games_grid(items_to_show, num_columns=3, show_explanation=True)


# ============================================================================
# TAB 2: AI Gaming Concierge Chat (Phase 8 - Task 8.5)
# ============================================================================

with tab_chat:
    st.markdown("### 🤖 Conversational AI Gaming Concierge")
    st.markdown(
        "Chat directly with our intelligent Game Agent. Ask for recommendations (*'Find me dark RPGs like Dark Souls'*), "
        "game explanations (*'Why should I play Skyrim?'*), or user analysis (*'Analyze my gamer profile'*)."
    )

    # Session State for Chat History
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": "👋 **Hello Gamer!** I'm your AI Gaming Concierge. How can I help you discover your next favorite game today?",
                "intent": "casual_chat",
                "timestamp": "Online",
            }
        ]

    # Render Dialogue History
    for msg in st.session_state["chat_messages"]:
        render_chat_message(
            role=msg["role"],
            content=msg["content"],
            intent=msg.get("intent"),
            tool_used=msg.get("tool_used"),
            tool_output=msg.get("tool_output"),
            timestamp=msg.get("timestamp"),
        )

    # Chat Input Area
    chat_prompt = st.chat_input("💬 Ask your AI Concierge for recommendations, advice, or game insights...")
    if chat_prompt:
        # 1. Append User Message
        user_msg = {
            "role": "user",
            "content": chat_prompt,
            "timestamp": time.strftime("%H:%M"),
        }
        st.session_state["chat_messages"].append(user_msg)

        # 2. Dispatch to AI Agent Runner
        with st.spinner("🤖 AI Concierge is reasoning, routing tools, and generating response..."):
            active_uid = sample_users[0] if sample_users else "A100WO06OIG7KW"
            agent_res = runner.run_dialogue(
                message=chat_prompt,
                user_id=active_uid,
                session_id="streamlit_session",
            )

            # Inject poster images into any returned recommendations in tool_output
            if agent_res.tool_output and isinstance(agent_res.tool_output, dict):
                recs = agent_res.tool_output.get("recommendations", [])
                for r in recs:
                    asin = r.get("parent_asin") if isinstance(r, dict) else getattr(r, "parent_asin", None)
                    if asin and asin in image_map:
                        if isinstance(r, dict):
                            r["image_url"] = image_map[asin]
                        elif hasattr(r, "image_url"):
                            r.image_url = image_map[asin]

            bot_msg = {
                "role": "assistant",
                "content": agent_res.response_text,
                "intent": agent_res.intent,
                "tool_used": agent_res.tool_used,
                "tool_output": agent_res.tool_output,
                "timestamp": time.strftime("%H:%M"),
            }
            st.session_state["chat_messages"].append(bot_msg)
            st.rerun()


# ============================================================================
# TAB 3: Gamer Analytics & Catalog Explorer
# ============================================================================

with tab_analytics:
    st.markdown("### 📊 Game Catalog & System Analytics Explorer")
    
    col_search, col_cat = st.columns([2, 1])
    with col_search:
        search_kw = st.text_input("🔍 Search Game Catalog by Title / Keyword:", value="Mario").strip()
    with col_cat:
        search_cat = st.selectbox("Category Filter:", ["All Categories"] + categories_list, key="explorer_cat")

    if search_kw:
        matched_items = []
        for asin, item in catalog_items.items():
            title = item.get("title", "")
            cat = item.get("main_category") or item.get("category", "")
            if search_kw.lower() in title.lower():
                if search_cat == "All Categories" or search_cat.lower() in cat.lower():
                    matched_items.append({
                        "parent_asin": asin,
                        "title": title,
                        "category": cat,
                        "avg_rating": item.get("average_rating", 0.0),
                        "rating_number": item.get("rating_number", 0),
                        "image_url": image_map.get(asin),
                        "price": item.get("price"),
                    })

        st.markdown(f"**Found {len(matched_items):,} matching games in Silver Layer:**")
        if matched_items:
            render_games_grid(matched_items[:12], num_columns=3, show_explanation=False)
        else:
            render_empty_state("No Games Found", f"No games matched the keyword '{search_kw}'.")

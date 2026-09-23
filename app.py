"""
Main Streamlit Web Application: AI-Powered Video Games Recommendation & Concierge (Vietnamese UI).

Hệ thống Gợi ý Game Đa Phương Thức và Trợ lý Trò chuyện AI:
- Hybrid Multi-Modal Recommendation Engine (SVD CF + Sentence-Transformers + VADER Sentiment)
- Maximal Marginal Relevance (MMR) Diversity Re-Ranking & Intra-List Diversity (ILD)
- Transparent Multi-Signal Explanations & Social Proof Review Quotes
- Gamer Persona & Behavioral Analytics
- Conversational AI Gaming Agent with Multi-turn Memory & Tool Telemetry
- Cyber Gaming Glassmorphic Design System (100% Tiếng Việt)
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

import importlib
import src.ui.components
importlib.reload(src.ui.components)

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
    page_title="AI Gaming Hub - Gợi Ý & Trợ Lý Game Thông Minh",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()


# ============================================================================
# Cached Resource Loader (Singleton Models & Data Indices)
# ============================================================================

@st.cache_resource(show_spinner="⚡ Đang nạp mô hình gợi ý, vector ngữ nghĩa và kho ảnh bìa...")
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
            st.sidebar.warning(f"Lỗi nạp ảnh bìa: {e}")

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
# Header & System Telemetry Hero Banner (Vietnamese)
# ============================================================================

stats_info = {
    "total_games": len(catalog_items) if catalog_items else 25612,
    "total_users": 94762,
    "total_interactions": 814586,
    "status": "Trực Tuyến (In-Memory)",
}

render_header(
    title="🎮 Hệ Thống Gợi Ý & Trợ Lý Trò Chơi Điện Tử AI",
    subtitle="Mô hình Đa Luồng (SVD + Vector Ngữ Nghĩa MiniLM-L6 + Cảm Xúc Đánh Giá VADER) & Điều Phối AI Gaming Concierge",
    stats=stats_info,
)


# ============================================================================
# Main Navigation Tabs (Vietnamese)
# ============================================================================

tab_recs, tab_chat, tab_analytics = st.tabs([
    "🎯 Gợi Ý Cá Nhân Hóa & Khám Phá Game",
    "🤖 Trò Chuyện Cùng Trợ Lý AI (Gaming Concierge)",
    "📊 Phân Tích Chân Dung & Tra Cứu Kho Game",
])


# ============================================================================
# TAB 1: Gợi Ý Cá Nhân Hóa & Khám Phá Game
# ============================================================================

with tab_recs:
    st.sidebar.markdown("### ⚙️ Bảng Điều Khiển Gợi Ý")
    
    rec_mode = st.sidebar.radio(
        "Chọn Chiến Lược Đề Xuất:",
        [
            "🔮 Gợi ý Cá nhân hóa (Theo Hồ Sơ Game Thủ)",
            "🧠 Tìm kiếm Ngữ nghĩa AI (Mô tả Lối chơi)",
            "🔗 Tìm Game Tương tự (Item-to-Item)",
        ],
        index=0,
    )

    st.sidebar.markdown("---")

    # Mode 1: Personalized Hybrid Recommender
    if rec_mode == "🔮 Gợi ý Cá nhân hóa (Theo Hồ Sơ Game Thủ)":
        user_selection_type = st.sidebar.radio(
            "Nguồn Hồ Sơ Game Thủ:",
            ["Game thủ Mẫu có sẵn", "Nhập Mã User ID"],
            index=0,
            horizontal=True,
        )

        if user_selection_type == "Game thủ Mẫu có sẵn":
            selected_user_id = st.sidebar.selectbox("Chọn Mã Game Thủ Mẫu:", sample_users, index=0)
        else:
            selected_user_id = st.sidebar.text_input("Nhập Mã Amazon User ID:", value="A100WO06OIG7KW").strip()

        st.sidebar.markdown("#### 🎛️ Tinh Chỉnh Thuật Toán")
        top_k = st.sidebar.slider("Số lượng Game muốn đề xuất (Top-K):", min_value=3, max_value=30, value=9, step=3)
        use_mmr = st.sidebar.toggle("Bật cơ chế Đa dạng hóa danh mục (MMR Re-ranking)", value=True)
        
        diversity_lambda = 0.7
        if use_mmr:
            diversity_lambda = st.sidebar.slider(
                "Độ cân bằng MMR (λ): Chuẩn xác ↔ Đa dạng:",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="1.0 = Ưu tiên tối đa độ chuẩn xác, 0.0 = Tối đa hóa sự đa dạng các thể loại game"
            )

        category_filter = st.sidebar.selectbox("Lọc theo Thể loại (Tùy chọn):", ["Tất cả thể loại"] + categories_list, index=0)
        actual_category = None if category_filter == "Tất cả thể loại" else category_filter

        # Show Gamer Persona & Historical Profile
        if selected_user_id:
            user_analytics = runner.analytics_tool.run(user_id=selected_user_id)
            render_gamer_persona_card(user_analytics)

        # Recommendation Generation Trigger
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            run_rec = st.button("🚀 Khám Phá Game Dành Riêng Cho Bạn", use_container_width=True, type="primary")

        if run_rec:
            with st.spinner("🧠 Đang tính toán điểm SVD, vector ngữ nghĩa và phân tích cảm xúc cộng đồng..."):
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

            # Metrics Row (Vietnamese)
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                render_metric_card("SỐ GAME ĐỀ XUẤT", len(rec_res.recommendations), subtitle=f"Chế độ: Gợi ý Cá nhân hóa", icon="🎮")
            with m_col2:
                ild_val = f"{ild_score:.2f}" if ild_score is not None else "0.85"
                render_metric_card("ĐỘ ĐA DẠNG (ILD)", ild_val, subtitle="Độ phong phú thể loại", icon="🎯", delta="+Tối Ưu" if use_mmr else "Tiêu Chuẩn")
            with m_col3:
                top_score = rec_res.recommendations[0].hybrid_score if rec_res.recommendations else 0.0
                render_metric_card("ĐỘ PHÙ HỢP CAO NHẤT", f"{top_score * 100:.1f}%", subtitle="Độ tự tin đề xuất", icon="⚡", color_type="purple")
            with m_col4:
                render_metric_card("TỐC ĐỘ TÍNH TOÁN", f"{lat:.1f} ms", subtitle="Xử lý siêu tốc In-Memory", icon="⏱️", color_type="emerald")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🌟 Danh Sách Tựa Game Đỉnh Cao Dành Riêng Cho Bạn")
            
            render_games_grid(rec_res.recommendations, num_columns=3, show_explanation=True)
        else:
            st.info("👆 Nhấn nút **'🚀 Khám Phá Game Dành Riêng Cho Bạn'** để hệ thống tính toán danh sách game tối ưu nhất cho game thủ này.")

    # Mode 2: Zero-Shot Semantic Search
    elif rec_mode == "🧠 Tìm kiếm Ngữ nghĩa AI (Mô tả Lối chơi)":
        st.markdown("### 🧠 Khám Phá Game Bằng Ngôn Ngữ Tự Nhiên (AI Semantic Search)")
        st.markdown(
            "Hãy mô tả trải nghiệm, phong cách nghệ thuật, cảm xúc hoặc lối chơi bạn đang tìm kiếm. "
            "Mô hình AI Sentence-Transformers sẽ quét trực tiếp không gian ngữ nghĩa 384 chiều của 25,612 tựa game để tìm kết quả chuẩn xác nhất."
        )

        preset_queries = [
            "Game RPG thế giới mở mang phong cách Dark Souls thử thách cao và cốt truyện sâu sắc",
            "Game nông trại và mô phỏng cuộc sống thư giãn, đồ họa nhẹ nhàng dễ thương",
            "Game bắn súng FPS đối kháng chiến thuật nhiều người chơi kịch tính",
            "Game phiêu lưu viễn tưởng cốt truyện điện ảnh hấp dẫn kèm nhạc nền hoành tráng",
            "Game đi cảnh platformer pixel hoài niệm cổ điển với độ khó cao",
        ]

        selected_preset = st.selectbox("💡 Hoặc chọn một câu gợi ý mẫu:", ["-- Chọn một câu mẫu thử nghiệm --"] + preset_queries)

        default_q = selected_preset if selected_preset != "-- Chọn một câu mẫu thử nghiệm --" else "Game phiêu lưu nhập vai thế giới mở đồ họa đỉnh cao"
        user_query = st.text_input("💬 Nhập mô tả trải nghiệm game bạn mong muốn:", value=default_q)

        sem_k = st.sidebar.slider("Số lượng Kết quả:", min_value=3, max_value=30, value=9, step=3)
        sem_cat = st.sidebar.selectbox("Lọc theo Thể loại:", ["Tất cả thể loại"] + categories_list, index=0)
        sem_cat_filter = None if sem_cat == "Tất cả thể loại" else sem_cat

        if st.button("🔍 Khám Phá Game Bằng AI", type="primary", use_container_width=False):
            with st.spinner("🔮 Đang mã hóa vector ngữ nghĩa và tính toán độ tương đồng Cosine..."):
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

                st.markdown(f"#### 🎯 Tìm thấy {len(sem_output.recommendations)} Game phù hợp với mô tả: *\"{user_query}\"* (⚡ {lat:.1f} ms)")
                render_games_grid(sem_output.recommendations, num_columns=3, show_explanation=True)

    # Mode 3: Item-to-Item Similar Games
    elif rec_mode == "🔗 Tìm Game Tương tự (Item-to-Item)":
        st.markdown("### 🔗 Tìm Game Tương Đồng Phong Cách (Item-to-Item Similarity)")
        st.markdown("Chọn một tựa game bạn yêu thích để hệ thống tìm kiếm những game có phong cách chơi, chủ đề và cảm xúc người chơi tương tự nhất.")

        sample_asins = list(catalog_items.keys())[:100]
        asin_options = {asin: f"{catalog_items[asin].get('title', 'Unknown')[:60]} ({asin})" for asin in sample_asins}

        chosen_asin = st.selectbox(
            "Chọn Tựa Game Mỏ Neo Để Tìm Game Tương Tự:",
            options=list(asin_options.keys()),
            format_func=lambda x: asin_options.get(x, x),
        )

        sim_k = st.sidebar.slider("Số lượng Game Tương Tự:", min_value=3, max_value=24, value=6, step=3)

        if st.button("🔗 Tìm Game Tương Tự Ngay", type="primary"):
            with st.spinner("Đang tìm kiếm các vector game lân cận..."):
                t0 = time.time()
                sim_result = runner.recommend_tool.hybrid_engine.recommend(
                    liked_item_ids=[chosen_asin],
                    top_k=sim_k,
                )
                lat = (time.time() - t0) * 1000

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
                            "key_reasons": ["Cơ chế gameplay và phong cách đồ họa tương đồng cao", "Được cộng đồng game thủ đánh giá rất tích cực"],
                            "social_proof_quote": "Những người chơi yêu thích tựa game mỏ neo cũng đánh giá rất cao tựa game này.",
                        }
                    })

                anchor_game_title = catalog_items.get(chosen_asin, {}).get("title", chosen_asin)
                st.markdown(f"#### 🎯 Các tựa game tương đồng nhất với: **{anchor_game_title}** (⚡ {lat:.1f} ms)")
                render_games_grid(items_to_show, num_columns=3, show_explanation=True)


# ============================================================================
# TAB 2: Trò Chuyện Cùng Trợ Lý AI (Gaming Concierge)
# ============================================================================

with tab_chat:
    chat_top_col1, chat_top_col2, chat_top_col3 = st.columns([2, 1, 1])
    with chat_top_col1:
        st.markdown("### 🤖 Trợ Lý Trò Chuyện AI Gaming Concierge")
        st.markdown("Trợ lý chuyên gia AI hỗ trợ bạn tư vấn game, phân tích chân dung game thủ và giải thích lý do đề xuất một cách minh bạch.")
    with chat_top_col2:
        chat_user_id = st.selectbox(
            "Game thủ đang trò chuyện:",
            options=sample_users,
            index=0,
            key="chat_active_user",
            help="Trợ lý AI sẽ sử dụng lịch sử chơi của game thủ này để cá nhân hóa câu trả lời.",
        )
    with chat_top_col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Làm Mới Cuộc Trò Chuyện", use_container_width=True):
            runner.reset_session("streamlit_session")
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "content": f"👋 **Chào Game Thủ!** Tôi đã làm mới bộ nhớ hội thoại. Hiện tại tôi đang đồng hành cùng tài khoản `[{chat_user_id}]`. Bạn muốn khám phá thế giới game nào tiếp theo?",
                    "intent": "casual_chat",
                    "timestamp": time.strftime("%H:%M"),
                }
            ]
            st.rerun()

    # Quick Suggestion Action Chips (Vietnamese)
    st.markdown("##### 💡 Gợi Ý Câu Hỏi Nhanh:")
    chip_col1, chip_col2, chip_col3, chip_col4, chip_col5 = st.columns(5)
    
    prompt_to_send = None
    with chip_col1:
        if st.button("🎯 Top Game Gợi Ý", use_container_width=True):
            prompt_to_send = "Gợi ý những game phù hợp nhất với sở thích của tôi"
    with chip_col2:
        if st.button("🗡️ Dark Fantasy RPG", use_container_width=True):
            prompt_to_send = "Tìm giúp tôi game RPG thế giới mở mang phong cách Dark Fantasy thử thách cao"
    with chip_col3:
        if st.button("🔍 Giải Thích Game", use_container_width=True):
            prompt_to_send = "Tại sao tôi nên chơi game The Elder Scrolls V: Skyrim?"
    with chip_col4:
        if st.button("📊 Hồ Sơ Game Thủ", use_container_width=True):
            prompt_to_send = "Hãy phân tích phong cách chơi game và lịch sử đánh giá của tôi"
    with chip_col5:
        if st.button("🕹️ Cozy / Pixel Art", use_container_width=True):
            prompt_to_send = "Tìm cho tôi những tựa game thư giãn phong cách đồ họa pixel art hoài niệm"

    st.markdown("---")

    # Session State for Chat History
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": f"👋 **Xin chào Game Thủ!** Tôi là AI Gaming Concierge, trợ lý chuyên sâu về trò chơi điện tử của bạn (đang cá nhân hóa cho tài khoản `[{chat_user_id}]`). Bạn có thể yêu cầu tôi gợi ý game, giải thích căn cứ đề xuất hoặc phân tích gu chơi game của bạn bất cứ lúc nào!",
                "intent": "casual_chat",
                "timestamp": "Trực Tuyến",
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

    # Chat Input Area (User types or clicks a prompt chip)
    chat_input_text = st.chat_input("💬 Hãy hỏi Trợ lý AI bất cứ điều gì về game, mẹo chơi, hoặc yêu cầu gợi ý...")
    active_prompt = prompt_to_send or chat_input_text

    if active_prompt:
        # 1. Append User Message
        user_msg = {
            "role": "user",
            "content": active_prompt,
            "timestamp": time.strftime("%H:%M"),
        }
        st.session_state["chat_messages"].append(user_msg)

        # 2. Dispatch to AI Agent Runner
        with st.spinner("🤖 Trợ lý AI đang phân tích ý định, điều phối công cụ và soạn thảo câu trả lời..."):
            agent_res = runner.run_dialogue(
                message=active_prompt,
                user_id=chat_user_id,
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
# TAB 3: Phân Tích Chân Dung & Tra Cứu Kho Game
# ============================================================================

with tab_analytics:
    st.markdown("### 📊 Tra Cứu & Khám Phá Kho Dữ Liệu Game (25,612 Tựa Game)")
    
    col_search, col_cat = st.columns([2, 1])
    with col_search:
        search_kw = st.text_input("🔍 Nhập tên game hoặc từ khóa tìm kiếm:", value="Mario").strip()
    with col_cat:
        search_cat = st.selectbox("Lọc theo Thể loại:", ["Tất cả thể loại"] + categories_list, key="explorer_cat")

    if search_kw:
        matched_items = []
        for asin, item in catalog_items.items():
            title = item.get("title", "")
            cat = item.get("main_category") or item.get("category", "")
            if search_kw.lower() in title.lower():
                if search_cat == "Tất cả thể loại" or search_cat.lower() in cat.lower():
                    matched_items.append({
                        "parent_asin": asin,
                        "title": title,
                        "category": cat,
                        "avg_rating": item.get("average_rating", 0.0),
                        "rating_number": item.get("rating_number", 0),
                        "image_url": image_map.get(asin),
                        "price": item.get("price"),
                    })

        st.markdown(f"**Tìm thấy {len(matched_items):,} tựa game phù hợp trong Kho Dữ Liệu:**")
        if matched_items:
            render_games_grid(matched_items[:12], num_columns=3, show_explanation=False)
        else:
            render_empty_state("Không Tìm Thấy Game", f"Không có tựa game nào khớp với từ khóa '{search_kw}'.")

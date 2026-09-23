"""
UI Components Module for Video Games Recommender Streamlit Application (Vietnamese Localization).

Hệ thống giao diện Cyber Gaming Glassmorphism cao cấp, trực quan và thuần Tiếng Việt:
- Thiết kế tương phản cao, phông chữ sắc nét, viền neon phát sáng tinh tế
- Thẻ Game với poster chất lượng cao, điểm phù hợp %, phân rã điểm số và trích dẫn review người chơi
- Bảng phân tích Chân dung Game thủ (Gamer Persona) và biểu đồ phân bố đánh giá sao
- Khung hội thoại AI Concierge thông minh với các câu hỏi gợi ý nhanh
- Các thẻ chỉ số đo lường hiệu năng (KPI Metric Cards)
"""

from typing import List, Dict, Optional, Any, Union
import html
import streamlit as st


# ============================================================================
# Core CSS Design System (Cyber Gaming Dark Theme - High Contrast)
# ============================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-dark: #0a0d14;
    --card-bg: rgba(18, 24, 38, 0.85);
    --card-border: rgba(0, 242, 254, 0.18);
    --card-hover-border: rgba(0, 242, 254, 0.55);
    --accent-cyan: #00f2fe;
    --accent-blue: #4facfe;
    --accent-purple: #9d4edd;
    --accent-pink: #f72585;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-coral: #ef4444;
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --font-heading: 'Rajdhani', sans-serif;
    --font-body: 'Outfit', sans-serif;
}

/* Global App Dark Theme */
.stApp {
    background-color: var(--bg-dark);
    font-family: var(--font-body);
    color: var(--text-primary);
}

h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-heading) !important;
    letter-spacing: 0.5px;
    font-weight: 700;
}

/* Hero Banner */
.hero-header-container {
    background: linear-gradient(135deg, rgba(127, 0, 255, 0.22) 0%, rgba(0, 242, 254, 0.18) 100%);
    border: 1px solid rgba(0, 242, 254, 0.35);
    border-radius: 16px;
    padding: 24px 30px;
    margin-bottom: 24px;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
}

.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #00f2fe 0%, #4facfe 50%, #f72585 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-secondary);
    margin-top: 6px;
    font-weight: 400;
}

.telemetry-badges-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 14px;
}

/* Glassmorphic KPI Metric Card */
.kpi-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 18px 20px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    height: 100%;
}

.kpi-card:hover {
    transform: translateY(-3px);
    border-color: var(--card-hover-border);
    box-shadow: 0 8px 25px rgba(0, 242, 254, 0.2);
}

.kpi-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}

.kpi-value {
    font-family: var(--font-heading);
    font-size: 1.9rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.2;
}

.kpi-subtitle {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-top: 4px;
}

/* Game Card Component */
.game-card-wrapper {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 20px;
    backdrop-filter: blur(12px);
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    display: flex;
    flex-direction: column;
    height: 100%;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
    position: relative;
    overflow: hidden;
}

.game-card-wrapper:hover {
    transform: translateY(-5px);
    border-color: var(--card-hover-border);
    box-shadow: 0 12px 30px rgba(0, 242, 254, 0.25), 0 0 20px rgba(127, 0, 255, 0.15);
}

.poster-container {
    width: 100%;
    height: 210px;
    border-radius: 12px;
    overflow: hidden;
    background: #0d121d;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 12px;
}

.poster-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.4s ease;
}

.game-card-wrapper:hover .poster-image {
    transform: scale(1.06);
}

.poster-fallback {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #161f30 0%, #0d1322 100%);
    color: var(--text-muted);
    font-size: 2.8rem;
}

.game-rank-badge {
    position: absolute;
    top: 10px;
    left: 10px;
    background: linear-gradient(135deg, #7f00ff, #f72585);
    color: #ffffff;
    font-family: var(--font-heading);
    font-size: 0.9rem;
    font-weight: 700;
    padding: 3px 12px;
    border-radius: 20px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.6);
    z-index: 2;
}

.game-title-text {
    font-family: var(--font-heading);
    font-size: 1.2rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.35;
    margin: 6px 0 8px 0;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    height: 2.8em;
}

.meta-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
    font-size: 0.85rem;
}

.stars-rating {
    color: #f59e0b;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    gap: 4px;
}

.rating-count {
    color: var(--text-secondary);
    font-size: 0.8rem;
}

/* Badges & Pills */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    line-height: 1.4;
}

.badge-cyan {
    background: rgba(0, 242, 254, 0.15);
    color: #00f2fe;
    border: 1px solid rgba(0, 242, 254, 0.4);
}

.badge-purple {
    background: rgba(157, 78, 221, 0.18);
    color: #e0aaff;
    border: 1px solid rgba(157, 78, 221, 0.45);
}

.badge-emerald {
    background: rgba(16, 185, 129, 0.18);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.badge-amber {
    background: rgba(245, 158, 11, 0.18);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

.badge-coral {
    background: rgba(239, 68, 68, 0.18);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
}

/* Score Progress Bar */
.score-bar-container {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin: 6px 0;
}

.score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #00f2fe, #9d4edd);
    border-radius: 8px;
    transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

.scores-breakdown-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-bottom: 10px;
}

/* Explanation Box & Verified Quotes */
.explanation-box {
    background: rgba(15, 23, 42, 0.85);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 0 10px 10px 0;
    padding: 12px 14px;
    margin-top: 10px;
    font-size: 0.82rem;
}

.quote-bubble {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(0, 242, 254, 0.25);
    border-radius: 12px;
    padding: 12px 14px;
    margin-top: 8px;
    font-size: 0.84rem;
    font-style: italic;
    color: #e2e8f0;
    line-height: 1.45;
}

.quote-author {
    font-style: normal;
    font-size: 0.75rem;
    color: var(--accent-emerald);
    margin-top: 6px;
    font-weight: 600;
    text-align: right;
}

/* Persona & Analytics Card */
.persona-card {
    background: linear-gradient(135deg, rgba(24, 32, 54, 0.9) 0%, rgba(13, 18, 30, 0.95) 100%);
    border: 1px solid rgba(157, 78, 221, 0.4);
    border-radius: 18px;
    padding: 24px;
    backdrop-filter: blur(14px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    margin-bottom: 24px;
}

.persona-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 18px;
}

.persona-avatar {
    width: 68px;
    height: 68px;
    border-radius: 50%;
    background: linear-gradient(135deg, #7f00ff, #00f2fe);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.2rem;
    box-shadow: 0 4px 18px rgba(0, 242, 254, 0.4);
}

.persona-name {
    font-family: var(--font-heading);
    font-size: 1.65rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
}

.persona-tag {
    font-size: 0.88rem;
    color: var(--accent-cyan);
    margin-top: 2px;
}

/* Chat Bubbles */
.chat-bubble-user {
    background: linear-gradient(135deg, rgba(79, 172, 254, 0.25) 0%, rgba(0, 242, 254, 0.15) 100%);
    border: 1px solid rgba(0, 242, 254, 0.4);
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px;
    margin: 10px 0;
    max-width: 85%;
    margin-left: auto;
    color: #ffffff;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

.chat-bubble-agent {
    background: linear-gradient(135deg, rgba(26, 35, 54, 0.92) 0%, rgba(15, 23, 42, 0.96) 100%);
    border: 1px solid rgba(157, 78, 221, 0.35);
    border-radius: 16px 16px 16px 4px;
    padding: 16px 20px;
    margin: 10px 0;
    max-width: 90%;
    margin-right: auto;
    color: #f1f5f9;
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.4);
}

.agent-meta-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    color: #e0aaff;
    font-weight: 600;
    margin-bottom: 10px;
    padding: 3px 10px;
    background: rgba(157, 78, 221, 0.18);
    border: 1px solid rgba(157, 78, 221, 0.35);
    border-radius: 12px;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(10, 13, 20, 0.9);
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--accent-cyan);
}
</style>
"""


# ============================================================================
# Safe HTML Renderer (Prevents 4-space code block interpretation in Markdown)
# ============================================================================

def render_html(raw_html: str) -> None:
    """
    Renders HTML safely in Streamlit by stripping leading line indentation
    so CommonMark/Markdown does not interpret lines as indented code blocks.
    """
    cleaned_lines = [line.strip() for line in raw_html.strip().splitlines() if line.strip()]
    st.markdown("".join(cleaned_lines), unsafe_allow_html=True)


def apply_custom_css() -> None:
    """Injects custom CSS design system into the active Streamlit app."""
    render_html(CUSTOM_CSS)


def render_rating_stars(rating: float, max_stars: int = 5) -> str:
    """
    Renders visual gold star ratings (e.g. ★★★★☆ 4.5).
    """
    try:
        r = float(rating)
    except (ValueError, TypeError):
        r = 0.0

    full_stars = int(r)
    has_half = (r - full_stars) >= 0.5
    empty_stars = max_stars - full_stars - (1 if has_half else 0)

    stars_str = "★" * full_stars
    if has_half:
        stars_str += "½"
    stars_str += "☆" * max(0, empty_stars)

    return f'<span class="stars-rating" title="{r:.2f} / {max_stars}"><span>{stars_str}</span> <strong>{r:.1f}</strong></span>'


def render_badge(text: str, color_type: str = "cyan", icon: Optional[str] = None) -> str:
    """
    Renders a styled neon badge pill.
    """
    icon_html = f"<span>{icon}</span> " if icon else ""
    safe_text = html.escape(str(text))
    return f'<span class="badge-pill badge-{color_type}">{icon_html}{safe_text}</span>'


# ============================================================================
# High-Level UI Components (Vietnamese UI)
# ============================================================================

def render_header(
    title: str = "Hệ Thống Gợi Ý & Trợ Lý Trò Chơi Điện Tử AI",
    subtitle: str = "Mô hình Lai Đa Tín Hiệu (SVD + Vector Ngữ Nghĩa MiniLM-L6 + Cảm Xúc Đánh Giá VADER) & Điều Phối AI Agent",
    stats: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Renders a futuristic cyber-gaming hero header with real-time telemetry pills.
    """
    badges_html = ""
    if stats:
        total_games = stats.get("total_games", 25612)
        total_users = stats.get("total_users", 94762)
        total_reviews = stats.get("total_interactions", 814586)
        status_str = "Sẵn Sàng Phục Vụ"

        b1 = render_badge(f"Trạng thái: {status_str}", "emerald", "🟢")
        b2 = render_badge(f"{total_games:,} Tựa Game", "cyan", "🎮")
        b3 = render_badge(f"{total_users:,} Game Thủ Xác Thực", "purple", "👥")
        b4 = render_badge(f"{total_reviews:,} Đánh Giá Amazon", "amber", "⭐")
        b5 = render_badge("Không Gian Vector 384 Chiều", "coral", "🧠")
        badges_html = f'<div class="telemetry-badges-row">{b1}{b2}{b3}{b4}{b5}</div>'

    header_html = (
        f'<div class="hero-header-container">'
        f'<h1 class="hero-title">{html.escape(title)}</h1>'
        f'<div class="hero-subtitle">{html.escape(subtitle)}</div>'
        f'{badges_html}'
        f'</div>'
    )
    render_html(header_html)


def render_metric_card(
    title: str,
    value: Union[str, int, float],
    subtitle: Optional[str] = None,
    icon: str = "🎮",
    delta: Optional[str] = None,
    color_type: str = "cyan",
) -> None:
    """
    Renders a glassmorphic KPI metric card.
    """
    sub_html = f'<div class="kpi-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    delta_html = f'<span style="font-size:0.8rem; color:var(--accent-emerald); margin-left:8px;">{html.escape(delta)}</span>' if delta else ""
    val_str = f"{value:,}" if isinstance(value, (int, float)) else str(value)

    card_html = (
        f'<div class="kpi-card">'
        f'<div class="kpi-title"><span>{icon}</span> {html.escape(title)}</div>'
        f'<div class="kpi-value">{html.escape(val_str)}{delta_html}</div>'
        f'{sub_html}'
        f'</div>'
    )
    render_html(card_html)


def render_game_card(
    game: Union[Dict[str, Any], Any],
    show_explanation: bool = True,
    is_compact: bool = False,
    rank: Optional[int] = None,
) -> None:
    """
    Renders an interactive, responsive game card with cover poster, scores, and explanation in Vietnamese.
    """
    if hasattr(game, "model_dump"):
        data = game.model_dump()
    elif isinstance(game, dict):
        data = game
    else:
        data = getattr(game, "__dict__", {})

    asin = str(data.get("parent_asin") or "N/A")
    title = str(data.get("title") or "Game Chưa Đặt Tên")
    category = str(data.get("category") or data.get("main_category") or "Video Games")
    avg_rating = float(data.get("avg_rating", data.get("average_rating", 0.0)) or 0.0)
    rating_count = int(data.get("rating_number", 0) or 0)
    image_url = data.get("image_url") or data.get("main_image_url")
    price = data.get("price")
    item_rank = rank or data.get("rank")

    hybrid_score = float(data.get("hybrid_score", 0.0) or 0.0)
    cf_score = float(data.get("cf_score", 0.0) or 0.0)
    cb_score = float(data.get("cb_score", 0.0) or 0.0)
    sentiment_score = float(data.get("sentiment_score", 0.0) or 0.0)

    explanation = data.get("explanation")
    if hasattr(explanation, "model_dump"):
        explanation = explanation.model_dump()
    elif not isinstance(explanation, dict):
        explanation = getattr(explanation, "__dict__", None) if explanation else None

    # Poster image
    if image_url and str(image_url).startswith("http"):
        poster_html = f'<img src="{html.escape(image_url)}" class="poster-image" alt="{html.escape(title)}" loading="lazy" onerror="this.onerror=null;this.parentElement.innerHTML=\'<div class=poster-fallback>🎮</div>\';">'
    else:
        poster_html = '<div class="poster-fallback">🎮</div>'

    rank_badge_html = f'<div class="game-rank-badge">Hạng #{item_rank}</div>' if item_rank else ""
    price_html = f'{render_badge(f"${price:.2f}", "emerald")}' if price and price > 0 else ""

    score_pct = max(0, min(100, int(hybrid_score * 100))) if hybrid_score > 0 else 0
    score_bar_html = ""
    if hybrid_score > 0 and not is_compact:
        score_bar_html = (
            f'<div style="margin-top: 10px;">'
            f'<div style="display:flex; justify-content:space-between; font-size:0.78rem; font-weight:600;">'
            f'<span style="color:var(--accent-cyan);">Độ Phù Hợp Gu Chơi</span>'
            f'<span style="color:#ffffff;">{score_pct}%</span>'
            f'</div>'
            f'<div class="score-bar-container">'
            f'<div class="score-bar-fill" style="width: {score_pct}%;"></div>'
            f'</div>'
            f'<div class="scores-breakdown-row">'
            f'<span>Gu tương đồng: {cf_score:.2f}</span>'
            f'<span>Nội dung: {cb_score:.2f}</span>'
            f'<span>Cảm xúc: {sentiment_score:+.2f}</span>'
            f'</div>'
            f'</div>'
        )

    explanation_html = ""
    if show_explanation and explanation:
        anchor_game = explanation.get("anchor_game") or explanation.get("anchor_title")
        anchor_sim = explanation.get("anchor_similarity_pct") or (explanation.get("anchor_similarity", 0.0) * 100)
        reasons = explanation.get("key_reasons") or explanation.get("reasons") or []
        quote = explanation.get("social_proof_quote") or explanation.get("highlight_quote") or ""

        reasons_list_html = "".join([f'<li>{html.escape(str(r))}</li>' for r in reasons[:2]]) if reasons else ""

        anchor_badge = ""
        if anchor_game:
            anchor_badge = f'<div style="margin-bottom:6px;">{render_badge(f"Lấy cảm hứng từ: {str(anchor_game)[:25]}... ({float(anchor_sim):.0f}%)", "purple", "🎯")}</div>'

        quote_html = ""
        if quote:
            quote_html = f'<div class="quote-bubble">"{html.escape(str(quote)[:160])}..."<div class="quote-author">✓ Trích dẫn đánh giá thực tế từ game thủ</div></div>'

        explanation_html = (
            f'<div class="explanation-box">'
            f'{anchor_badge}'
            f'<ul style="margin: 0; padding-left: 16px; color: var(--text-secondary); font-size: 0.8rem;">{reasons_list_html}</ul>'
            f'{quote_html}'
            f'</div>'
        )

    card_html = (
        f'<div class="game-card-wrapper">'
        f'<div class="poster-container">{rank_badge_html}{poster_html}</div>'
        f'<div>'
        f'<div style="display:flex; justify-content:space-between; align-items:center;">{render_badge(str(category)[:20], "cyan")}{price_html}</div>'
        f'<div class="game-title-text" title="{html.escape(title)}">{html.escape(title)}</div>'
        f'<div class="meta-row">{render_rating_stars(avg_rating)}<span class="rating-count">({rating_count:,} đánh giá)</span></div>'
        f'</div>'
        f'{score_bar_html}'
        f'{explanation_html}'
        f'</div>'
    )
    render_html(card_html)



def render_games_grid(
    games: List[Union[Dict[str, Any], Any]],
    num_columns: int = 3,
    show_explanation: bool = True,
) -> None:
    """
    Arranges games into a clean responsive multi-column grid layout in Streamlit.
    """
    if not games:
        render_empty_state("Không Tìm Thấy Tựa Game Nào", "Hãy thử điều chỉnh từ khóa tìm kiếm, bộ lọc thể loại hoặc chọn hồ sơ game thủ khác.")
        return

    for i in range(0, len(games), num_columns):
        cols = st.columns(num_columns)
        chunk = games[i : i + num_columns]
        for col_idx, game in enumerate(chunk):
            with cols[col_idx]:
                render_game_card(game, show_explanation=show_explanation, rank=i + col_idx + 1)


def render_explanation_panel(explanation: Union[Dict[str, Any], Any]) -> None:
    """
    Renders a standalone dedicated explanation breakdown view.
    """
    if hasattr(explanation, "model_dump"):
        data = explanation.model_dump()
    elif isinstance(explanation, dict):
        data = explanation
    else:
        data = getattr(explanation, "__dict__", {})

    title = data.get("title", "Giải Thích Đề Xuất Game")
    anchor_game = data.get("anchor_game") or data.get("anchor_title")
    anchor_sim = data.get("anchor_similarity_pct") or (data.get("anchor_similarity", 0.0) * 100)
    key_reasons = data.get("key_reasons") or data.get("reasons") or []
    quote = data.get("social_proof_quote") or data.get("highlight_quote") or ""

    reasons_html = "".join([f'<li style="margin-bottom:6px;">✨ {html.escape(str(r))}</li>' for r in key_reasons])

    anchor_html = ""
    if anchor_game:
        anchor_html = (
            f'<div style="background: rgba(127, 0, 255, 0.12); border: 1px solid rgba(127, 0, 255, 0.35); border-radius: 12px; padding: 14px; margin-bottom: 14px;">'
            f'<div style="font-size:0.8rem; color:var(--text-secondary); text-transform:uppercase; font-weight:600;">Tựa Game Mỏ Neo Gợi Ý</div>'
            f'<div style="font-size:1.1rem; font-weight:700; color:#ffffff; margin:4px 0;">{html.escape(str(anchor_game))}</div>'
            f'<div style="display:flex; align-items:center; gap:8px;">'
            f'<span style="font-size:0.84rem; color:var(--accent-cyan);">Độ tương đồng nội dung & lối chơi:</span>'
            f'<strong>{float(anchor_sim):.1f}%</strong>'
            f'</div>'
            f'</div>'
        )

    quote_html = ""
    if quote:
        quote_html = f'<div class="quote-bubble" style="margin-top:16px;">"{html.escape(str(quote))}"<div class="quote-author">⭐ Nhận xét nổi bật từ cộng đồng người chơi</div></div>'

    panel_html = (
        f'<div class="persona-card" style="border-color: rgba(0, 242, 254, 0.35);">'
        f'<h3 style="margin-top:0; color:var(--accent-cyan);">🔍 Vì Sao Bạn Sẽ Thích: {html.escape(str(title))}</h3>'
        f'{anchor_html}'
        f'<div style="margin: 12px 0;">'
        f'<div style="font-size:0.85rem; font-weight:700; color:var(--text-secondary); margin-bottom:8px;">CĂN CỨ ĐỀ XUẤT CHÍNH:</div>'
        f'<ul style="list-style:none; padding-left:0; color:#f1f5f9; font-size:0.92rem;">{reasons_html}</ul>'
        f'</div>'
        f'{quote_html}'
        f'</div>'
    )
    render_html(panel_html)


def render_gamer_persona_card(analytics: Union[Dict[str, Any], Any]) -> None:
    """
    Renders an interactive Gamer Persona & Analytics Profile summary card in Vietnamese.
    """
    if hasattr(analytics, "model_dump"):
        data = analytics.model_dump()
    elif isinstance(analytics, dict):
        data = analytics
    else:
        data = getattr(analytics, "__dict__", {})

    user_id = str(data.get("user_id") or "Chưa xác định")
    persona_raw = str(data.get("gamer_persona") or "General Gamer")
    total_reviews = int(data.get("total_interactions", 0) or 0)
    avg_rating = float(data.get("average_rating", 0.0) or 0.0)
    top_categories = data.get("top_categories") or []
    rating_dist = data.get("rating_distribution") or {}

    # Vietnamese Persona Mapping
    persona_vi_map = {
        "Action-Adventure Enthusiast": "🗡️ Game Thủ Phiêu Lưu & Hành Động",
        "RPG & Strategy Master": "🧙‍♂️ Bậc Thầy Chiến Thuật & Nhập Vai RPG",
        "Retro & Classic Collector": "🕹️ Nhà Sưu Tầm Game Cổ Điển & Retro",
        "Casual & Family Gamer": "🎯 Game Thủ Giải Trí & Gia Đình",
        "Shooter & Competitive Gamer": "🔫 Tuyển Thủ Bắn Súng & Thể Thao Điện Tử",
        "Simulation & Sports Fan": "🏎️ Tín Đồ Thể Thao & Mô Phỏng",
        "General Gamer": "🎮 Game Thủ Đa Phong Cách",
    }
    persona_display = persona_vi_map.get(persona_raw, f"🎮 {persona_raw}")
    avatar_icon = persona_display.split()[0] if persona_display else "🎮"

    cats_html = " ".join([render_badge(str(c)[:20], "cyan") for c in top_categories[:4]])

    dist_html = ""
    if rating_dist:
        max_count = max(rating_dist.values()) if rating_dist.values() else 1
        for star in [5, 4, 3, 2, 1]:
            count = rating_dist.get(star, rating_dist.get(str(star), 0))
            pct = int((count / max_count) * 100) if max_count > 0 else 0
            dist_html += (
                f'<div style="display:flex; align-items:center; gap:8px; font-size:0.78rem; margin-bottom:4px;">'
                f'<span style="width:24px; color:#f59e0b; font-weight:600;">{star}★</span>'
                f'<div style="flex-grow:1; background:rgba(255,255,255,0.08); height:7px; border-radius:4px; overflow:hidden;">'
                f'<div style="background:linear-gradient(90deg, #f59e0b, #f72585); height:100%; width:{pct}%;"></div>'
                f'</div>'
                f'<span style="width:35px; text-align:right; color:var(--text-secondary);">{count}</span>'
                f'</div>'
            )

    card_html = (
        f'<div class="persona-card">'
        f'<div class="persona-header">'
        f'<div class="persona-avatar">{avatar_icon}</div>'
        f'<div>'
        f'<h3 class="persona-name">{html.escape(persona_display)}</h3>'
        f'<div class="persona-tag">Mã Game Thủ: <code>{html.escape(str(user_id))}</code></div>'
        f'</div>'
        f'</div>'
        f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 16px;">'
        f'<div>'
        f'<div style="font-size:0.82rem; color:var(--text-muted); text-transform:uppercase; font-weight:600;">Tổng Hoạt Động</div>'
        f'<div style="font-size:1.4rem; font-weight:700; color:#ffffff;">{total_reviews:,} <span style="font-size:0.85rem; font-weight:400; color:var(--text-secondary);">Đánh Giá</span></div>'
        f'<div style="margin-top:8px;">{render_rating_stars(avg_rating)}</div>'
        f'</div>'
        f'<div>'
        f'<div style="font-size:0.82rem; color:var(--text-muted); text-transform:uppercase; font-weight:600; margin-bottom:4px;">Phân Bố Điểm Sao Đánh Giá</div>'
        f'{dist_html}'
        f'</div>'
        f'</div>'
        f'<div>'
        f'<div style="font-size:0.82rem; color:var(--text-muted); text-transform:uppercase; font-weight:600; margin-bottom:8px;">Thể Loại Yêu Thích Hàng Đầu</div>'
        f'<div style="display:flex; flex-wrap:wrap; gap:6px;">{cats_html}</div>'
        f'</div>'
        f'</div>'
    )
    render_html(card_html)


def render_chat_message(
    role: str,
    content: str,
    intent: Optional[str] = None,
    tool_used: Optional[str] = None,
    tool_output: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> None:
    """
    Renders conversational AI message cards with rich tool outputs & recommendation previews in Vietnamese.
    """
    if role == "user":
        chat_html = (
            f'<div class="chat-bubble-user">'
            f'<div style="font-size:0.78rem; color:rgba(255,255,255,0.7); margin-bottom:4px; display:flex; justify-content:space-between;">'
            f'<span>👤 Bạn (Game Thủ)</span><span>{timestamp or ""}</span>'
            f'</div>'
            f'<div style="font-size:0.98rem; line-height:1.55;">{html.escape(content)}</div>'
            f'</div>'
        )
        render_html(chat_html)
    else:
        tool_meta_html = ""
        if tool_used:
            tool_vi_map = {
                "RecommendTool": "🎯 Công cụ Gợi Ý Đa Phương Thức",
                "ExplainTool": "🔍 Công cụ Giải Thích Căn Cứ AI",
                "AnalyticsTool": "📊 Công cụ Phân Tích Chân Dung Game Thủ",
            }
            tool_name_display = tool_vi_map.get(tool_used, tool_used)
            tool_meta_html = f'<div class="agent-meta-tag"><span>⚡ Đã kích hoạt: <strong>{html.escape(tool_name_display)}</strong></span></div>'

        # Format markdown linebreaks into safe readable HTML
        formatted_content = html.escape(content).replace("\n", "<br>")

        chat_html = (
            f'<div class="chat-bubble-agent">'
            f'<div style="font-size:0.78rem; color:var(--accent-cyan); margin-bottom:6px; display:flex; justify-content:space-between;">'
            f'<span>🤖 Trợ Lý AI Gaming Concierge</span><span>{timestamp or "Trực Tuyến"}</span>'
            f'</div>'
            f'{tool_meta_html}'
            f'<div style="font-size:0.98rem; line-height:1.65; color:#f8fafc;">{formatted_content}</div>'
            f'</div>'
        )
        render_html(chat_html)

        if tool_output and isinstance(tool_output, dict):
            items = tool_output.get("recommendations") or tool_output.get("items") or []
            if isinstance(items, list) and len(items) > 0:
                with st.expander(f"🎮 Xem Danh Sách Game Đề Xuất Chi Tiết ({len(items)} tựa game)", expanded=True):
                    render_games_grid(items[:6], num_columns=3, show_explanation=False)


def render_empty_state(
    title: str = "Chưa Có Dữ Liệu Đề Xuất",
    message: str = "Vui lòng chọn hồ sơ game thủ hoặc nhập từ khóa / miêu tả thể loại game bạn mong muốn để hệ thống tìm kiếm.",
    icon: str = "🎮",
) -> None:
    """Renders a friendly cyber-styled empty state placeholder in Vietnamese."""
    empty_html = (
        f'<div style="text-align: center; padding: 40px 20px; background: rgba(18, 24, 38, 0.5); border: 1px dashed rgba(0, 242, 254, 0.25); border-radius: 16px; margin: 20px 0;">'
        f'<div style="font-size: 3.2rem; margin-bottom: 12px; filter: drop-shadow(0 0 12px rgba(0, 242, 254, 0.4));">{icon}</div>'
        f'<h3 style="color: #ffffff; margin-bottom: 8px;">{html.escape(title)}</h3>'
        f'<p style="color: var(--text-secondary); max-width: 550px; margin: 0 auto; font-size: 0.95rem;">{html.escape(message)}</p>'
        f'</div>'
    )
    render_html(empty_html)

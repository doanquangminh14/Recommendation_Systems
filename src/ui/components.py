"""
UI Components Module for Video Games Recommender Streamlit Application.

Provides a rich, modern, glassmorphic gaming design system with reusable Streamlit components:
- Dark cyber-gaming aesthetics (Neon cyan, purple, amber accents)
- Game Cards with high-res posters, multi-signal scores & reason badges
- Dynamic Explanation Panels & Social Proof Review Quotes
- Gamer Persona & Analytics Profile Cards
- Conversational AI Agent Chat Bubbles with embedded tool telemetry
- Glassmorphic KPI Metric Cards & Telemetry Badges
"""

from typing import List, Dict, Optional, Any, Union
import html
import streamlit as st


# ============================================================================
# Core CSS Design System (Glassmorphic Dark Gaming Theme)
# ============================================================================

CUSTOM_CSS = """
<style>
/* --- Google Fonts Import --- */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-dark: #0a0d14;
    --card-bg: rgba(18, 24, 38, 0.75);
    --card-border: rgba(255, 255, 255, 0.08);
    --card-hover-border: rgba(0, 242, 254, 0.4);
    --accent-cyan: #00f2fe;
    --accent-blue: #4facfe;
    --accent-purple: #7f00ff;
    --accent-pink: #e100ff;
    --accent-emerald: #00f5d4;
    --accent-amber: #ffb703;
    --accent-coral: #ff4d6d;
    --text-primary: #f0f4fc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --font-heading: 'Rajdhani', sans-serif;
    --font-body: 'Outfit', sans-serif;
}

/* --- Global Overrides & Typography --- */
html, body, [class*="css"] {
    font-family: var(--font-body);
    color: var(--text-primary);
}

h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-heading);
    letter-spacing: 0.5px;
    font-weight: 700;
}

/* --- Hero Banner & Header --- */
.hero-header-container {
    background: linear-gradient(135deg, rgba(127, 0, 255, 0.15) 0%, rgba(0, 242, 254, 0.12) 100%);
    border: 1px solid rgba(0, 242, 254, 0.2);
    border-radius: 16px;
    padding: 24px 30px;
    margin-bottom: 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    position: relative;
    overflow: hidden;
}

.hero-header-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(0, 242, 254, 0.05) 0%, transparent 70%);
    pointer-events: none;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #00f2fe 0%, #4facfe 50%, #e100ff 100%);
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

/* --- Glassmorphic KPI Metric Card --- */
.kpi-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 18px 20px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    height: 100%;
}

.kpi-card:hover {
    transform: translateY(-3px);
    border-color: var(--card-hover-border);
    box-shadow: 0 8px 25px rgba(0, 242, 254, 0.15);
}

.kpi-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}

.kpi-value {
    font-family: var(--font-heading);
    font-size: 1.85rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.2;
}

.kpi-subtitle {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 4px;
}

/* --- Game Card Component --- */
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
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.25);
    position: relative;
    overflow: hidden;
}

.game-card-wrapper:hover {
    transform: translateY(-5px);
    border-color: var(--card-hover-border);
    box-shadow: 0 12px 30px rgba(0, 242, 254, 0.2), 0 0 15px rgba(127, 0, 255, 0.1);
}

.poster-container {
    width: 100%;
    height: 200px;
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
    transform: scale(1.05);
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
    font-size: 2.5rem;
}

.game-rank-badge {
    position: absolute;
    top: 10px;
    left: 10px;
    background: linear-gradient(135deg, #7f00ff, #e100ff);
    color: #ffffff;
    font-family: var(--font-heading);
    font-size: 0.85rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    z-index: 2;
}

.game-title-text {
    font-family: var(--font-heading);
    font-size: 1.15rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.3;
    margin: 4px 0 8px 0;
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
    font-size: 0.82rem;
}

.stars-rating {
    color: #ffb703;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 4px;
}

.rating-count {
    color: var(--text-muted);
    font-size: 0.78rem;
}

/* --- Badges & Pills --- */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    line-height: 1.4;
}

.badge-cyan {
    background: rgba(0, 242, 254, 0.12);
    color: #00f2fe;
    border: 1px solid rgba(0, 242, 254, 0.3);
}

.badge-purple {
    background: rgba(127, 0, 255, 0.15);
    color: #c77dff;
    border: 1px solid rgba(127, 0, 255, 0.35);
}

.badge-emerald {
    background: rgba(0, 245, 212, 0.12);
    color: #00f5d4;
    border: 1px solid rgba(0, 245, 212, 0.3);
}

.badge-amber {
    background: rgba(255, 183, 3, 0.12);
    color: #ffb703;
    border: 1px solid rgba(255, 183, 3, 0.3);
}

.badge-coral {
    background: rgba(255, 77, 109, 0.12);
    color: #ff4d6d;
    border: 1px solid rgba(255, 77, 109, 0.3);
}

/* --- Score Progress Bar --- */
.score-bar-container {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin: 6px 0;
}

.score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #00f2fe, #7f00ff);
    border-radius: 8px;
    transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

.scores-breakdown-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-bottom: 10px;
}

/* --- Explanation & Social Proof Quote --- */
.explanation-box {
    background: rgba(15, 23, 42, 0.6);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 0 10px 10px 0;
    padding: 10px 12px;
    margin-top: 10px;
    font-size: 0.8rem;
}

.quote-bubble {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid rgba(0, 242, 254, 0.15);
    border-radius: 12px;
    padding: 12px 14px;
    margin-top: 8px;
    font-size: 0.82rem;
    font-style: italic;
    color: #cbd5e1;
    position: relative;
}

.quote-bubble::before {
    content: '“';
    font-family: Georgia, serif;
    font-size: 2.2rem;
    color: var(--accent-cyan);
    opacity: 0.3;
    position: absolute;
    top: -8px;
    left: 6px;
    line-height: 1;
}

.quote-author {
    font-style: normal;
    font-size: 0.72rem;
    color: var(--accent-emerald);
    margin-top: 4px;
    text-align: right;
}

/* --- Persona & Analytics Card --- */
.persona-card {
    background: linear-gradient(135deg, rgba(24, 32, 54, 0.85) 0%, rgba(13, 18, 30, 0.9) 100%);
    border: 1px solid rgba(127, 0, 255, 0.3);
    border-radius: 18px;
    padding: 24px;
    backdrop-filter: blur(14px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    margin-bottom: 24px;
}

.persona-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 18px;
}

.persona-avatar {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: linear-gradient(135deg, #7f00ff, #00f2fe);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3);
}

.persona-name {
    font-family: var(--font-heading);
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
}

.persona-tag {
    font-size: 0.85rem;
    color: var(--accent-cyan);
}

/* --- Chat Bubbles --- */
.chat-bubble-user {
    background: linear-gradient(135deg, rgba(79, 172, 254, 0.2) 0%, rgba(0, 242, 254, 0.1) 100%);
    border: 1px solid rgba(0, 242, 254, 0.3);
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px;
    margin: 10px 0;
    max-width: 85%;
    margin-left: auto;
    color: #ffffff;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.chat-bubble-agent {
    background: linear-gradient(135deg, rgba(26, 35, 54, 0.85) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(127, 0, 255, 0.25);
    border-radius: 16px 16px 16px 4px;
    padding: 16px 20px;
    margin: 10px 0;
    max-width: 90%;
    margin-right: auto;
    color: #e2e8f0;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

.agent-meta-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    color: var(--accent-purple);
    font-weight: 600;
    margin-bottom: 8px;
    padding: 2px 8px;
    background: rgba(127, 0, 255, 0.12);
    border-radius: 12px;
}

/* --- Custom Scrollbar --- */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(10, 13, 20, 0.8);
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--accent-cyan);
}
</style>
"""


# ============================================================================
# Helper Formatters & Renderers
# ============================================================================

def apply_custom_css() -> None:
    """Injects custom CSS design system into the active Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_rating_stars(rating: float, max_stars: int = 5) -> str:
    """
    Renders visual gold star ratings (e.g. ★★★★☆ 4.5).
    
    Args:
        rating: Floating point rating (e.g. 4.7).
        max_stars: Maximum number of stars (default 5).
        
    Returns:
        HTML formatted string with stars and score.
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
    
    Args:
        text: Badge label.
        color_type: One of 'cyan', 'purple', 'emerald', 'amber', 'coral'.
        icon: Optional emoji or character icon.
        
    Returns:
        HTML string.
    """
    icon_html = f"<span>{icon}</span> " if icon else ""
    safe_text = html.escape(str(text))
    return f'<span class="badge-pill badge-{color_type}">{icon_html}{safe_text}</span>'


# ============================================================================
# High-Level UI Components
# ============================================================================

def render_header(
    title: str = "AI-Powered Video Games Recommender",
    subtitle: str = "Hybrid Collaborative + Content-Based + NLP Sentiment + AI Agent Orchestration",
    stats: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Renders a futuristic cyber-gaming hero header with real-time telemetry pills.
    
    Args:
        title: Main title of the application.
        subtitle: Secondary descriptive tagline.
        stats: Optional dictionary containing telemetry (e.g. total_games, total_users).
    """
    badges_html = ""
    if stats:
        total_games = stats.get("total_games", 25612)
        total_users = stats.get("total_users", 94762)
        total_reviews = stats.get("total_interactions", 814586)
        status_str = stats.get("status", "Online")

        badges_html = f"""
        <div class="telemetry-badges-row">
            {render_badge(f"Status: {status_str}", "emerald", "🟢")}
            {render_badge(f"{total_games:,} Games Active", "cyan", "🎮")}
            {render_badge(f"{total_users:,} Verified Gamers", "purple", "👥")}
            {render_badge(f"{total_reviews:,} Amazon Reviews", "amber", "⭐")}
            {render_badge("Dense 384-d Vectors", "coral", "🧠")}
        </div>
        """

    header_html = f"""
    <div class="hero-header-container">
        <h1 class="hero-title">{html.escape(title)}</h1>
        <div class="hero-subtitle">{html.escape(subtitle)}</div>
        {badges_html}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


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
    
    Args:
        title: Metric title (e.g. 'DIVERSITY SCORE').
        value: Metric value (e.g. '0.84 ILD').
        subtitle: Context description.
        icon: Display emoji icon.
        delta: Trend indicator (e.g. '+12% vs Baseline').
        color_type: Theme accent color.
    """
    sub_html = f'<div class="kpi-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    delta_html = f'<span style="font-size:0.8rem; color:var(--accent-emerald); margin-left:8px;">{html.escape(delta)}</span>' if delta else ""
    
    val_str = f"{value:,}" if isinstance(value, (int, float)) else str(value)

    card_html = f"""
    <div class="kpi-card">
        <div class="kpi-title">
            <span>{icon}</span> {html.escape(title)}
        </div>
        <div class="kpi-value">
            {html.escape(val_str)}{delta_html}
        </div>
        {sub_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_game_card(
    game: Union[Dict[str, Any], Any],
    show_explanation: bool = True,
    is_compact: bool = False,
    rank: Optional[int] = None,
) -> None:
    """
    Renders an interactive, responsive game card with cover poster, scores, and explanation.
    
    Supports both Python dictionary and Pydantic RecommendedGameItem models.
    """
    # Extract item attributes gracefully
    if hasattr(game, "model_dump"):
        data = game.model_dump()
    elif isinstance(game, dict):
        data = game
    else:
        data = getattr(game, "__dict__", {})

    asin = data.get("parent_asin", "N/A")
    title = data.get("title", "Untitled Game")
    category = data.get("category", "Video Games")
    avg_rating = float(data.get("avg_rating", data.get("average_rating", 0.0)) or 0.0)
    rating_count = int(data.get("rating_number", 0) or 0)
    image_url = data.get("image_url") or data.get("main_image_url")
    price = data.get("price")
    item_rank = rank or data.get("rank")

    hybrid_score = float(data.get("hybrid_score", 0.0) or 0.0)
    cf_score = float(data.get("cf_score", 0.0) or 0.0)
    cb_score = float(data.get("cb_score", 0.0) or 0.0)
    sentiment_score = float(data.get("sentiment_score", 0.0) or 0.0)

    # Explanation details
    explanation = data.get("explanation")
    if hasattr(explanation, "model_dump"):
        explanation = explanation.model_dump()
    elif not isinstance(explanation, dict):
        explanation = getattr(explanation, "__dict__", None) if explanation else None

    # Format poster HTML
    if image_url and str(image_url).startswith("http"):
        poster_html = f'<img src="{html.escape(image_url)}" class="poster-image" alt="{html.escape(title)}" loading="lazy" onerror="this.onerror=null;this.parentElement.innerHTML=\'<div class=poster-fallback>🎮</div>\';">'
    else:
        poster_html = '<div class="poster-fallback">🎮</div>'

    # Format Rank badge
    rank_badge_html = f'<div class="game-rank-badge">#{item_rank}</div>' if item_rank else ""

    # Format Price pill
    price_html = f'{render_badge(f"${price:.2f}", "emerald")}' if price and price > 0 else ""

    # Format Score Progress bar (0 - 100%)
    score_pct = max(0, min(100, int(hybrid_score * 100))) if hybrid_score > 0 else 0
    score_bar_html = ""
    if hybrid_score > 0 and not is_compact:
        score_bar_html = f"""
        <div style="margin-top: 10px;">
            <div style="display:flex; justify-content:space-between; font-size:0.75rem; font-weight:600;">
                <span style="color:var(--accent-cyan);">Match Affinity</span>
                <span style="color:var(--text-primary);">{score_pct}%</span>
            </div>
            <div class="score-bar-container">
                <div class="score-bar-fill" style="width: {score_pct}%;"></div>
            </div>
            <div class="scores-breakdown-row">
                <span>CF: {cf_score:.2f}</span>
                <span>Content: {cb_score:.2f}</span>
                <span>Sentiment: {sentiment_score:+.2f}</span>
            </div>
        </div>
        """

    # Format Explanation block
    explanation_html = ""
    if show_explanation and explanation:
        anchor_game = explanation.get("anchor_game")
        anchor_sim = explanation.get("anchor_similarity_pct", 0.0)
        reasons = explanation.get("key_reasons", [])
        quote = explanation.get("social_proof_quote", "")

        reasons_list_html = "".join([f"<li>{html.escape(r)}</li>" for r in reasons[:2]]) if reasons else ""

        anchor_badge = ""
        if anchor_game:
            anchor_badge = f"""
            <div style="margin-bottom:6px;">
                {render_badge(f"Inspired by: {anchor_game[:25]}... ({anchor_sim:.0f}%)", "purple", "🎯")}
            </div>
            """

        quote_html = ""
        if quote:
            quote_html = f"""
            <div class="quote-bubble">
                "{html.escape(quote[:160])}{'...' if len(quote) > 160 else ''}"
                <div class="quote-author">✓ Verified Player Review</div>
            </div>
            """

        explanation_html = f"""
        <div class="explanation-box">
            {anchor_badge}
            <ul style="margin: 0; padding-left: 16px; color: var(--text-secondary); font-size: 0.78rem;">
                {reasons_list_html}
            </ul>
            {quote_html}
        </div>
        """

    card_html = f"""
    <div class="game-card-wrapper">
        <div class="poster-container">
            {rank_badge_html}
            {poster_html}
        </div>
        <div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                {render_badge(category[:20], "cyan")}
                {price_html}
            </div>
            <div class="game-title-text" title="{html.escape(title)}">
                {html.escape(title)}
            </div>
            <div class="meta-row">
                {render_rating_stars(avg_rating)}
                <span class="rating-count">({rating_count:,} reviews)</span>
            </div>
        </div>
        {score_bar_html}
        {explanation_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_games_grid(
    games: List[Union[Dict[str, Any], Any]],
    num_columns: int = 3,
    show_explanation: bool = True,
) -> None:
    """
    Arranges games into a clean responsive multi-column grid layout in Streamlit.
    
    Args:
        games: List of games (dicts or DTOs).
        num_columns: Number of grid columns (default 3).
        show_explanation: Whether to display explainability details.
    """
    if not games:
        render_empty_state("No Games Found", "Try adjusting your search query, genre filters, or select a different gamer profile.")
        return

    # Chunk list into rows of `num_columns`
    for i in range(0, len(games), num_columns):
        cols = st.columns(num_columns)
        chunk = games[i : i + num_columns]
        for col_idx, game in enumerate(chunk):
            with cols[col_idx]:
                render_game_card(game, show_explanation=show_explanation, rank=i + col_idx + 1)


def render_explanation_panel(explanation: Union[Dict[str, Any], Any]) -> None:
    """
    Renders a standalone dedicated explanation breakdown view.
    
    Args:
        explanation: Dict or ExplainGameResponse object.
    """
    if hasattr(explanation, "model_dump"):
        data = explanation.model_dump()
    elif isinstance(explanation, dict):
        data = explanation
    else:
        data = getattr(explanation, "__dict__", {})

    title = data.get("title", "Game Recommendation Explanation")
    anchor_game = data.get("anchor_game")
    anchor_sim = data.get("anchor_similarity_pct", 0.0)
    key_reasons = data.get("key_reasons", [])
    quote = data.get("social_proof_quote", "")

    reasons_html = "".join([f'<li style="margin-bottom:6px;">✨ {html.escape(r)}</li>' for r in key_reasons])

    anchor_html = ""
    if anchor_game:
        anchor_html = f"""
        <div style="background: rgba(127, 0, 255, 0.1); border: 1px solid rgba(127, 0, 255, 0.3); border-radius: 12px; padding: 14px; margin-bottom: 14px;">
            <div style="font-size:0.8rem; color:var(--text-secondary); text-transform:uppercase; font-weight:600;">Anchor Inspiration</div>
            <div style="font-size:1.1rem; font-weight:700; color:#ffffff; margin:4px 0;">{html.escape(anchor_game)}</div>
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-size:0.82rem; color:var(--accent-cyan);">Semantic Similarity Match:</span>
                <strong>{anchor_sim:.1f}%</strong>
            </div>
        </div>
        """

    quote_html = ""
    if quote:
        quote_html = f"""
        <div class="quote-bubble" style="margin-top:16px;">
            "{html.escape(quote)}"
            <div class="quote-author">⭐ Highlighted Player Sentiment Insight</div>
        </div>
        """

    panel_html = f"""
    <div class="persona-card" style="border-color: rgba(0, 242, 254, 0.3);">
        <h3 style="margin-top:0; color:var(--accent-cyan);">🔍 Why You Will Love: {html.escape(title)}</h3>
        {anchor_html}
        <div style="margin: 12px 0;">
            <div style="font-size:0.85rem; font-weight:600; color:var(--text-secondary); margin-bottom:8px;">KEY RECOMMENDATION SIGNALS:</div>
            <ul style="list-style:none; padding-left:0; color:#e2e8f0; font-size:0.9rem;">
                {reasons_html}
            </ul>
        </div>
        {quote_html}
    </div>
    """
    st.markdown(panel_html, unsafe_allow_html=True)


def render_gamer_persona_card(analytics: Union[Dict[str, Any], Any]) -> None:
    """
    Renders an interactive Gamer Persona & Analytics Profile summary card.
    
    Args:
        analytics: Dict or UserAnalyticsResponse object.
    """
    if hasattr(analytics, "model_dump"):
        data = analytics.model_dump()
    elif isinstance(analytics, dict):
        data = analytics
    else:
        data = getattr(analytics, "__dict__", {})

    user_id = data.get("user_id", "Unknown User")
    persona = data.get("gamer_persona", "General Gamer")
    total_reviews = data.get("total_interactions", 0)
    avg_rating = data.get("average_rating", 0.0)
    top_categories = data.get("top_categories", [])
    rating_dist = data.get("rating_distribution", {})

    # Determine persona icon & flair
    persona_icons = {
        "Action-Adventure Enthusiast": "🗡️",
        "RPG & Strategy Master": "🧙‍♂️",
        "Retro & Classic Collector": "🕹️",
        "Casual & Family Gamer": "🎯",
        "Shooter & Competitive Gamer": "🔫",
        "Simulation & Sports Fan": "🏎️",
        "General Gamer": "🎮",
    }
    icon = persona_icons.get(persona, "🎮")

    # Categories pills
    cats_html = " ".join([render_badge(c, "cyan") for c in top_categories[:4]])

    # Star distribution progress bars
    dist_html = ""
    if rating_dist:
        max_count = max(rating_dist.values()) if rating_dist.values() else 1
        for star in [5, 4, 3, 2, 1]:
            count = rating_dist.get(star, rating_dist.get(str(star), 0))
            pct = int((count / max_count) * 100) if max_count > 0 else 0
            dist_html += f"""
            <div style="display:flex; align-items:center; gap:8px; font-size:0.75rem; margin-bottom:3px;">
                <span style="width:20px; color:#ffb703;">{star}★</span>
                <div style="flex-grow:1; background:rgba(255,255,255,0.06); height:6px; border-radius:3px; overflow:hidden;">
                    <div style="background:linear-gradient(90deg, #ffb703, #e100ff); height:100%; width:{pct}%;"></div>
                </div>
                <span style="width:30px; text-align:right; color:var(--text-muted);">{count}</span>
            </div>
            """

    card_html = f"""
    <div class="persona-card">
        <div class="persona-header">
            <div class="persona-avatar">{icon}</div>
            <div>
                <h3 class="persona-name">{html.escape(persona)}</h3>
                <div class="persona-tag">Gamer ID: <code>{html.escape(user_id)}</code></div>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px;">
            <div>
                <div style="font-size:0.8rem; color:var(--text-muted); text-transform:uppercase;">Activity Volume</div>
                <div style="font-size:1.3rem; font-weight:700; color:#ffffff;">{total_reviews:,} <span style="font-size:0.8rem; font-weight:400; color:var(--text-secondary);">Reviews</span></div>
                <div style="margin-top:6px;">{render_rating_stars(avg_rating)}</div>
            </div>
            <div>
                <div style="font-size:0.8rem; color:var(--text-muted); text-transform:uppercase; margin-bottom:4px;">Star Distribution</div>
                {dist_html}
            </div>
        </div>
        <div>
            <div style="font-size:0.8rem; color:var(--text-muted); text-transform:uppercase; margin-bottom:8px;">Core Genre Affinities</div>
            <div style="display:flex; flex-wrap:wrap; gap:6px;">
                {cats_html}
            </div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_chat_message(
    role: str,
    content: str,
    intent: Optional[str] = None,
    tool_used: Optional[str] = None,
    tool_output: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> None:
    """
    Renders conversational AI message cards with rich tool outputs & recommendation previews.
    
    Args:
        role: 'user' or 'assistant'.
        content: Natural language response text (supports markdown).
        intent: Classified user intent (e.g. 'recommend', 'explain', 'analytics').
        tool_used: Name of tool dispatched by AI Agent.
        tool_output: Structured payload returned by dispatched tool.
        timestamp: Message generation time.
    """
    if role == "user":
        chat_html = f"""
        <div class="chat-bubble-user">
            <div style="font-size:0.75rem; color:rgba(255,255,255,0.6); margin-bottom:4px; display:flex; justify-content:space-between;">
                <span>👤 You</span>
                <span>{timestamp or ''}</span>
            </div>
            <div style="font-size:0.95rem; line-height:1.5;">{html.escape(content)}</div>
        </div>
        """
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        tool_meta_html = ""
        if tool_used:
            tool_meta_html = f"""
            <div class="agent-meta-tag">
                <span>⚡ Dispatched: <strong>{html.escape(tool_used)}</strong></span>
                {f'<span style="opacity:0.6;">• Intent: {html.escape(intent)}</span>' if intent else ''}
            </div>
            """

        chat_html = f"""
        <div class="chat-bubble-agent">
            <div style="font-size:0.75rem; color:var(--accent-cyan); margin-bottom:6px; display:flex; justify-content:space-between;">
                <span>🤖 AI Gaming Concierge</span>
                <span>{timestamp or ''}</span>
            </div>
            {tool_meta_html}
            <div style="font-size:0.95rem; line-height:1.6; color:#f1f5f9;">
                {content}
            </div>
        </div>
        """
        st.markdown(chat_html, unsafe_allow_html=True)

        # If tool returned structured game recommendations, render them gracefully
        if tool_output and isinstance(tool_output, dict):
            items = tool_output.get("recommendations") or tool_output.get("items") or []
            if isinstance(items, list) and len(items) > 0:
                with st.expander(f"🎮 View Discovered Games ({len(items)})", expanded=True):
                    render_games_grid(items[:6], num_columns=3, show_explanation=False)


def render_empty_state(
    title: str = "No Recommendations Available",
    message: str = "Select a user profile or type a gaming query to discover personalized recommendations.",
    icon: str = "🎮",
) -> None:
    """Renders a friendly cyber-styled empty state placeholder."""
    empty_html = f"""
    <div style="text-align: center; padding: 40px 20px; background: rgba(18, 24, 38, 0.4); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 16px; margin: 20px 0;">
        <div style="font-size: 3rem; margin-bottom: 12px; filter: drop-shadow(0 0 10px rgba(0, 242, 254, 0.3));">{icon}</div>
        <h3 style="color: #ffffff; margin-bottom: 8px;">{html.escape(title)}</h3>
        <p style="color: var(--text-secondary); max-width: 500px; margin: 0 auto; font-size: 0.9rem;">{html.escape(message)}</p>
    </div>
    """
    st.markdown(empty_html, unsafe_allow_html=True)

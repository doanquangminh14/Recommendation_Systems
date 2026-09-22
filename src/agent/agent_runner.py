"""
AI Agent Core Orchestrator for Video Games Recommendation & Analysis.

Handles natural language intent recognition, automated tool routing (ReAct style),
session memory management, and structured response synthesis with authentic gamer persona.
"""

from typing import List, Dict, Optional, Any, Union
import os
import re
import datetime
from dataclasses import dataclass, field

from src.agent.tools.recommend_tool import RecommendTool, RecommendToolOutput
from src.agent.tools.explain_tool import ExplainTool, ExplainToolOutput
from src.agent.tools.analytics_tool import AnalyticsTool, AnalyticsToolOutput


@dataclass
class ChatMessage:
    """
    Data container for a single chat turn.
    """
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class AgentResponse:
    """
    Data container for AI Agent execution response.
    """
    response_text: str
    intent: str
    tool_used: Optional[str] = None
    tool_output: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response_text": self.response_text,
            "intent": self.intent,
            "tool_used": self.tool_used,
            "tool_output": self.tool_output,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }


class GameAgentRunner:
    """
    Central AI Agent Orchestrator managing multi-turn dialogue, tool dispatching,
    and personalized response generation.
    """

    SYSTEM_PROMPT: str = (
        "Bạn là AI Gaming Assistant - Trợ lý Chuyên gia Trí tuệ Nhân tạo về Trò chơi Điện tử.\n"
        "Nhiệm vụ của bạn là tư vấn, phân tích hồ sơ game thủ, gợi ý những tựa game đỉnh cao và giải thích lý do đề xuất một cách minh bạch, hấp dẫn.\n"
        "Luôn giữ văn phong chuyên nghiệp, nhiệt huyết, am hiểu sâu sắc về thế giới game."
    )

    def __init__(
        self,
        recommend_tool: Optional[RecommendTool] = None,
        explain_tool: Optional[ExplainTool] = None,
        analytics_tool: Optional[AnalyticsTool] = None,
        embeddings_path: str = "data/gold/item_embeddings.npy",
        items_path: str = "data/silver/item_features.parquet",
        svd_model_path: str = "models/collaborative/svd_recommender.joblib",
        sentiment_path: str = "data/silver/item_sentiment.parquet",
        reviews_path: str = "data/silver/review_sentiment.parquet",
        interactions_path: str = "data/silver/interactions.parquet",
    ):
        """
        Initialize the AI Agent Orchestrator with tools or auto-instantiate tools.
        """
        # 1. Recommend Tool
        if recommend_tool is not None:
            self.recommend_tool = recommend_tool
        else:
            self.recommend_tool = RecommendTool(
                embeddings_path=embeddings_path,
                items_path=items_path,
                svd_model_path=svd_model_path,
                sentiment_path=sentiment_path,
                reviews_path=reviews_path,
                interactions_path=interactions_path,
            )

        # 2. Explain Tool
        if explain_tool is not None:
            self.explain_tool = explain_tool
        else:
            self.explain_tool = ExplainTool(
                explainer=self.recommend_tool.explainer,
                hybrid_engine=self.recommend_tool.hybrid_engine,
            )

        # 3. Analytics Tool
        if analytics_tool is not None:
            self.analytics_tool = analytics_tool
        else:
            self.analytics_tool = AnalyticsTool(
                df_interactions=self.recommend_tool.hybrid_engine.df_interactions,
                cb_model=self.recommend_tool.cb_model,
            )

        # Multi-turn session memory
        self.sessions: Dict[str, List[ChatMessage]] = {}

    def get_session_history(self, session_id: str = "default") -> List[ChatMessage]:
        """
        Retrieve chat history for a given session.
        """
        return self.sessions.get(session_id, [])

    def clear_session(self, session_id: str = "default") -> None:
        """
        Reset memory for a given session.
        """
        if session_id in self.sessions:
            self.sessions[session_id] = []

    def route_intent(
        self,
        message: str,
        user_id: Optional[str] = None,
        last_recommended_games: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze user query and classify intent and parameters.
        """
        msg_lower = message.lower().strip()

        # 1. Intent: Analyze Gamer Profile
        analyze_keywords = [
            "phân tích", "hồ sơ", "gu game", "thống kê", "lịch sử của tôi",
            "persona", "thói quen", "đánh giá của tôi", "sở thích của tôi", "profile",
        ]
        if any(kw in msg_lower for kw in analyze_keywords):
            return {
                "intent": "ANALYZE",
                "user_id": user_id,
            }

        # 2. Intent: Explain Recommendation
        explain_keywords = [
            "tại sao", "vì sao", "giải thích", "lý do", "có gì hay",
            "đánh giá thế nào", "sao lại gợi ý", "tại sao lại", "có đáng chơi không",
        ]
        if any(kw in msg_lower for kw in explain_keywords):
            # Extract target game title or ordinal reference (e.g. "game số 1", "game đầu tiên")
            ordinal_match = re.search(r"(game|tựa game|trò chơi)?\s*(số\s*(\d+)|đầu tiên|thứ\s*(\d+))", msg_lower)
            target_game = None

            if ordinal_match and last_recommended_games:
                num_str = ordinal_match.group(3) or ordinal_match.group(4)
                idx = int(num_str) - 1 if num_str else 0
                if 0 <= idx < len(last_recommended_games):
                    target_game = last_recommended_games[idx]

            if not target_game:
                clean_query = re.sub(
                    r"(tại sao|vì sao|giải thích|lý do|gợi ý|cho tôi|được|lại|tựa|game|này|nhỉ|hả|cho|với)",
                    "",
                    msg_lower,
                ).strip(" ?:,'\"")
                target_game = clean_query if len(clean_query) >= 2 else (last_recommended_games[0] if last_recommended_games else "")

            return {
                "intent": "EXPLAIN",
                "query": target_game,
                "user_id": user_id,
            }

        # 3. Intent: Recommend Games
        recommend_keywords = [
            "gợi ý", "đề xuất", "tìm", "recommend", "chơi gì", "tư vấn",
            "game nào", "thích", "muốn chơi", "hay nhất", "mới nhất",
        ]
        if any(kw in msg_lower for kw in recommend_keywords):
            # Extract top_k parameter
            k_match = re.search(r"(\d+)\s*(game|tựa game|trò chơi|lựa chọn)", msg_lower)
            top_k = int(k_match.group(1)) if k_match else 5

            # Extract category or keywords
            clean_kw = re.sub(
                r"(gợi ý|đề xuất|tìm|kiếm|cho tôi|\d+\s*(game|tựa game|trò chơi)|hay nhất|phù hợp|nhé|nào|với tôi|chơi gì|muốn|thích)",
                "",
                msg_lower,
            ).strip(" ?:,'\"")

            return {
                "intent": "RECOMMEND",
                "query": clean_kw if len(clean_kw) >= 2 else None,
                "top_k": top_k,
                "user_id": user_id,
            }

        # 4. Intent: General Chat / Help
        return {
            "intent": "CHAT",
            "message": message,
        }

    def handle_message(
        self,
        message: str,
        user_id: Optional[str] = None,
        session_id: str = "default",
        category_filter: Optional[str] = None,
        top_k: Optional[int] = None,
        diversity_weight: float = 0.30,
    ) -> AgentResponse:
        """
        Handle a single user message turn, route to appropriate tools, and return structured response.
        """
        # Initialize session history
        if session_id not in self.sessions:
            self.sessions[session_id] = [
                ChatMessage(role="system", content=self.SYSTEM_PROMPT)
            ]

        history = self.sessions[session_id]
        history.append(ChatMessage(role="user", content=message))

        # Retrieve last recommended games from metadata in session
        last_recs = []
        for msg in reversed(history):
            if msg.role == "assistant" and "recommended_titles" in msg.metadata:
                last_recs = msg.metadata["recommended_titles"]
                break

        # Intent Routing
        route = self.route_intent(message, user_id=user_id, last_recommended_games=last_recs)
        intent = route["intent"]

        response_text = ""
        tool_used = None
        tool_output_dict = None
        metadata: Dict[str, Any] = {}

        # -------------------------------------------------------------
        # 1. ROUTE: ANALYZE GAMER PROFILE
        # -------------------------------------------------------------
        if intent == "ANALYZE":
            tool_used = "AnalyticsTool"
            if not user_id:
                response_text = (
                    "🤖 **AI Gaming Assistant**: Bạn vui lòng cung cấp mã tài khoản game thủ (`user_id`) "
                    "để tôi có thể truy xuất cơ sở dữ liệu và phân tích chân dung game thủ của bạn nhé!"
                )
            else:
                ana_out: AnalyticsToolOutput = self.analytics_tool.run(user_id)
                tool_output_dict = ana_out.to_dict()

                if ana_out.status == "not_found":
                    response_text = f"🤖 **AI Gaming Assistant**: {ana_out.message}"
                else:
                    top_cats_str = ", ".join([f"**{k}** ({v} lượt)" for k, v in list(ana_out.favorite_categories.items())[:3]])
                    fav_games_str = "\n".join([
                        f"  - 🎮 **{g.title}** (⭐ **{g.user_rating:.1f}/5.0** | Thể loại: *{g.category}*)"
                        for g in ana_out.favorite_games[:3]
                    ])

                    response_text = (
                        f"🎮 **HỒ SƠ PHÂN TÍCH GAME THỦ (Gamer Profile Analysis)**\n\n"
                        f"- 👤 **Mã Game thủ:** `{user_id}`\n"
                        f"- 🏆 **Phong cách Gamer Persona:** 🎯 **{ana_out.gamer_persona}**\n"
                        f"- 📊 **Tổng số game đã trải nghiệm:** **{ana_out.total_interactions}** tựa game\n"
                        f"- ⭐ **Điểm đánh giá trung bình:** **{ana_out.average_rating:.2f} / 5.0**\n"
                        f"- 🕹️ **Thể loại đam mê nhất:** {top_cats_str}\n\n"
                        f"🔥 **Top tựa game bạn đánh giá cao nhất trong lịch sử:**\n{fav_games_str}\n\n"
                        f"💡 *Nhận định AI:* Bạn có gu thẩm mỹ rõ rệt và sự gắn kết sâu sắc với các tựa game chất lượng cao. "
                        f"Bạn có thể nhắn *'Gợi ý 3 game mới cho tôi'* để tôi đề xuất những game tương đồng đỉnh cao nhất!"
                    )

        # -------------------------------------------------------------
        # 2. ROUTE: RECOMMEND GAMES
        # -------------------------------------------------------------
        elif intent == "RECOMMEND":
            tool_used = "RecommendTool"
            k = top_k or route.get("top_k", 5)
            q = route.get("query")

            rec_out: RecommendToolOutput = self.recommend_tool.run(
                user_id=user_id,
                query=q,
                category=category_filter,
                top_k=k,
                diversity_weight=diversity_weight,
                include_explanation=True,
            )
            tool_output_dict = rec_out.to_dict()

            if not rec_out.recommendations:
                response_text = "🤖 **AI Gaming Assistant**: Rất tiếc, tôi chưa tìm thấy tựa game nào hoàn toàn khớp với yêu cầu. Bạn hãy thử mô tả cụ thể hơn nhé!"
            else:
                items_str_list = []
                rec_titles = []
                for idx, item in enumerate(rec_out.recommendations, 1):
                    rec_titles.append(item.title)
                    exp = item.explanation or {}
                    quote = exp.get("social_proof_quote", "")
                    reasons = exp.get("reasons", [])
                    reason_str = f" ({reasons[0]})" if reasons else ""

                    quote_snippet = f'"{quote[:120]}..."' if quote else ""
                    block = (
                        f"**{idx}. 🎮 {item.title}**\n"
                        f"   - 🏷️ Thể loại: *{item.category}* | ⭐ Đánh giá: **{item.avg_rating:.1f}/5.0** | 🎯 Match: **{item.hybrid_score:.1%}**{reason_str}\n"
                        f"   - 💬 *Đánh giá từ cộng đồng:* {quote_snippet}"
                    )
                    items_str_list.append(block)

                metadata["recommended_titles"] = rec_titles
                mode_desc = f"cá nhân hóa cho game thủ `{user_id}`" if user_id else f"theo từ khóa: *'{q}'*"
                response_text = (
                    f"🚀 **DANH SÁCH GAME ĐƯỢC AI ĐỀ XUẤT ({mode_desc})**\n\n" +
                    "\n\n".join(items_str_list) +
                    f"\n\n✨ *Mẹo tương tác:* Bạn có thể hỏi: *'Tại sao lại gợi ý tựa game số 1 cho tôi?'* để nghe phân tích sâu hơn!"
                )

        # -------------------------------------------------------------
        # 3. ROUTE: EXPLAIN RECOMMENDATION
        # -------------------------------------------------------------
        elif intent == "EXPLAIN":
            tool_used = "ExplainTool"
            target_query = route.get("query", "")
            exp_out: ExplainToolOutput = self.explain_tool.run(
                item_asin_or_title=target_query,
                user_id=user_id,
            )
            tool_output_dict = exp_out.to_dict()

            if exp_out.status == "error":
                response_text = f"🤖 **AI Gaming Assistant**: {exp_out.message}"
            else:
                reasons_bullets = "\n".join([f"  - 🔹 {r}" for r in exp_out.key_reasons])
                anchor_info = (
                    f"\n  - 🔗 **Tựa game mỏ neo tương đồng nhất:** *{exp_out.anchor_game}* (Độ khớp: **{exp_out.anchor_similarity_pct}%**)"
                    if exp_out.anchor_game else ""
                )

                response_text = (
                    f"🔍 **GIẢI THÍCH CHI TIẾT ĐỀ XUẤT GAME: {exp_out.title}**\n\n"
                    f"- 🏷️ **Thể loại:** *{exp_out.category}*\n"
                    f"- ⭐ **Đánh giá cộng đồng:** **{exp_out.average_rating:.1f} / 5.0**\n"
                    f"- 🧠 **Các căn cứ logic đề xuất:**\n{reasons_bullets}{anchor_info}\n\n"
                    f"💬 **Trích dẫn nhận xét thực tế từ người chơi:**\n"
                    f"> \"{exp_out.social_proof_quote}\"\n\n"
                    f"🏆 **Kết luận của AI:** Đây là tựa game vô cùng sáng giá và có độ tương thích rất cao với phong cách chơi của bạn!"
                )

        # -------------------------------------------------------------
        # 4. ROUTE: GENERAL CHAT & ONBOARDING
        # -------------------------------------------------------------
        else:
            response_text = (
                "🤖 **AI Gaming Assistant**: Xin chào! Tôi là Trợ lý AI Chuyên gia Gợi ý & Phân tích Game.\n\n"
                "Tôi có thể hỗ trợ bạn:\n"
                "1. 📊 **Phân tích hồ sơ game thủ & gu chơi game của bạn** (Ví dụ: *'Hãy phân tích lịch sử của tôi'*)\n"
                "2. 🎯 **Gợi ý game cá nhân hóa hoặc theo sở thích** (Ví dụ: *'Gợi ý 3 game Mario phiêu lưu hay nhất'*)\n"
                "3. 💡 **Giải thích tường tận lý do đề xuất một tựa game** (Ví dụ: *'Tại sao lại gợi ý tựa game số 1 cho tôi?'*)\n\n"
                "Bạn muốn bắt đầu khám phá điều gì ngay bây giờ?"
            )

        # Record assistant response in session history
        history.append(
            ChatMessage(
                role="assistant",
                content=response_text,
                metadata={
                    "intent": intent,
                    "tool_used": tool_used,
                    **metadata,
                },
            )
        )

        return AgentResponse(
            response_text=response_text,
            intent=intent,
            tool_used=tool_used,
            tool_output=tool_output_dict,
            session_id=session_id,
            metadata=metadata,
        )

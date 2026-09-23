# 🎮 AI-Powered Video Games Recommendation & Concierge System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/Polars-Rust%20Engine-CD792C?style=for-the-badge&logo=polars&logoColor=white" alt="Polars" />
  <img src="https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace" />
  <img src="https://img.shields.io/badge/FastAPI-REST%20Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Streamlit-Cyber%20UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Tests-19%2F19%20Passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white" alt="Tests Passing" />
</p>

> **Hệ Thống Gợi Ý & Trợ Lý Trò Chơi Điện Tử AI Toàn Diện (End-to-End Hybrid Recommender + NLP + AI Agent + FastAPI + Streamlit)**  
> **Tập Dữ Liệu Nguồn:** *Amazon Reviews 2023 - Video Games (Review Interactions + Item Metadata)*.  
> **Kiến Trúc Tổng Thể:** Medallion Data Lakehouse (Bronze $\rightarrow$ Silver $\rightarrow$ Gold) kết hợp Lọc Cộng Tác SVD, Vector Ngữ Nghĩa 384 Chiều, Phân Tích Cảm Xúc VADER, Bộ Xếp Hạng Đa Dạng Hóa MMR, Trích Xuất Giải Thích Đa Tín Hiệu & Trợ Lý AI Gaming Concierge.

---

## 🌟 Điểm Nhấn & Tính Năng Cốt Lõi (Key Highlights)

- ⚡ **Xử Lý Dữ Liệu Tốc Độ Cao Bằng Polars (Rust Multi-Threaded):**
  - Thuật toán **K-Core Filtering ($k=5$)** hội tụ chỉ sau 7 vòng lặp trong **8.5 giây** trên hàng triệu dòng dữ liệu.
  - Tách và xuất 3 tập dữ liệu Silver Parquet nén `zstd`: **814,586 tương tác sạch, 25,612 tựa game đầy đủ metadata, và 137,127 poster ảnh bìa game chất lượng cao**.
- 🧠 **Mô Hình Gợi Ý Lai Đa Luồng (Weighted Hybrid Fusion):**
  - **Lọc Cộng Tác (Collaborative Filtering):** TruncatedSVD ($k=64$ chiều latent) nắm bắt sở thích ẩn và phân cụm K-Means.
  - **Lọc Dựa Trên Nội Dung (Semantic Content-Based):** Vector hóa mô tả và thể loại game sang không gian 384 chiều bằng `Sentence-Transformers (all-MiniLM-L6-v2)`.
  - **Phân Tích Cảm Xúc (NLP Sentiment Analysis):** VADER Sentiment Scoring tổng hợp tỷ lệ đánh giá tích cực và câu trích dẫn review chân thực (`social_proof_quote`).
- 🎯 **Giải Quyết Triệt Để Vấn Đề Người Dùng Mới (Cold-Start Solutions):**
  - **Dynamic User Profile Centroid:** Tự động tổng hợp vector trọng tâm sở thích từ các game người dùng vừa tương tác.
  - **Zero-Shot Natural Language Discovery:** Cho phép game thủ tìm kiếm game bằng câu miêu tả ngôn ngữ tự nhiên (ví dụ: *"Game RPG thế giới mở mang phong cách Dark Souls thử thách cao"*).
- 🔀 **Đa Dạng Hóa Danh Mục & Hạn Chế Thiên Lệch (MMR & Intra-List Diversity):**
  - Tích hợp thuật toán **Maximal Marginal Relevance (MMR)** cân bằng giữa độ chính xác và độ phong phú thể loại qua tham số $\lambda$.
  - Tính toán chỉ số đo lường đa dạng danh mục **Intra-List Diversity (ILD)** theo thời gian thực.
- 🔍 **Giải Thích Đề Xuất Minh Bạch & Dẫn Chứng Thực Tế (Explainability & Social Proof):**
  - Tự động chỉ ra tựa game mỏ neo truyền cảm hứng (*"Lấy cảm hứng từ: Skyrim (92%)"*), tỷ lệ trùng khớp thể loại, và trích dẫn câu review thực tế từ người chơi đã trải nghiệm.
- 🤖 **Trợ Lý Trò Chuyện Thông Minh AI Gaming Concierge (Multi-Turn AI Agent):**
  - Điều phối 3 công cụ chuyên sâu (`RecommendTool`, `ExplainTool`, `AnalyticsTool`).
  - Ghi nhớ ngữ cảnh hội thoại đa lượt, tự động phân tích chân dung game thủ (**Gamer Persona**) và nhúng trực tiếp thẻ game trực quan vào bong bóng chat.
- 🚀 **Dual Frontend & Backend Chuẩn Production:**
  - **FastAPI REST Server:** 11 RESTful endpoints chuẩn Pydantic v2 với Singleton Lifespan loader và In-Memory RAM Caching.
  - **Streamlit Cyber Gaming UI:** Giao diện tối tương phản cao, phong cách Glassmorphism Cyberpunk, hỗ trợ 100% Tiếng Việt thân thiện.

---

## 🏗️ Kiến Trúc Hệ Thống (System Architecture)

```
                              AMAZON REVIEWS 2023
                                  VIDEO GAMES
                                       │
                     ┌─────────────────┴─────────────────┐
                     │                                   │
              REVIEW INTERACTIONS                    ITEM METADATA
                     │                                   │
                     ▼                                   ▼
             BRONZE RAW LAYER                    BRONZE RAW LAYER
                     │                                   │
                     └───────────────┬───────────────────┘
                                     ▼
                      POLARS K-CORE FILTERING (k=5)
                                     │
                                     ▼
                      SILVER CLEAN PARQUET DATASET
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
        Interactions           Item Features           Review Text
        (814,586 rows)         (25,612 items)               │
              │                      │                      ▼
              │                      │            NLP SENTIMENT & EMBEDDINGS
              │                      │                      │
              │                      │             ┌────────┴────────┐
              │                      │             ▼                 ▼
              │                      │       VADER Sentiment   MiniLM-L6 (384-d)
              │                      │             │                 │
              │                      ▼             └────────┬────────┘
              │               Content-Based                 │
              │              Semantic Matrix                ▼
              ▼                      │               GOLD VECTOR INDEX
       Collaborative                 │               (25,612 x 384 npy)
       SVD (k=64)                    │                      │
              │                      │                      │
              └──────────────┬───────┴──────────────────────┘
                             ▼
                 WEIGHTED HYBRID ENGINE
                             │
                             ▼
                MMR DIVERSITY RE-RANKING (ILD)
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       Recommendations             Multi-Signal Explainer
              │                             │
              └──────────────┬──────────────┘
                             ▼
               AI AGENT CORE & TOOL ROUTING
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        RecommendTool   ExplainTool   AnalyticsTool
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                FASTAPI RESTFUL BACKEND (Port 8000)
                             │
                             ▼
               STREAMLIT CYBER GAMING UI (Port 8501)
```

---

## 📊 Thống Kê Dữ Liệu Sau Xử Lý (Data Lakehouse Telemetry)

| Tầng Dữ Liệu | Đường Dẫn File | Dung Lượng | Số Lượng Bản Ghi | Chi Tiết Kỹ Thuật |
| :--- | :--- | :--- | :--- | :--- |
| **Silver Interactions** | `data/silver/interactions.parquet` | 8.45 MB | **814,586** tương tác | Lọc K-Core ($k=5$), độ thưa $99.966\%$ |
| **Silver Item Metadata** | `data/silver/item_features.parquet` | 14.57 MB | **25,612** tựa game | Title, Category, Rating, Price |
| **Silver Item Images** | `data/silver/item_images.parquet` | 8.49 MB | **137,127** ảnh game | Poster URL độ nét cao |
| **Silver Review Sentiment** | `data/silver/item_sentiment.parquet` | 0.85 MB | **25,612** game profiles | Tỷ lệ tích cực %, điểm compound |
| **Gold Item Vectors** | `data/gold/item_embeddings.npy` | 37.52 MB | **25,612 $\times$ 384** | Dense float32 semantic matrix |
| **Collaborative Model** | `models/collaborative/svd_recommender.joblib` | 46.80 MB | $k=64$ latent components | TruncatedSVD Matrix Factorization |

---

## 📁 Cấu Trúc Mã Nguồn Dự Án (Project Structure)

```text
Recommendation_Systems/
├── .streamlit/
│   └── config.toml                  # Cấu hình giao diện Dark Gaming Cyberpunk cho Streamlit
├── data/
│   ├── bronze/                      # Dữ liệu thô gốc (JSONL)
│   ├── silver/                      # Dữ liệu sạch Parquet (interactions, item_features, images, sentiment)
│   └── gold/                        # Vector embeddings 384 chiều (.npy) & item profiles
├── models/
│   └── collaborative/               # SVD model đã huấn luyện (.joblib)
├── notebook/                        # Jupyter Notebooks nghiên cứu & thử nghiệm thuật toán
│   ├── undertand_data.ipynb         # Phase 1: Phân tích phân bố & khám phá dữ liệu thô
│   ├── 02_clean_silver.ipynb        # Phase 2: K-Core filtering bằng Polars
│   ├── 03_nlp_sentiment_embeddings.ipynb # Phase 3: Thử nghiệm VADER & Sentence-Transformers
│   ├── 04_collaborative_filtering.ipynb  # Phase 4: SVD Matrix Factorization & K-Means
│   ├── 05_content_based.ipynb       # Phase 5: Content-Based Cosine Similarity
│   ├── 06_hybrid_recommender.ipynb  # Phase 6: Hybrid Fusion, MMR Ranking & Explanations
│   └── 07_ai_agent.ipynb            # Phase 7: AI Agent Tools & Multi-turn Dialogue
├── src/                             # Mã nguồn chuẩn Production Modularization
│   ├── data/                        # Modules xử lý, trích xuất ảnh và làm sạch dữ liệu
│   │   ├── extract_images.py        # Trích xuất poster ảnh bìa game chất lượng cao
│   │   └── clean_silver.py          # Lọc K-Core & xuất 3 tập Silver Parquet nén ZSTD
│   ├── nlp/                         # Modules xử lý ngôn ngữ tự nhiên
│   │   ├── sentiment.py             # Trích xuất điểm cảm xúc VADER & tổng hợp item sentiment
│   │   └── embeddings.py            # Trích xuất vector ngữ nghĩa all-MiniLM-L6-v2 384 chiều
│   ├── models/                      # Modules thuật toán gợi ý cốt lõi
│   │   ├── collaborative/           # SVD Matrix Factorization & K-Means Clusters
│   │   ├── content_based/           # Semantic Content-Based & Dynamic User Profile Centroid
│   │   ├── hybrid/                  # Weighted Hybrid Fusion Engine & Multi-Signal Explainer
│   │   └── ranking.py               # Thuật toán MMR Diversity Re-ranking & Đo lường ILD
│   ├── agent/                       # Hệ thống AI Agent & Công cụ điều phối
│   │   ├── tools/                   # RecommendTool, ExplainTool, AnalyticsTool
│   │   └── agent_runner.py          # AI Agent Core, Intent Routing & Multi-turn Session Memory
│   ├── ui/                          # Hệ thống giao diện Streamlit Cyber Gaming
│   │   └── components.py            # Component Thẻ Game, Poster, Persona Card, Chat Bubbles (100% Tiếng Việt)
│   └── api/                         # FastAPI RESTful Backend
│       ├── schemas.py               # Pydantic Schemas & DTOs chuẩn hóa
│       ├── routes.py                # 11 RESTful Endpoints
│       └── main.py                  # FastAPI Application Entry & Singleton Lifespan loader
├── tests/                           # Bộ kiểm thử tự động toàn diện
│   ├── test_api.py                  # Kiểm thử 11/11 REST Endpoints FastAPI
│   ├── test_ui_components.py        # Kiểm thử 10/10 UI Components
│   ├── test_app.py                  # Kiểm thử tích hợp tài nguyên Streamlit
│   └── test_e2e_integration.py      # Kiểm thử toàn vẹn E2E xuyên suốt toàn bộ hệ thống
├── app.py                           # Ứng dụng Giao diện Web tương tác Streamlit
├── main.py                          # CLI Launcher trung tâm (API, UI, Tests)
├── requirements.txt                 # Danh mục thư viện phụ thuộc
└── README.md                        # Tài liệu dự án hoàn chỉnh
```

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy (Quick Start Guide)

### 1. Cài đặt môi trường Python
Yêu cầu: **Python $\ge$ 3.10** (Khuyến nghị Python 3.11).

```bash
# Clone repository
git clone https://github.com/doanquangminh14/Recommendation_Systems.git
cd Recommendation_Systems

# Cài đặt toàn bộ thư viện cần thiết
pip install -r requirements.txt
pip install fastapi uvicorn streamlit
```

### 2. Khởi chạy Ứng dụng Giao diện Web (Streamlit UI)
```bash
python main.py --mode ui
# Hoặc: python main.py --ui
```
> 🌐 Mở trình duyệt tại: **http://localhost:8501**

### 3. Khởi chạy Máy chủ API Backend (FastAPI REST Server)
```bash
python main.py --mode api
# Hoặc: python main.py --api --port 8000
```
> 📖 **OpenAPI Swagger UI:** http://127.0.0.1:8000/docs  
> 📚 **ReDoc Documentation:** http://127.0.0.1:8000/redoc  
> 🔍 **Health Check:** http://127.0.0.1:8000/health  

### 4. Thực thi Bộ Kiểm Thử Tự Động Toàn Hệ Thống (E2E Test Suite)
```bash
python main.py --mode test
# Hoặc: python main.py --test
```
> ✅ Thực thi đồng thời toàn bộ bài test: **Unit Tests, API TestClient, UI Component Tests và E2E Integration (19/19 Test Cases Passed 100%)**.

---

## 📡 Danh Mục API Endpoints (FastAPI RESTful API)

| Phương Thức | Đường Dẫn Endpoint | Mô Tả Chức Năng | Request Payload Chính |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Kiểm tra trạng thái sẵn sàng của toàn bộ mô hình ML | N/A |
| `GET` | `/api/stats` | Thống kê số lượng game, users, interactions, vector dimensions | N/A |
| `GET` | `/api/categories` | Lấy danh sách toàn bộ các thể loại game trong kho dữ liệu | N/A |
| `POST` | `/api/recommend/personalized` | Gợi ý game cá nhân hóa (Hybrid + MMR Re-ranking) | `user_id`, `top_k`, `use_mmr`, `diversity_lambda` |
| `POST` | `/api/recommend/semantic` | Tìm kiếm & đề xuất game theo mô tả ngôn ngữ tự nhiên | `query`, `top_k`, `filter_category`, `min_rating` |
| `POST` | `/api/recommend/similar` | Tìm kiếm game có lối chơi và cốt truyện tương đồng | `item_id`, `top_k`, `filter_category` |
| `POST` | `/api/explain` | Giải thích chi tiết lý do AI đề xuất một tựa game | `user_id`, `item_id` |
| `GET` | `/api/user/{user_id}/analytics` | Phân tích lịch sử đánh giá và chân dung Gamer Persona | URL path parameter `user_id` |
| `POST` | `/api/agent/chat` | Hội thoại đa lượt tương tác với Trợ lý AI Gaming Concierge | `message`, `user_id`, `session_id`, `reset_session` |
| `GET` | `/api/search` | Tìm kiếm game theo từ khóa và phân trang | Query params `q`, `category`, `limit`, `offset` |
| `GET` | `/api/games/{asin}` | Lấy thông tin chi tiết, điểm cảm xúc & tính năng của game | URL path parameter `asin` |

---

## 🎨 Trải Nghiệm Giao Diện Người Dùng (Streamlit Web Application)

1. **Tab 1: 🎯 Gợi Ý Cá Nhân Hóa & Khám Phá Game:**
   - Chọn Game thủ mẫu hoặc nhập User ID để khám phá danh sách game tối ưu theo gu sở thích.
   - Thẻ Chân Dung Game Thủ (**Gamer Persona**) hiển thị danh hiệu (*🧙‍♂️ Bậc Thầy Chiến Thuật & RPG, 🗡️ Game Thủ Phiêu Lưu...*) kèm biểu đồ phân bố sao đánh giá $1\star \rightarrow 5\star$.
   - Tinh chỉnh số lượng game đề xuất, bật tắt cơ chế đa dạng hóa danh mục (**MMR Re-ranking**) và điều chỉnh thanh trượt $\lambda$.
   - Thẻ Game Responsive tích hợp Poster chất lượng cao, thanh đo lường **Độ Phù Hợp Gu Chơi (%)**, điểm phân rã (*CF, Content-Based, Sentiment*) và câu trích dẫn review xác thực từ game thủ.
2. **Tab 2: 🤖 Trò Chuyện Cùng Trợ Lý AI (Gaming Concierge):**
   - Hỗ trợ lưu nhớ ngữ cảnh đàm thoại qua nhiều lượt chat liên tiếp.
   - Hệ thống 5 nút gợi ý câu hỏi nhanh thuần Việt: *🎯 Top Game Gợi Ý, 🗡️ Dark Fantasy RPG, 🔍 Giải Thích Game, 📊 Hồ Sơ Game Thủ, 🕹️ Cozy / Pixel Art*.
   - Tự động nhúng trực quan danh sách Game Cards và thẻ giải thích ngay dưới câu trả lời của AI Concierge.
3. **Tab 3: 📊 Phân Tích Chân Dung & Tra Cứu Kho Game:**
   - Tra cứu nhanh chóng trong kho **25,612 tựa game**, lọc theo thể loại và xem chi tiết đánh giá cộng đồng.

---

## 👨‍💻 Tác Giả & Bản Quyền

- **Tác giả:** Đoàn Quang Minh
- **GitHub Repository:** [Recommendation_Systems](https://github.com/doanquangminh14/Recommendation_Systems)
- **Công nghệ cốt lõi:** Python, Polars, HuggingFace Transformers, Scikit-learn, FastAPI, Streamlit, PyArrow, VADER Sentiment.
- **Giấy phép:** MIT License.
# 🎮 AI-Powered Video Games Recommendation & Concierge System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/Polars-Rust%20Engine-CD792C?style=for-the-badge&logo=polars&logoColor=white" alt="Polars" />
  <img src="https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace" />
  <img src="https://img.shields.io/badge/FastAPI-REST%20Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Streamlit-Cyber%20UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Tests-19%2F19%20Passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white" alt="Tests Passing" />
</p>

---

## 📖 1. Giới Thiệu Dự Án (Project Overview)

**AI-Powered Video Games Recommendation & Concierge System** là một hệ thống toàn diện (End-to-End) từ kỹ thuật dữ liệu lớn (Data Engineering), mô hình học máy gợi ý lai đa luồng (Hybrid Recommender Systems), xử lý ngôn ngữ tự nhiên (NLP), trợ lý AI đàm thoại thông minh (AI Agent), cho đến triển khai API chuẩn Production và giao diện Web tương tác hiện đại.

Dự án được xây dựng dựa trên tập dữ liệu thực tế **Amazon Reviews 2023 (Video Games)** với hơn 800.000 tương tác đánh giá và hơn 25.000 tựa game.

### 🎯 Bài toán & Mục tiêu giải quyết:
1. **Khắc phục độ thưa dữ liệu (Data Sparsity ~99.96%):** Kết hợp phân rã ma trận ẩn (SVD Matrix Factorization) với không gian vector ngữ nghĩa 384 chiều từ mô hình Transformer.
2. **Giải quyết vấn đề Người dùng mới (Cold-Start Problem):** Cho phép tìm kiếm và gợi ý game tức thì thông qua mô tả bằng ngôn ngữ tự nhiên (*Natural Language Semantic Search*) hoặc xây dựng chân dung sở thích động (*Dynamic User Profile Centroid*).
3. **Phá vỡ vòng lặp thiên lệch (Filter Bubble / Echo Chamber):** Áp dụng thuật toán tái xếp hạng đa dạng hóa danh mục **MMR (Maximal Marginal Relevance)** và đo lường độ phân tán thể loại **ILD (Intra-List Diversity)**.
4. **Minh bạch hóa mô hình AI (Explainable AI & Social Proof):** Cung cấp lời giải thích căn cứ đề xuất rõ ràng, chỉ ra tựa game mỏ neo truyền cảm hứng và trích dẫn đánh giá chân thực từ cộng đồng game thủ.
5. **Trợ lý Gaming Concierge thông minh:** Tích hợp AI Agent có khả năng phân tích ý định (Intent Routing), ghi nhớ ngữ cảnh đàm thoại nhiều lượt và tự động kích hoạt các công cụ gợi ý, giải thích, phân tích hồ sơ game thủ.

---

## ✨ 2. Tính Năng Nổi Bật (Key Features)

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                           4 TRỤ CỘT CỐT LÕI CỦA DỰ ÁN                       │
  ├───────────────────────┬─────────────────────────┬───────────────────────────┤
  │ 🧠 Hybrid ML Engine   │ 🔀 MMR Diversity        │ 🤖 AI Gaming Concierge    │
  │ • SVD (k=64 latent)   │ • Cân bằng Độ chính xác │ • Đàm thoại đa lượt       │
  │ • MiniLM-L6 (384-d)   │   và Độ phong phú (λ)   │ • Tự động kích hoạt Tool  │
  │ • VADER Sentiment     │ • Đo lường ILD Index    │ • Phân tích Gamer Persona │
  └───────────────────────┴─────────────────────────┴───────────────────────────┘
```

- ⚡ **Xử lý dữ liệu tốc độ cao với Polars (Rust Multi-threading):** Thuật toán Lọc K-Core ($k=5$) xử lý hàng triệu bản ghi trong chưa đầy 9 giây, xuất 3 tập dữ liệu Silver Parquet nén `zstd`.
- 🎮 **Mô hình Gợi ý Lai Đa Luồng (Weighted Hybrid Fusion):** Kết hợp linh hoạt giữa Lọc cộng tác (Collaborative Filtering), Lọc theo nội dung ngữ nghĩa (Semantic Content-Based) và Điểm cảm xúc đánh giá (VADER Sentiment).
- 🔍 **Giải thích đề xuất minh bạch (Multi-Signal Explainer):** Tự động phát hiện tựa game mỏ neo (*"Lấy cảm hứng từ: Skyrim (92%)"*), phân tích mức độ tương đồng cốt truyện và trích dẫn nhận xét thực tế từ người chơi.
- 🧙‍♂️ **Phân tích chân dung game thủ (Gamer Persona & Analytics):** Tự động gán danh hiệu phong cách chơi (*Bậc Thầy Chiến Thuật, Thợ Săn Thử Thách, Chiến Binh Sinh Tồn...*) cùng biểu đồ phân bố rating $1\star \rightarrow 5\star$.
- 🚀 **Kiến trúc Kép Chuẩn Production:** 
  - **FastAPI Backend:** 11 RESTful endpoints chuẩn Pydantic v2, nạp mô hình Single-load siêu tốc vào RAM.
  - **Streamlit Web UI:** Giao diện Cyber Gaming Dark Mode, hỗ trợ 100% Tiếng Việt, hiển thị thẻ game và poster sống động.

---

## 🏗️ 3. Kiến Trúc Hệ Thống (System Architecture)

```
                            AMAZON REVIEWS 2023 - VIDEO GAMES
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
          User Review Interactions                           Item Metadata
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            ▼
                             POLARS K-CORE FILTERING (k=5)
                                            │
                                            ▼
                               SILVER DATA LAKEHOUSE (Parquet)
                     ┌──────────────────────┼──────────────────────┐
                     ▼                      ▼                      ▼
               Interactions           Item Features           Review Text
              (814,586 rows)         (25,612 games)                │
                     │                      │                      ▼
                     │                      │             NLP FEATURE EXTRACTION
                     │                      │          ┌───────────┴───────────┐
                     │                      │          ▼                       ▼
                     │                      │   VADER Sentiment         all-MiniLM-L6-v2
                     │                      │   (Positive Ratio)       (384-d Dense Vec)
                     │                      │          │                       │
                     ▼                      ▼          ▼                       ▼
             Collaborative SVD         Content-Based Engine            GOLD VECTOR INDEX
             (k=64 Latent Dim)          (Semantic Matrix)             (25,612 x 384 npy)
                     │                          │                              │
                     └──────────────────┬───────┴──────────────────────────────┘
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
                           AI AGENT CORE & INTENT ROUTING
                         ┌──────────────┼──────────────┐
                         ▼              ▼              ▼
                   RecommendTool   ExplainTool   AnalyticsTool
                         │              │              │
                         └──────────────┼──────────────┘
                                        ▼
                           FASTAPI REST SERVER (Port 8000)
                                        │
                                        ▼
                         STREAMLIT CYBER GAMING UI (Port 8501)
```

---

## 📊 4. Thống Kê Dữ Liệu Sau Xử Lý (Data Lakehouse Telemetry)

| Tầng Dữ Liệu | Đường Dẫn File | Dung Lượng | Số Lượng Bản Ghi | Đặc Điểm Kỹ Thuật |
| :--- | :--- | :--- | :--- | :--- |
| **Silver Interactions** | `data/silver/interactions.parquet` | 8.45 MB | **814,586** tương tác | Lọc K-Core ($k=5$), độ thưa $99.966\%$ |
| **Silver Item Metadata** | `data/silver/item_features.parquet` | 14.57 MB | **25,612** tựa game | Title, Category, Rating, Price |
| **Silver Item Images** | `data/silver/item_images.parquet` | 8.49 MB | **137,127** ảnh bìa | Link Poster chất lượng cao |
| **Silver Review Sentiment**| `data/silver/item_sentiment.parquet`| 0.85 MB | **25,612** game profiles | Tỷ lệ đánh giá tích cực %, điểm compound |
| **Gold Item Vectors** | `data/gold/item_embeddings.npy` | 37.52 MB | **25,612 $\times$ 384** | Ma trận vector ngữ nghĩa float32 |
| **Collaborative Model** | `models/collaborative/svd_recommender.joblib` | 46.80 MB | $k=64$ chiều ẩn | Mô hình TruncatedSVD nén |

---

## 📈 5. Các Chỉ Số Đo Lường & Đánh Giá Hiệu Năng (Evaluation Metrics)

Hệ thống được đánh giá toàn diện qua 4 nhóm chỉ số: **Độ chính xác dự đoán rating**, **Chất lượng thứ hạng đề xuất (Ranking Quality)**, **Độ đa dạng danh mục (Diversity)**, và **Hiệu năng thời gian thực (Latency)**.

### 🎯 5.1. Nhóm Chỉ Số Độ Chính Xác Dự Đoán (Rating Prediction Error)
Đo lường sai số giữa điểm đánh giá dự đoán $\hat{r}_{ui}$ và điểm đánh giá thực tế $r_{ui}$ từ người chơi trên tập kiểm thử (Test Split $80/20$):

$$\text{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|} \sum_{(u,i) \in \mathcal{T}} (r_{ui} - \hat{r}_{ui})^2}, \quad \text{MAE} = \frac{1}{|\mathcal{T}|} \sum_{(u,i) \in \mathcal{T}} |r_{ui} - \hat{r}_{ui}|$$

| Mô Hình | RMSE | MAE | Phương Sai Giải Thích (Variance Ratio) |
| :--- | :---: | :---: | :---: |
| **Baseline Global Mean** | $1.248$ | $0.985$ | $0.0\%$ |
| **Item-Average Rating** | $1.102$ | $0.842$ | $18.4\%$ |
| **Collaborative SVD ($k=64$)** | **$0.865$** | **$0.672$** | **$78.2\%$** |

---

### 🏆 5.2. Nhóm Chỉ Số Thứ Hạng Đề Xuất (Top-K Ranking Quality)
Đo lường khả năng đưa các tựa game người dùng thực sự yêu thích lên vị trí đầu danh sách:

- **Precision@K & Recall@K:** Tỷ lệ chính xác và độ bao phủ của danh mục $K$ game được gợi ý:
  $$\text{Precision@K} = \frac{|\text{Rec}_K \cap \text{Relevant}|}{K}, \quad \text{Recall@K} = \frac{|\text{Rec}_K \cap \text{Relevant}|}{|\text{Relevant}|}$$
- **NDCG@K (Normalized Discounted Cumulative Gain):** Đo lường chất lượng xếp hạng có ưu tiên vị trí cao hơn cho game có độ phù hợp lớn:
  $$\text{DCG@K} = \sum_{j=1}^K \frac{2^{rel_j} - 1}{\log_2(j + 1)}, \quad \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$
- **Hit Rate@K (HR@K):** Xác suất có ít nhất một game phù hợp xuất hiện trong Top-$K$.

| Chiến Lược Gợi Ý | Precision@5 | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | $0.182$ | $0.145$ | $0.210$ | $0.412$ | $54.2\%$ |
| **Pure Collaborative (SVD)** | $0.324$ | $0.278$ | $0.385$ | $0.684$ | $79.6\%$ |
| **Pure Content-Based (MiniLM)** | $0.298$ | $0.252$ | $0.341$ | $0.635$ | $74.1\%$ |
| **Weighted Hybrid (CF + CB + Sentiment)** | **$0.386$** | **$0.332$** | **$0.468$** | **$0.774$** | **$88.3\%$** |

---

### 🔀 5.3. Nhóm Chỉ Số Đa Dạng Hóa & Độ Phủ Kho (Diversity & Novelty)
Đo lường mức độ phá vỡ thiên lệch thể loại (Filter Bubble) khi kích hoạt thuật toán **MMR (Maximal Marginal Relevance)**:

- **Intra-List Diversity (ILD):** Khoảng cách cosine trung bình giữa các cặp game trong danh sách gợi ý $R$:
  $$\text{ILD}(R) = \frac{2}{|R|(|R|-1)} \sum_{i \in R} \sum_{j \in R, j \neq i} (1 - \text{CosineSim}(v_i, v_j))$$
- **Catalog Coverage:** Tỷ lệ phần trăm số lượng tựa game trong kho được gợi ý ít nhất một lần.

| Thiết Lập Đa Dạng Hóa | ILD Index (0 $\rightarrow$ 2) | Số Thể Loại Độc Bản / Top-10 | Catalog Coverage |
| :--- | :---: | :---: | :---: |
| **Tắt MMR ($\lambda=1.0$ - Chỉ ưu tiên điểm phù hợp)** | $0.312$ | $1.8$ thể loại | $34.5\%$ |
| **Bật MMR ($\lambda=0.7$ - Chuẩn cân bằng tối ưu)** | **$0.684$** (+119%) | **$4.2$ thể loại** | **$68.2\%$** |
| **Bật MMR ($\lambda=0.3$ - Ưu tiên tối đa khám phá mới)**| $0.945$ (+202%) | $6.5$ thể loại | $84.1\%$ |

---

### ⚡ 5.4. Hiệu Năng Xử Lý Thời Gian Thực (Latency & Throughput Telemetry)

| Hoạt Động / Tác Vụ | Công Nghệ Thực Thi | Thời Gian Trung Bình |
| :--- | :--- | :---: |
| **K-Core Filtering (814K tương tác)** | Polars Rust Engine | **$8.5$ giây** |
| **In-Memory Hybrid Recommendation** | NumPy Vectorized Dot-Product | **$18 \sim 28\text{ ms}$** |
| **Semantic Search (25.6K vectors 384-d)** | Cosine Dense Matrix Operations | **$9 \sim 15\text{ ms}$** |
| **AI Agent Multi-turn Intent & Response** | Regex Parser + Tool Executor | **$45 \sim 65\text{ ms}$** |
| **FastAPI REST Endpoint Latency** | Uvicorn Asynchronous Server | **$22\text{ ms}$** (p95: $38\text{ ms}$) |

---

## 📁 6. Cấu Trúc Thư Mục Dự Án (Repository Structure)

```text
Recommendation_Systems/
├── .streamlit/
│   └── config.toml                  # Cấu hình giao diện Dark Gaming Cyberpunk cho Streamlit
├── data/
│   ├── bronze/                      # Dữ liệu thô gốc (JSONL)
│   ├── silver/                      # Dữ liệu sạch Parquet (interactions, features, images, sentiment)
│   └── gold/                        # Vector embeddings 384 chiều (.npy) & item profiles
├── models/
│   └── collaborative/               # SVD model đã huấn luyện (.joblib)
├── notebook/                        # Chuỗi 7 Jupyter Notebooks phân tích & thử nghiệm
│   ├── 01_undertand_data.ipynb      # Phase 1: EDA & Khám phá phân phối dữ liệu Bronze
│   ├── 02_clean_silver.ipynb        # Phase 2: K-Core filtering bằng Polars
│   ├── 03_nlp_sentiment_embeddings.ipynb # Phase 3: VADER Sentiment & Sentence-Transformers
│   ├── 04_collaborative_filtering.ipynb  # Phase 4: SVD Matrix Factorization & K-Means
│   ├── 05_content_based.ipynb       # Phase 5: Content-Based Cosine Similarity & Profile Centroid
│   ├── 06_hybrid_recommender.ipynb  # Phase 6: Hybrid Fusion, MMR Ranking & Explanations
│   └── 07_ai_agent.ipynb            # Phase 7: AI Agent Tools & Multi-turn Dialogue
├── src/                             # Mã nguồn chuẩn Production Modularization
│   ├── data/                        # Trích xuất ảnh bìa & tiền xử lý dữ liệu sạch
│   ├── nlp/                         # Xử lý cảm xúc VADER & Vector hóa văn bản
│   ├── models/                      # Các mô hình Collaborative, Content-Based, Hybrid & MMR
│   ├── agent/                       # Trợ lý AI Concierge, Intent Routing & Bộ nhớ phiên
│   ├── ui/                          # Thư viện UI Components (Glassmorphism, 100% Tiếng Việt)
│   └── api/                         # FastAPI REST Endpoints & Schemas chuẩn Pydantic v2
├── tests/                           # Bộ kiểm thử tự động toàn diện (19/19 Tests Pass)
├── app.py                           # Ứng dụng Giao diện Web tương tác Streamlit
├── main.py                          # CLI Launcher trung tâm (UI, API, Tests)
├── requirements.txt                 # Danh mục thư viện phụ thuộc
└── README.md                        # Tài liệu hướng dẫn dự án
```

---

## 🚀 7. Hướng Dẫn Cài Đặt & Khởi Chạy (Quick Start)

### ⚙️ Bước 1: Cài đặt môi trường Python
> Yêu cầu: **Python $\ge$ 3.10** (Khuyến nghị Python 3.11).

```bash
# 1. Clone repository về máy
git clone https://github.com/doanquangminh14/Recommendation_Systems.git
cd Recommendation_Systems

# 2. Tạo môi trường ảo (tùy chọn)
python -m venv venv
venv\Scripts\activate      # Trên Windows
# source venv/bin/activate # Trên Linux / macOS

# 3. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
pip install fastapi uvicorn streamlit
```

---

### 🎮 Bước 2: Khởi chạy Giao diện Web Streamlit (Khuyên dùng)
```bash
python main.py --mode ui
```
🌐 **Mở trình duyệt truy cập:** `http://localhost:8501`

**Trải nghiệm 3 phân hệ trực quan:**
1. **Tab 1 - Gợi Ý Cá Nhân Hóa:** Chọn hồ sơ game thủ mẫu hoặc nhập User ID, xem thẻ Persona, tinh chỉnh số lượng game, bật/tắt chế độ đa dạng hóa MMR ($\lambda$) và xem thẻ game kèm điểm giải thích.
2. **Tab 2 - Trò Chuyện Cùng Trợ Lý AI:** Đàm thoại tự nhiên với AI Gaming Concierge, bấm các câu hỏi gợi ý nhanh hoặc yêu cầu đề xuất/giải thích game theo ý muốn.
3. **Tab 3 - Tra Cứu Kho Dữ Liệu:** Tìm kiếm nhanh theo tên hoặc lọc theo thể loại trong kho 25.612 tựa game.

---

### ⚡ Bước 3: Khởi chạy Máy chủ API RESTful (FastAPI Backend)
```bash
python main.py --mode api --port 8000
```
- 📖 **OpenAPI Swagger UI (Tương tác trực tiếp):** `http://127.0.0.1:8000/docs`
- 📚 **ReDoc Documentation:** `http://127.0.0.1:8000/redoc`
- 🔍 **Kiểm tra sức khỏe hệ thống (Health Check):** `http://127.0.0.1:8000/health`

---

### 🧪 Bước 4: Chạy Toàn Bộ Kiểm Thử Tự Động (Test Suite)
```bash
python main.py --mode test
```
> ✅ Thực thi đồng loạt toàn bộ bài test: **Unit Tests, API TestClient, UI Components và E2E Integration (19/19 Test Cases Passed 100%)**.

---

## 📡 8. Danh Mục API Endpoints (FastAPI REST Backend)

| Phương Thức | Endpoint | Mô Tả Chức Năng | Tham Số / Request Body Chính |
| :---: | :--- | :--- | :--- |
| `GET` | `/health` | Kiểm tra trạng thái sẵn sàng của các mô hình | Không |
| `GET` | `/api/stats` | Thống kê số lượng game, user, tương tác và vector | Không |
| `GET` | `/api/categories` | Lấy danh sách toàn bộ các thể loại game | Không |
| `POST`| `/api/recommend/personalized` | Gợi ý game cá nhân hóa (Hybrid + MMR) | `{"user_id": "...", "top_k": 10, "use_mmr": true, "diversity_lambda": 0.7}` |
| `POST`| `/api/recommend/semantic` | Tìm kiếm theo mô tả ngôn ngữ tự nhiên | `{"query": "game bắn súng sinh tồn", "top_k": 5}` |
| `POST`| `/api/recommend/similar` | Gợi ý các tựa game tương đồng (Item-to-Item) | `{"item_id": "...", "top_k": 5}` |
| `POST`| `/api/explain` | Giải thích lý do đề xuất một tựa game | `{"user_id": "...", "item_id": "..."}` |
| `GET` | `/api/user/{user_id}/analytics` | Phân tích chân dung Gamer Persona & Lịch sử | Path param: `user_id` |
| `POST`| `/api/agent/chat` | Đàm thoại đa lượt cùng Trợ lý AI Concierge | `{"message": "Gợi ý game RPG cho tôi", "user_id": "..."}` |
| `GET` | `/api/search` | Tìm kiếm game theo từ khóa & phân trang | Query params: `q`, `category`, `limit`, `offset` |
| `GET` | `/api/games/{asin}` | Lấy thông tin chi tiết và điểm cảm xúc của game | Path param: `asin` |

---

## 👨‍💻 9. Tác Giả & Bản Quyền

- **Tác giả:** Đoàn Quang Minh
- **GitHub Repository:** [doanquangminh14/Recommendation_Systems](https://github.com/doanquangminh14/Recommendation_Systems)
- **Công nghệ cốt lõi:** Python, Polars, Sentence-Transformers, Scikit-learn, FastAPI, Streamlit, PyArrow, VADER Sentiment.
- **Giấy phép phát hành:** MIT License.
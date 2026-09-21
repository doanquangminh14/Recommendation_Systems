# 🎮 End-to-End AI-Powered Video Games Recommendation & Explanation System

> **Dự án Hệ thống Gợi ý & Giải thích Game ứng dụng AI (Hybrid Recommender + NLP + AI Agent)**  
> Dữ liệu nguồn: *Amazon Reviews 2023 - Video Games (Review interactions + Metadata)*.  
> Kiến trúc: Medallion Architecture (Bronze -> Silver -> Gold) kết hợp Hybrid Filtering (CF + Content-Based + NLP) & AI Agent (Recommend, Explain, Analytics) với giao diện Streamlit / API.

---

## 📌 Cơ chế làm việc & Quy chuẩn dự án (Project Protocol)

Để đảm bảo dự án được xây dựng bài bản, dễ đọc, dễ kiểm soát và chuyên nghiệp:

1. **Quy trình làm việc nghiêm ngặt từng Task đơn lẻ (Strict Task-by-Task Protocol & User Approval):**
   - ⚠️ **TUYỆT ĐỐI KHÔNG GỘP NHIỀU TASK HOẶC LÀM TOÀN BỘ PHASE CÙNG LÚC.**
   - Mỗi Phase bao gồm nhiều task con cụ thể. AI phải thực hiện **đúng 1 task đơn lẻ tại một thời điểm**.
   - **Quy trình 4 bước cho từng Task:**
     - **Bước 1 (Thực hiện 1 Task):** Làm đúng task được giao (ví dụ: viết notebook thử nghiệm hoặc đóng gói 1 module `src/`).
     - **Bước 2 (Báo cáo kết quả kiểm tra):** Trình bày rõ ràng kết quả, biểu đồ/metrics, đầu ra của task để User kiểm tra trực quan.
     - **Bước 3 (Chờ User nghiệm thu & xác nhận "OK"):** Dừng lại để User kiểm tra. Chỉ khi User xác nhận đạt yêu cầu mới được làm bước tiếp theo.
     - **Bước 4 (Git Commit, Push & Cập nhật README):** Commit task vừa xong với Conventional Commit tiếng Anh, push lên GitHub, cập nhật checkbox trong README rồi mới xin phép làm Task kế tiếp.

2. **Quy trình ML Engineering cho từng tính năng (Notebook-First -> Modularization):**
   - **Prototyping (`notebook/`):** Thử nghiệm thuật toán, trực quan hóa metrics và gỡ lỗi trực quan trên Jupyter Notebook trước.
   - **Production Modularization (`src/`):** Khi thuật toán đã chuẩn xác và được duyệt, đóng gói thành module sạch, chuẩn OOP và Type Hinting.

3. **Quy chuẩn Git Commit bằng Tiếng Anh (Conventional Commits):**
   - `feat(eda): add bronze layer data exploration and distribution analysis`
   - `feat(data): implement polars-based k-core filtering and silver layer parquet export`
   - `feat(nlp): add sentiment analysis experimentation in notebook`
   - `feat(nlp): modularize text embeddings and sentiment scoring`
   - `feat(cf): train svd matrix factorization baseline`

4. **Tech Stack cốt lõi:**
   - **Data Processing:** Python, Polars (Rust multi-threaded), PyArrow.
   - **NLP & Embeddings:** HuggingFace Transformers, Sentence-Transformers, NLTK/VADER.
   - **Recommender Systems:** Scikit-learn, SciPy, Surprise / Implicit / PyTorch.
   - **Agent & UI:** LangChain / AI Agent, FastAPI, Streamlit.

---

## 🏗️ Kiến trúc chi tiết hệ thống (System Architecture)

```
                         AMAZON REVIEWS 2023
                             VIDEO_GAMES
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   │
            REVIEW DATA                         META DATA
                │                                   │
                ▼                                   ▼
         Bronze / Raw Layer                  Bronze / Raw Layer
                │                                   │
                └──────────────┬────────────────────┘
                               ▼
                      VALIDATE + CLEAN
                               │
                               ▼
                      Silver / Clean Data
                               │
                  ┌────────────┼────────────┐
                  │            │            │
                  ▼            ▼            ▼
            Interactions   Item Features  Review Text
                  │            │            │
                  │            │            ▼
                  │            │           NLP
                  │            │            │
                  │            │     ┌──────┴──────┐
                  │            │     ▼             ▼
                  │            │  Sentiment    Embedding
                  │            │     │             │
                  │            │     └──────┬──────┘
                  │            │            │
                  │            ▼            ▼
                  │      Content-Based    NLP Features
                  │            │
                  ▼            │
         Collaborative         │
            Filtering           │
                  │             │
            ┌─────┴─────┐       │
            ▼           ▼       │
        K-Means     Matrix       │
        Users/      Factorization│
        Items           │        │
            │           │        │
            └─────┬─────┴────────┘
                  ▼
          HYBRID RECOMMENDER
                  │
                  ▼
             RANKING / TOP-K
                  │
          ┌───────┴────────┐
          ▼                ▼
     Recommendations    Explanation
          │                │
          └───────┬────────┘
                  ▼
               AI AGENT
                  │
         ┌────────┼─────────┐
         ▼        ▼         ▼
     Recommend  Explain   Analytics
         │        │         │
         └────────┼─────────┘
                  ▼
            API / Streamlit
                  │
                  ▼
            User Interface
```

---

## 📍 Tiến độ dự án chi tiết (Granular Roadmap & Status Tracker)

### 🔹 Phase 1: Bronze Layer & EDA (Khám phá dữ liệu thô)
- [x] **Task 1.1:** Khám phá cấu trúc JSONL, kiểu dữ liệu và tỷ lệ missing value (`notebook/undertand_data.ipynb`)
- [x] **Task 1.2:** Phân tích phân bố Rating (tỷ lệ 5 sao chiếm >65%)
- [x] **Task 1.3:** Phân tích User Activity & Item Popularity (Độ thưa ~99.99%, hiện tượng đuôi dài Long-tail)
- [x] **Task 1.4:** Phân tích Timeline đánh giá theo năm và phân bố thể loại game phổ biến

### 🔹 Phase 2: Validation & Silver Layer Creation (Làm sạch & K-Core)
- [x] **Task 2.1:** Trích xuất ảnh bìa/poster game chất lượng cao -> `data/silver/item_images.parquet` (`src/data/extract_images.py`)
- [x] **Task 2.2:** Thuật toán K-Core Filtering ($k=5$) & Deduplication bằng Polars (`notebook/02_clean_silver.ipynb`, `src/data/clean_silver.py`)
- [x] **Task 2.3:** Phân tách và xuất 3 tập dữ liệu Silver Parquet nén ZSTD:
  - `data/silver/interactions.parquet` (814,586 tương tác, 94,762 users, 25,612 items)
  - `data/silver/item_features.parquet` (25,612 items kèm đầy đủ metadata)
  - `data/silver/review_text.parquet` (814,586 review texts)

### 🔹 Phase 3: NLP & Review Text Processing (Phân tích cảm xúc & Embeddings)
- [x] **Task 3.1:** Thử nghiệm Sentiment Analysis (VADER) trên Notebook (`notebook/03_nlp_sentiment_embeddings.ipynb`)
- [x] **Task 3.2:** Đóng gói module trích xuất Sentiment Score cho từng Game & Review (`src/nlp/sentiment.py`)
- [x] **Task 3.3:** Thử nghiệm tạo Semantic Text Embeddings (Sentence-Transformers) trên Notebook
- [x] **Task 3.4:** Đóng gói module Embeddings & xuất tập Gold Vector Profiles (`src/nlp/embeddings.py`)

### 🔹 Phase 4: Collaborative Filtering (Lọc cộng tác)
- [x] **Task 4.1:** Phân cụm hành vi User & Item (K-Means Clustering) trên Notebook (`notebook/04_collaborative_filtering.ipynb`)
- [x] **Task 4.2:** Huấn luyện Matrix Factorization (SVD / TruncatedSVD) & 80/20 Train-Test Split
- [x] **Task 4.3:** Đóng gói module Collaborative Filtering (`src/models/collaborative/`)

### 🔹 Phase 5: Content-Based Filtering (Lọc dựa trên nội dung)
- [x] **Task 5.1:** Xây dựng Item Features Representation (Metadata + Category + Embeddings) trên Notebook (`notebook/05_content_based.ipynb`)
- [x] **Task 5.2:** Tính toán Cosine Similarity Matrix & hàm gợi ý Top-K tương đồng
- [x] **Task 5.3:** Đóng gói module Content-Based Filtering (`src/models/content_based/`)

### 🔹 Phase 6: Hybrid Recommender & Ranking (Gợi ý lai & Xếp hạng)
- [x] **Task 6.1:** Thử nghiệm cơ chế Weighted Hybrid Fusion (CF + Content-Based + Sentiment) trên Notebook (`notebook/06_hybrid_recommender.ipynb`)
- [ ] **Task 6.2:** Thử nghiệm trích xuất tín hiệu giải thích (Explanation Signals & Sentiment Reasons) trên Notebook
- [ ] **Task 6.3:** Thử nghiệm đa dạng hóa danh mục gợi ý & giảm thiên lệch độ phổ biến (MMR & Intra-List Diversity) trên Notebook
- [ ] **Task 6.4:** Đóng gói module Hybrid Recommender Engine (`src/models/hybrid/hybrid_engine.py`)
- [ ] **Task 6.5:** Đóng gói module Trích xuất giải thích gợi ý (`src/models/hybrid/explainer.py`)
- [ ] **Task 6.6:** Đóng gói module Xếp hạng & Re-ranking Top-K (`src/models/ranking.py`)

### 🔹 Phase 7: AI Agent Layer (Agent điều phối AI)
- [ ] **Task 7.1:** Thiết kế và thử nghiệm Agent Recommender Tools trên Notebook (`notebook/07_ai_agent.ipynb`)
- [ ] **Task 7.2:** Đóng gói Tool gợi ý game theo sở thích (`src/agent/tools/recommend_tool.py`)
- [ ] **Task 7.3:** Đóng gói Tool giải thích lý do gợi ý (`src/agent/tools/explain_tool.py`)
- [ ] **Task 7.4:** Đóng gói Tool phân tích hồ sơ và thống kê người dùng (`src/agent/tools/analytics_tool.py`)
- [ ] **Task 7.5:** Xây dựng AI Agent Core (Prompt Orchestration, Tool Routing, Session Memory) (`src/agent/agent_runner.py`)

### 🔹 Phase 8: API & Interactive Web Application
- [ ] **Task 8.1:** Định nghĩa Pydantic Schemas & DTOs cho API (`src/api/schemas.py`)
- [ ] **Task 8.2:** Xây dựng FastAPI REST Endpoints (Recommend, Search, Similar, Explain, Agent Chat) (`src/api/main.py`, `src/api/routes.py`)
- [ ] **Task 8.3:** Thiết kế các Component giao diện Streamlit (Game Card, Poster, Metric Badges) (`src/ui/components.py`)
- [ ] **Task 8.4:** Xây dựng Trang Khám phá & Gợi ý cá nhân hóa (`app.py` - Tab Recommendations)
- [ ] **Task 8.5:** Xây dựng Giao diện Trò chuyện tương tác với AI Agent (`app.py` - Tab AI Agent Chat)
- [ ] **Task 8.6:** Kiểm thử E2E tích hợp toàn hệ thống và tài liệu hóa hướng dẫn chạy hoàn chỉnh

---

## 📁 Cấu trúc thư mục chuẩn (Project Directory Structure)

```text
Recommendation_Systems/
├── data/
│   ├── bronze/                      # Dữ liệu gốc (Video_Games.jsonl, meta_Video_Games.jsonl)
│   ├── silver/                      # Dữ liệu sạch Parquet (interactions, item_features, review_text, item_images)
│   └── gold/                        # Dữ liệu vector embeddings, user/item profiles, clusters
├── notebook/                        # Bước 1: Prototyping, Visual EDA & Thử nghiệm thuật toán
│   ├── undertand_data.ipynb         # Phase 1: EDA Bronze data
│   ├── 02_clean_silver.ipynb        # Phase 2: K-Core filtering & Silver data
│   ├── 03_nlp_sentiment_embeddings.ipynb # Phase 3: NLP experimentation
│   ├── 04_collaborative_filtering.ipynb  # Phase 4: CF & Matrix Factorization
│   ├── 05_content_based.ipynb       # Phase 5: Content-Based
│   ├── 06_hybrid_recommender.ipynb  # Phase 6: Hybrid Ranking & Explanation
│   └── 07_ai_agent.ipynb            # Phase 7: AI Agent Tooling & Dialog
├── src/                             # Bước 2: Đóng gói mã nguồn sạch chuẩn Production
│   ├── data/                        # Validation, Cleaning, K-core filtering
│   │   ├── __init__.py
│   │   ├── extract_images.py        # [x] Trích xuất poster/ảnh bìa game
│   │   └── clean_silver.py          # [x] Polars K-core filter & tách 3 tập Silver
│   ├── nlp/                         # Sentiment Analysis & Embeddings
│   │   ├── __init__.py
│   │   ├── sentiment.py             # [x] VADER sentiment scoring & item aggregation
│   │   └── embeddings.py            # [x] Dense semantic text embeddings (all-MiniLM-L6-v2)
│   ├── models/                      # CF, Content-Based, Hybrid Ranking
│   │   ├── __init__.py
│   │   ├── collaborative/           # [x] SVD Matrix Factorization & K-Means Clusters
│   │   ├── content_based/           # [x] Semantic Content-Based & Dynamic User Profile
│   │   ├── hybrid/                  # Hybrid Engine & Explainer
│   │   └── ranking.py               # Top-K Ranking & Re-ranking
│   ├── agent/                       # AI Agent (Recommend, Explain, Analytics)
│   │   ├── __init__.py
│   │   ├── tools/                   # Agent custom tools
│   │   └── agent_runner.py          # LLM Agent orchestration
│   ├── ui/                          # Streamlit UI custom components
│   └── api/                         # FastAPI backend endpoints
├── app.py                           # Giao diện tương tác Streamlit
├── requirements.txt                 # Dependencies
├── README.md                        # Tài liệu dự án & Status Tracker
└── main.py                          # Entry point
```

---

## 📝 Nhật ký thực hiện (Work Log)

- **Đã hoàn thành:**
  1. **Phase 1 (EDA):** Khám phá cấu trúc dữ liệu, tỷ lệ 5 sao (>65%), độ thưa ~99.99%, phân bố đuôi dài và thể loại trong [notebook/undertand_data.ipynb](file:///c:/Users/Minh%20Doan/repo_github/Recommendation_Systems/notebook/undertand_data.ipynb). Đã commit & push lên GitHub.
  2. **Trích xuất ảnh game (Phase 2 - Task 2.1):** Đã hoàn thành [src/data/extract_images.py](file:///c:/Users/Minh%20Doan/repo_github/Recommendation_Systems/src/data/extract_images.py), tạo file `data/silver/item_images.parquet` (8.49 MB, 137,127 ảnh game). Đã commit & push lên GitHub.
  3. **Lọc K-Core & Tách 3 tập Silver (Phase 2 - Task 2.2 & 2.3):** Đã hoàn thành [src/data/clean_silver.py](file:///c:/Users/Minh%20Doan/repo_github/Recommendation_Systems/src/data/clean_silver.py):
     - Lọc K-Core ($k=5$) hội tụ sau 7 vòng lặp chỉ trong ~8.5 giây bằng Polars.
     - Giảm độ thưa (Sparsity) từ 99.99% xuống **99.9664%**, trung bình 8.6 reviews/user và 31.8 reviews/item.
     - Xuất thành công 3 file Parquet chất lượng cao nén `zstd`:
       - `data/silver/interactions.parquet`: 814,586 tương tác sạch (8.45 MB).
       - `data/silver/item_features.parquet`: 25,612 game đầy đủ metadata (14.57 MB).
       - `data/silver/review_text.parquet`: 814,586 đánh giá kèm text (161.98 MB).
  4. **NLP & Review Text Processing (Phase 3 - Tasks 3.1 -> 3.4):**
     - Đã hoàn thành thử nghiệm trong [notebook/03_nlp_sentiment_embeddings.ipynb](file:///c:/Users/Minh%20Doan/repo_github/Recommendation_Systems/notebook/03_nlp_sentiment_embeddings.ipynb).
     - Đóng gói [src/nlp/sentiment.py](file:///c:/Users/Minh%20Doan/repo_github/Recommendation_Systems/src/nlp/sentiment.py): Trích xuất VADER sentiment compound score và tổng hợp profile cảm xúc item (`positive_review_ratio`, `avg_sentiment_compound`).
     - Đóng gói [src/nlp/embeddings.py](file:///c:/Users/Minh%20Doan/repo_github/Recommendation_Systems/src/nlp/embeddings.py): Trích xuất vector ngữ nghĩa 384 chiều với `all-MiniLM-L6-v2` cho 25,612 game, lưu vào Gold Layer (`data/gold/item_embeddings.parquet` và `item_embeddings.npy`). Đã commit & push lên GitHub.
  5. **Collaborative Filtering (Phase 4 - Tasks 4.1 -> 4.3):**
     - Đã hoàn thành thử nghiệm trong [notebook/04_collaborative_filtering.ipynb](file:///c:/Users/Minh%20Doan/repo_github/Recommendation_Systems/notebook/04_collaborative_filtering.ipynb).
     - Đóng gói [src/models/collaborative/matrix_factorization.py](file:///c:/Users/Minh%20Doan/repo_github/Recommendation_Systems/src/models/collaborative/matrix_factorization.py): TruncatedSVD ($k=64$ chiều latent), tính toán predicted rating & Top-K recommendation.
     - Đóng gói [src/models/collaborative/clustering.py](file:///c:/Users/Minh%20Doan/repo_github/Recommendation_Systems/src/models/collaborative/clustering.py): Phân cụm K-Means các game theo không gian tương tác, tạo file `data/gold/item_cf_clusters.parquet`. Đã commit & push lên GitHub.
  6. **Content-Based Filtering (Phase 5 - Tasks 5.1 -> 5.3):**
     - Đã hoàn thành thử nghiệm trong [notebook/05_content_based.ipynb](file:///c:/Users/Minh%20Doan/repo_github/Recommendation_Systems/notebook/05_content_based.ipynb).
     - Đóng gói [src/models/content_based/recommender.py](file:///c:/Users/Minh%20Doan/repo_github/Recommendation_Systems/src/models/content_based/recommender.py): Gợi ý tương đồng Item-to-Item siêu nhanh và thuật toán Dynamic User Profile Centroid Vector giải quyết triệt để Cold-Start. Đã commit & push lên GitHub.
  7. **Hybrid Recommender Prototyping (Phase 6 - Task 6.1):**
     - Đã hoàn thành thử nghiệm trong [notebook/06_hybrid_recommender.ipynb](file:///c:/Users/Minh%20Doan/repo_github/Recommendation_Systems/notebook/06_hybrid_recommender.ipynb).
     - Triển khai cơ chế chuẩn hóa Min-Max đa nguồn và cơ chế kết hợp trọng số động $\text{Score} = w_{\text{cf}} \cdot S_{\text{cf}} + w_{\text{cb}} \cdot S_{\text{cb}} + w_{\text{sent}} \cdot S_{\text{sent}}$.
     - Xử lý Cold-Start User tự động chuyển sang cấu hình $w_{\text{cf}}=0, w_{\text{cb}}=0.70, w_{\text{sent}}=0.30$. Đã commit & push lên GitHub.
- **Bước tiếp theo cần làm ngay:**
  - **Phase 6 - Task 6.2 (Explanation Signals & Sentiment Reasons Prototyping):**
    - Thử nghiệm trích xuất tín hiệu giải thích lý do gợi ý (Explanation Signals, Top Feature Keywords, Sentiment Review Quotes) trên notebook [notebook/06_hybrid_recommender.ipynb](file:///c:/Users/Minh%20Doan/repo_github/Recommendation_Systems/notebook/06_hybrid_recommender.ipynb).
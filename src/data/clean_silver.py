"""
Module: clean_silver.py
Mục đích:
1. Đọc dữ liệu Bronze (Video_Games.jsonl và meta_Video_Games.jsonl) bằng Polars hiệu năng cao.
2. Làm sạch dữ liệu: loại bỏ null, deduplicate (giữ đánh giá mới nhất theo timestamp).
3. Áp dụng thuật toán lọc K-Core (k=5) đa vòng lặp để loại bỏ cold-start users & items ít tương tác.
4. Tách và xuất 3 tập dữ liệu chuẩn định dạng Parquet (nén zstd) vào data/silver/:
   - interactions.parquet : [user_id, parent_asin, rating, timestamp]
   - item_features.parquet: [parent_asin, title, main_category, categories, store, average_rating, rating_number, price, features, description]
   - review_text.parquet  : [user_id, parent_asin, rating, title, text, timestamp]
"""

import os
import sys
import time
import polars as pl

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def load_and_preprocess_reviews(review_path: str) -> pl.DataFrame:
    """
    Đọc dữ liệu reviews bằng Polars LazyFrame, lọc null và deduplicate theo timestamp mới nhất.
    """
    print(f"[1/4] Đang đọc và tiền xử lý dữ liệu Reviews từ: {review_path}...")
    start_time = time.time()

    # Quét dữ liệu bằng Polars LazyFrame với schema xác định
    schema_overrides = {
        "rating": pl.Float64,
        "title": pl.String,
        "text": pl.String,
        "asin": pl.String,
        "parent_asin": pl.String,
        "user_id": pl.String,
        "timestamp": pl.Int64,
    }

    lazy_df = pl.scan_ndjson(review_path, schema_overrides=schema_overrides)

    # Chọn các cột cần thiết và lọc null
    df_reviews = (
        lazy_df.select([
            pl.col("user_id"),
            pl.col("parent_asin"),
            pl.col("rating"),
            pl.col("timestamp"),
            pl.col("title"),
            pl.col("text"),
        ])
        .filter(
            pl.col("user_id").is_not_null()
            & pl.col("parent_asin").is_not_null()
            & pl.col("rating").is_not_null()
            & (pl.col("user_id") != "")
            & (pl.col("parent_asin") != "")
        )
        .collect()
    )

    initial_count = len(df_reviews)
    print(f"  -> Tổng số bản ghi hợp lệ ban đầu: {initial_count:,} ({time.time() - start_time:.2f}s)")

    # Deduplicate: nếu 1 user đánh giá cùng 1 parent_asin nhiều lần, giữ đánh giá mới nhất theo timestamp
    print("  -> Đang deduplicate (giữ tương tác mới nhất theo timestamp)...")
    df_reviews = (
        df_reviews.sort(by=["user_id", "parent_asin", "timestamp"])
        .unique(subset=["user_id", "parent_asin"], keep="last")
    )

    dedup_count = len(df_reviews)
    print(f"  -> Số tương tác sau khi deduplicate: {dedup_count:,} (Loại bỏ {initial_count - dedup_count:,} bản ghi trùng)")

    return df_reviews


def apply_k_core_filter(df: pl.DataFrame, k: int = 5) -> pl.DataFrame:
    """
    Áp dụng thuật toán K-Core Filtering lặp cho đến khi hội tụ:
    Mỗi user phải có >= k tương tác VÀ mỗi item phải có >= k tương tác.
    """
    print(f"\n[2/4] Bắt đầu K-Core Filtering với k={k}...")
    start_time = time.time()

    iteration = 0
    current_df = df

    while True:
        iteration += 1
        num_records_before = len(current_df)

        # 1. Lọc users có >= k tương tác
        user_counts = current_df.group_by("user_id").len(name="user_count")
        valid_users = user_counts.filter(pl.col("user_count") >= k).select("user_id")
        current_df = current_df.join(valid_users, on="user_id", how="inner")

        # 2. Lọc items có >= k tương tác
        item_counts = current_df.group_by("parent_asin").len(name="item_count")
        valid_items = item_counts.filter(pl.col("item_count") >= k).select("parent_asin")
        current_df = current_df.join(valid_items, on="parent_asin", how="inner")

        num_records_after = len(current_df)
        num_dropped = num_records_before - num_records_after
        unique_users = current_df.select(pl.col("user_id").n_unique()).item()
        unique_items = current_df.select(pl.col("parent_asin").n_unique()).item()

        print(
            f"  -> Vòng lặp {iteration}: Còn {num_records_after:,} tương tác "
            f"({unique_users:,} users, {unique_items:,} items) - Đã loại {num_dropped:,} tương tác."
        )

        if num_dropped == 0:
            print(f"  -> K-Core Filtering hội tụ hoàn toàn sau {iteration} vòng lặp ({time.time() - start_time:.2f}s)!")
            break

    return current_df


def extract_and_export_silver(
    df_clean_reviews: pl.DataFrame,
    meta_path: str = "data/meta_Video_Games.jsonl",
    output_dir: str = "data/silver",
) -> None:
    """
    Trích xuất và xuất 3 tập dữ liệu Silver:
    1. interactions.parquet
    2. item_features.parquet
    3. review_text.parquet
    """
    print(f"\n[3/4] Đang tách và xuất các tập dữ liệu Silver sang: {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    # 1. Xuất interactions.parquet [user_id, parent_asin, rating, timestamp]
    interactions_path = os.path.join(output_dir, "interactions.parquet")
    df_interactions = df_clean_reviews.select([
        pl.col("user_id"),
        pl.col("parent_asin"),
        pl.col("rating"),
        pl.col("timestamp"),
    ]).sort(by=["user_id", "timestamp"])

    df_interactions.write_parquet(interactions_path, compression="zstd")
    interactions_size = os.path.getsize(interactions_path) / (1024**2)
    print(f"  -> [ĐÃ XUẤT] {interactions_path} ({len(df_interactions):,} dòng, {interactions_size:.2f} MB)")

    # 2. Xuất review_text.parquet [user_id, parent_asin, rating, title, text, timestamp]
    review_text_path = os.path.join(output_dir, "review_text.parquet")
    df_review_text = df_clean_reviews.select([
        pl.col("user_id"),
        pl.col("parent_asin"),
        pl.col("rating"),
        pl.col("title"),
        pl.col("text"),
        pl.col("timestamp"),
    ]).sort(by=["user_id", "timestamp"])

    df_review_text.write_parquet(review_text_path, compression="zstd")
    review_text_size = os.path.getsize(review_text_path) / (1024**2)
    print(f"  -> [ĐÃ XUẤT] {review_text_path} ({len(df_review_text):,} dòng, {review_text_size:.2f} MB)")

    # 3. Đọc và lọc metadata cho item_features.parquet
    print(f"\n[4/4] Đang xử lý metadata cho tập item_features từ: {meta_path}...")
    active_items = df_clean_reviews.select("parent_asin").unique()

    meta_schema_overrides = {
        "price": pl.String,
        "average_rating": pl.Float64,
        "rating_number": pl.Int64,
    }

    lazy_meta = pl.scan_ndjson(meta_path, schema_overrides=meta_schema_overrides)
    df_meta = (
        lazy_meta.select([
            pl.col("parent_asin"),
            pl.col("title"),
            pl.col("main_category"),
            pl.col("categories"),
            pl.col("store"),
            pl.col("average_rating"),
            pl.col("rating_number"),
            pl.col("price").str.replace_all(r"[^\d.]", "").cast(pl.Float64, strict=False).alias("price"),
            pl.col("features"),
            pl.col("description"),
        ])
        .filter(pl.col("parent_asin").is_not_null() & (pl.col("parent_asin") != ""))
        .collect()
    )

    # Deduplicate metadata theo parent_asin (giữ bản ghi đầu tiên)
    df_meta = df_meta.unique(subset=["parent_asin"], keep="first")

    # Chỉ giữ các item có trong tập tương tác sau khi lọc K-core
    df_item_features = active_items.join(df_meta, on="parent_asin", how="left")

    item_features_path = os.path.join(output_dir, "item_features.parquet")
    df_item_features.write_parquet(item_features_path, compression="zstd")
    item_features_size = os.path.getsize(item_features_path) / (1024**2)
    print(f"  -> [ĐÃ XUẤT] {item_features_path} ({len(df_item_features):,} items, {item_features_size:.2f} MB)")

    # Tổng kết
    total_users = df_interactions.select(pl.col("user_id").n_unique()).item()
    total_items = df_interactions.select(pl.col("parent_asin").n_unique()).item()
    total_interactions = len(df_interactions)
    sparsity = (1.0 - (total_interactions / (total_users * total_items))) * 100

    print("\n" + "=" * 65)
    print("--- TỔNG KẾT TẬP DỮ LIỆU SILVER (SAU K-CORE k=5) ---")
    print(f"• Số lượng người dùng (Users) : {total_users:,}")
    print(f"• Số lượng sản phẩm (Items)   : {total_items:,}")
    print(f"• Số lượng tương tác (Reviews): {total_interactions:,}")
    print(f"• Độ thưa ma trận (Sparsity)  : {sparsity:.4f}%")
    print(f"• Trung bình review / user    : {total_interactions / total_users:.2f}")
    print(f"• Trung bình review / item    : {total_interactions / total_items:.2f}")
    print(f"• Tổng thời gian xuất Silver  : {time.time() - start_time:.2f}s")
    print("=" * 65 + "\n")


def main(
    review_path: str = "data/Video_Games.jsonl",
    meta_path: str = "data/meta_Video_Games.jsonl",
    output_dir: str = "data/silver",
    k: int = 5,
) -> None:
    """
    Hàm thực thi toàn bộ pipeline tạo tầng Silver.
    """
    print("=================================================================")
    print("🚀 BẮT ĐẦU PIPELINE TẠO TẦNG SILVER (VALIDATION & K-CORE FILTER)")
    print("=================================================================\n")

    overall_start = time.time()

    # 1. Đọc và làm sạch raw reviews
    df_reviews = load_and_preprocess_reviews(review_path)

    # 2. Lọc K-Core
    df_clean_reviews = apply_k_core_filter(df_reviews, k=k)

    # 3. Xuất 3 tập Parquet
    extract_and_export_silver(
        df_clean_reviews=df_clean_reviews,
        meta_path=meta_path,
        output_dir=output_dir,
    )

    print(f"🎉 HOÀN THÀNH TOÀN BỘ PIPELINE TRONG {time.time() - overall_start:.2f} GIÂY!\n")


if __name__ == "__main__":
    main()

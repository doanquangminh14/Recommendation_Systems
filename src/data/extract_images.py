"""
Module: extract_images.py
Mục đích: Trích xuất và chuẩn hóa URL hình ảnh (Poster, Thumbnail) của các Game từ file Metadata
và lưu vào tầng Silver (data/silver/item_images.parquet) để phục vụ hiển thị UI và Recommender.
"""

import os
import sys
import json
import polars as pl

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x




def extract_game_images(
    meta_path: str = "data/meta_Video_Games.jsonl",
    output_path: str = "data/silver/item_images.parquet",
) -> pl.DataFrame:
    """
    Đọc file meta_Video_Games.jsonl và trích xuất đường link ảnh bìa/poster chất lượng cao.
    """
    print(f"Bắt đầu trích xuất hình ảnh từ: {meta_path}")

    # Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = []

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Đang đọc metadata"):
            data = json.loads(line)
            parent_asin = data.get("parent_asin")
            title = data.get("title", "")
            images = data.get("images", [])

            main_image_url = None
            thumb_image_url = None
            hi_res_url = None

            if images and isinstance(images, list) and len(images) > 0:
                first_img = images[0]
                if isinstance(first_img, dict):
                    main_image_url = first_img.get("large") or first_img.get("thumb")
                    thumb_image_url = first_img.get("thumb")
                    hi_res_url = first_img.get("hi_res")

            records.append(
                {
                    "parent_asin": parent_asin,
                    "title": title,
                    "main_image_url": main_image_url,
                    "thumb_image_url": thumb_image_url,
                    "hi_res_url": hi_res_url,
                    "has_image": main_image_url is not None,
                }
            )

    # Chuyển đổi sang Polars DataFrame để tối ưu hóa hiệu năng
    df_images = pl.DataFrame(records)

    # Lưu sang định dạng Parquet siêu nhẹ và tối ưu nén
    df_images.write_parquet(output_path, compression="zstd")

    total_items = len(df_images)
    items_with_images = df_images.filter(pl.col("has_image")).height

    print("\n" + "=" * 50)
    print("--- KẾT QUẢ TRÍCH XUẤT HÌNH ẢNH SẢN PHẨM ---")
    print(f"Tổng số game: {total_items:,}")
    print(f"Số game có ảnh hợp lệ: {items_with_images:,} ({items_with_images / total_items * 100:.2f}%)")
    print(f"Đã lưu thành công vào: {output_path} ({os.path.getsize(output_path) / 1024**2:.2f} MB)")
    print("=" * 50 + "\n")

    return df_images


if __name__ == "__main__":
    extract_game_images()

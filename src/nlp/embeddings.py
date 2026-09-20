"""
Semantic Text Embeddings Module for Video Games.

Encodes game metadata (title, category, store, description, features) into
dense semantic vector representations using Sentence-Transformers (all-MiniLM-L6-v2)
for Content-Based Filtering and AI Agent context retrieval.
"""

from typing import Optional, List, Union
import os
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class ItemTextEmbedder:
    """
    Generates dense semantic embeddings for game items from their textual metadata.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Initialize the sentence transformer model.

        Parameters
        ----------
        model_name : str
            Name of the HuggingFace sentence-transformers model.
        device : str or None
            'cuda', 'cpu', or auto-detected.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def build_item_text_repr(self, row: dict) -> str:
        """
        Build a concise, rich textual representation for a single game.
        """
        title = row.get("title") or ""
        main_cat = row.get("main_category") or ""
        categories = row.get("categories") or []
        if isinstance(categories, list):
            cat_str = ", ".join([str(c) for c in categories if c])
        else:
            cat_str = str(categories) if categories else ""

        store = row.get("store") or ""
        features = row.get("features") or []
        feat_str = " ".join([str(f) for f in features if f]) if isinstance(features, list) else str(features)
        
        description = row.get("description") or []
        desc_str = " ".join([str(d) for d in description if d]) if isinstance(description, list) else str(description)

        # Truncate long descriptions to retain key signals without blowing token limits
        parts = []
        if title:
            parts.append(f"Game Title: {title}")
        if main_cat or cat_str:
            parts.append(f"Categories: {main_cat} | {cat_str}")
        if store:
            parts.append(f"Developer/Store: {store}")
        if feat_str:
            parts.append(f"Features: {feat_str[:300]}")
        if desc_str:
            parts.append(f"Description: {desc_str[:400]}")

        return " \n ".join(parts).strip()

    def generate_item_texts(self, df_items: pl.DataFrame) -> List[str]:
        """
        Convert a DataFrame of item features into a list of consolidated text descriptions.
        """
        dicts = df_items.to_dicts()
        return [self.build_item_text_repr(row) for row in dicts]

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 128,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encode a list of texts into dense vectors.

        Parameters
        ----------
        texts : list of str
            Input texts.
        batch_size : int
            Batch size for model inference.
        show_progress_bar : bool
            Whether to show tqdm progress bar.
        normalize_embeddings : bool
            Whether to L2-normalize vectors (ideal for Cosine Similarity / Dot Product).

        Returns
        -------
        np.ndarray
            Matrix of shape (N, embedding_dim), default (N, 384) for all-MiniLM-L6-v2.
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings
        )


def run_item_embeddings_pipeline(
    items_path: str = "data/silver/item_features.parquet",
    output_embeddings_path: str = "data/gold/item_embeddings.parquet",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 128
) -> None:
    """
    End-to-end execution of Item Semantic Text Embeddings extraction -> Gold Layer.
    """
    print(f"[*] Loading clean item metadata from {items_path}...")
    df_items = pl.read_parquet(items_path)
    print(f"    Loaded {len(df_items):,} unique games.")

    embedder = ItemTextEmbedder(model_name=model_name)
    print(f"[*] Building consolidated text representations for {len(df_items):,} games...")
    item_texts = embedder.generate_item_texts(df_items)

    print(f"[*] Encoding {len(item_texts):,} game descriptions into dense vectors (dim=384)...")
    embeddings = embedder.encode_texts(item_texts, batch_size=batch_size, normalize_embeddings=True)

    print(f"[*] Preparing Gold Layer item embeddings DataFrame...")
    os.makedirs(os.path.dirname(output_embeddings_path), exist_ok=True)
    
    # Store embedding array as list of floats in Polars for standard Parquet compatibility
    embedding_lists = embeddings.tolist()
    
    df_embeddings = pl.DataFrame({
        "parent_asin": df_items["parent_asin"],
        "title": df_items["title"],
        "text_representation": item_texts,
        "embedding": embedding_lists
    })

    print(f"[*] Saving Gold Layer embeddings to {output_embeddings_path}...")
    df_embeddings.write_parquet(output_embeddings_path, compression="zstd")
    
    # Also save raw numpy array for ultra-fast matrix multiplication during Cosine Similarity
    numpy_path = output_embeddings_path.replace(".parquet", ".npy")
    np.save(numpy_path, embeddings)
    print(f"[*] Saved raw numpy matrix to {numpy_path} (shape: {embeddings.shape})")

    print(f"[+] Item embeddings pipeline completed successfully!")


if __name__ == "__main__":
    run_item_embeddings_pipeline()

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.config import MODEL_NAME, TOP_K


class ProductSearcher:
    def __init__(self, products_df: pd.DataFrame):
        self.model = SentenceTransformer(MODEL_NAME)
        self.products_df = products_df.drop_duplicates(subset="id").reset_index(drop=True)
        self._build_index()

    def _build_index(self):
        texts = (
            self.products_df["title"].fillna("") + " " +
            self.products_df["category"].fillna("")
        ).tolist()
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

    def search(self, query: str, top_k: int = TOP_K) -> pd.DataFrame:
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)
        results = self.products_df.iloc[indices[0]].copy()
        results["score"] = scores[0]
        return results

    def filter(
        self,
        df: pd.DataFrame,
        min_price: float = None,
        max_price: float = None,
        brand: str = None,
        min_rate: float = None,
    ) -> pd.DataFrame:
        if min_price is not None:
            df = df[df["price"] >= min_price]
        if max_price is not None:
            df = df[df["price"] <= max_price]
        if brand is not None:
            df = df[df["brand"].str.contains(brand, case=False, na=False)]
        if min_rate is not None:
            df = df[df["rate"] >= min_rate]
        return df.sort_values("score", ascending=False)

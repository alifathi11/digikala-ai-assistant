import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from src.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    PROCESSED_COMMENTS_CSV,
    TOP_K,
)
from src.generator import generate_answer


class QAPipeline:
    def __init__(self, product_id: int):
        self.product_id = product_id
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.df, self.index = self._build_product_index()

    def _build_product_index(self) -> tuple[pd.DataFrame, faiss.Index]:
        df = pd.read_csv(PROCESSED_COMMENTS_CSV, low_memory=False)
        df = df[df["product_id"] == self.product_id].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"هیچ نظری برای product_id={self.product_id} یافت نشد.")

        texts = df["body"].tolist()
        embeddings = self.model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return df, index

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        vec = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self.df.iloc[idx]
            results.append({
                "score": float(score),
                "id": int(row["id"]),
                "body": row["body"],
                "rate": row.get("rate"),
                "recommendation_status": row.get("recommendation_status"),
            })
        return results

    def answer(self, query: str) -> str:
        contexts = self.retrieve(query)
        return generate_answer(query, contexts)

    def run(self, query: str, return_contexts: bool = False):
        contexts = self.retrieve(query)
        answer = generate_answer(query, contexts)
        if return_contexts:
            return answer, contexts
        return answer

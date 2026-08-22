import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME, TOP_K


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index = None
        self.metadata: list[dict] = []

    def build_index(self, comments: pd.DataFrame):
        self.metadata = comments[["id", "body", "rate", "recommendation_status"]].to_dict("records")
        texts = [str(r.get("body", "")) for r in self.metadata]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            row = self.metadata[idx].copy()
            row["score"] = float(score)
            results.append(row)
        return results

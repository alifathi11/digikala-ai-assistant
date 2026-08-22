import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from src.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
)


def build_index(texts: list[str], ids: list) -> tuple[faiss.Index, list]:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, ids


def save_index(index: faiss.Index, ids: list):
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_META_PATH, "wb") as f:
        pickle.dump(ids, f)


def load_index() -> tuple[faiss.Index, list]:
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(FAISS_META_PATH, "rb") as f:
        ids = pickle.load(f)
    return index, ids

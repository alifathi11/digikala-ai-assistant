from dataclasses import dataclass
from pathlib import Path

from .config import load_config
from .embedding.factory import (
    EmbeddingFactory,
)
from .preprocessing.processor import (
    TextProcessor,
)
from .retrieval.bm25 import (
    BM25Retriever,
)
from .retrieval.embedding import (
    EmbeddingRetriever,
)
from .retrieval.hybrid import (
    HybridRetriever,
)
from .vector_store.faiss import (
    FAISSVectorStore,
)


@dataclass
class RetrievalStack:
    processor: TextProcessor
    embedding_model: object
    vector_store: FAISSVectorStore
    bm25: BM25Retriever
    embedding: EmbeddingRetriever
    hybrid: HybridRetriever

    @property
    def documents(
        self,
    ):
        return self.vector_store.documents


def load_retrieval_stack(
    project_root,
    rag_config=None,
    faiss_path=None,
    bm25_path=None,
):
    project_root = Path(
        project_root
    )

    if rag_config is None:
        rag_config = load_config(
            project_root
            / "configs"
            / "rag.yaml"
        )

    if faiss_path is None:
        faiss_path = (
            project_root
            / "data"
            / "indexes"
            / "product_comments_embedding"
        )

    if bm25_path is None:
        bm25_path = (
            project_root
            / "data"
            / "indexes"
            / "product_comments_bm25_tantivy"
        )

    processor = TextProcessor()

    embedding_model = (
        EmbeddingFactory
        .create(
            provider=rag_config[
                "embedding"
            ]["provider"],
            model_name=rag_config[
                "embedding"
            ]["model"],
        )
    )

    vector_store = (
        FAISSVectorStore()
        .load(
            faiss_path
        )
    )

    embedding_retriever = (
        EmbeddingRetriever(
            embedding_model=(
                embedding_model
            ),
            vector_store=(
                vector_store
            ),
            processor=processor,
        )
    )

    bm25_retriever = (
        BM25Retriever(
            processor=processor
        )
    )

    bm25_retriever.load(
        bm25_path,
        documents=(
            vector_store.documents
        ),
    )

    hybrid_config = (
        rag_config[
            "retrieval"
        ]["hybrid"]
    )

    hybrid_retriever = (
        HybridRetriever(
            bm25_retriever=(
                bm25_retriever
            ),
            embedding_retriever=(
                embedding_retriever
            ),
            bm25_weight=(
                hybrid_config[
                    "bm25_weight"
                ]
            ),
            embedding_weight=(
                hybrid_config[
                    "embedding_weight"
                ]
            ),
        )
    )

    return RetrievalStack(
        processor=processor,
        embedding_model=(
            embedding_model
        ),
        vector_store=(
            vector_store
        ),
        bm25=bm25_retriever,
        embedding=embedding_retriever,
        hybrid=hybrid_retriever,
    )

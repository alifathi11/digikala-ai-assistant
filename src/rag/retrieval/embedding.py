import pandas as pd

from .base import BaseRetriever
from ..embedding.base import BaseEmbedding
from ..vector_store.faiss import FAISSVectorStore


class EmbeddingRetriever(BaseRetriever):

    def __init__(
        self,
        documents,
        embedding_model,
        processor=None,
        text_column="search_text"
    ):
        super().__init__(
            processor
        )

        self.documents = (
            documents
            .reset_index(drop=True)
            .copy()
        )

        self.embedding_model = embedding_model
        self.text_column = text_column

        self.vector_store = FAISSVectorStore()

        self._build_index()


    def _build_index(self):

        texts = (
            self.documents[self.text_column]
            .fillna("")
            .astype(str)
            .tolist()
        )

        embeddings = (
            self.embedding_model.encode(texts)
        )

        self.vector_store.build(
            embeddings,
            self.documents
        )


    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ):

        query = self.process_query(query)

        query_embedding = (
            self.embedding_model.encode(
                [query]
            )
        )

        return self.vector_store.search(
            query_embedding,
            top_k
        )
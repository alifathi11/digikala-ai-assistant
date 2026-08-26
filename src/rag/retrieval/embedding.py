from .base import BaseRetriever
from ..vector_store.faiss import FAISSVectorStore


class EmbeddingRetriever(BaseRetriever):

    def __init__(
        self,
        documents=None,
        embedding_model=None,
        processor=None,
        text_column="search_text",
        vector_store=None
    ):
        super().__init__(processor)

        if embedding_model is None:
            raise ValueError(
                "embedding_model is required"
            )

        self.embedding_model = embedding_model
        self.text_column = text_column

        if vector_store is not None:
            self.vector_store = vector_store
            self.documents = (
                vector_store.documents
                .reset_index(drop=True)
                .copy()
            )
            return

        if documents is None:
            raise ValueError(
                "documents or vector_store is required"
            )

        self.documents = (
            documents
            .reset_index(drop=True)
            .copy()
        )

        self.vector_store = FAISSVectorStore()
        self._build_index()


    def _build_index(self):
        texts = (
            self.documents[self.text_column]
            .fillna("")
            .astype(str)
            .tolist()
        )

        embeddings = self.embedding_model.encode(
            texts
        )

        self.vector_store.build(
            embeddings,
            self.documents
        )


    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_ids=None
    ):
        query = self.process_query(query)

        query_embedding = (
            self.embedding_model.encode(
                [query]
            )
        )

        return self.vector_store.search(
            query_embedding,
            top_k=top_k,
            candidate_ids=candidate_ids
        )

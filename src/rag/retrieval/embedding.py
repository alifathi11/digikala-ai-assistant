from .base import BaseRetriever


class EmbeddingRetriever(
    BaseRetriever
):
    def __init__(
        self,
        embedding_model,
        vector_store,
        processor=None,
    ):
        super().__init__(
            processor
        )

        if embedding_model is None:
            raise ValueError(
                "embedding_model is required"
            )

        if vector_store is None:
            raise ValueError(
                "vector_store is required"
            )

        if (
            vector_store.documents
            is None
        ):
            raise ValueError(
                "vector_store must be loaded "
                "before creating the retriever"
            )

        self.embedding_model = (
            embedding_model
        )

        self.vector_store = (
            vector_store
        )

        self.documents = (
            vector_store.documents
            .reset_index(drop=True)
        )


    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_ids=None,
    ):
        query = self.process_query(
            query
        )

        query_embedding = (
            self.embedding_model
            .encode(
                [query]
            )
        )

        return (
            self.vector_store
            .search(
                query_embedding,
                top_k=top_k,
                candidate_ids=(
                    candidate_ids
                ),
            )
        )

from .tfidf import TFIDFRetriever
from .bm25 import BM25Retriever
from .embedding import EmbeddingRetriever
from .hybrid import HybridRetriever

class RetrieverFactory:

    @staticmethod
    def create(
        retriever_type: str,
        documents,
        embedding_model=None,
        **kwargs
    ):

        retriever_type = retriever_type.lower()


        if retriever_type == "tfidf":

            return TFIDFRetriever(
                documents,
                **kwargs
            )


        elif retriever_type == "bm25":

            return BM25Retriever(
                documents,
                **kwargs
            )


        elif retriever_type == "embedding":

            if embedding_model is None:
                raise ValueError(
                    "Embedding model is required"
                )


            return EmbeddingRetriever(
                documents,
                embedding_model,
                **kwargs
            )

        elif retriever_type == "hybrid":

            return HybridRetriever(
                bm25_retriever=kwargs["bm25_retriever"],
                embedding_retriever=kwargs["embedding_retriever"],
                bm25_weight=kwargs.get(
                    "bm25_weight",
                    0.3
                ),
                embedding_weight=kwargs.get(
                    "embedding_weight",
                    0.7
                )
            )

        else:
            raise ValueError(
                f"Unknown retriever: {retriever_type}"
            )
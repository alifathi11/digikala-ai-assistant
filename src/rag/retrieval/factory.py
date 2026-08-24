from .tfidf import TFIDFRetriever
from .bm25 import BM25Retriever
from .embedding import EmbeddingRetriever

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


        else:
            raise ValueError(
                f"Unknown retriever: {retriever_type}"
            )
from sentence_transformers import SentenceTransformer

from .base import BaseEmbedding


class SentenceTransformerEmbedding(BaseEmbedding):

    def __init__(
        self,
        model_name: str
    ):

        self.model = SentenceTransformer(
            model_name
        )


    def encode(
        self,
        texts: list[str]
    ):

        return self.model.encode(
            texts,
            normalize_embeddings=True
        )
from .sentence_transformer import SentenceTransformerEmbedding


class EmbeddingFactory:

    @staticmethod
    def create(
        provider: str,
        model_name: str,
    ):

        provider = provider.lower()


        if provider == "sentence_transformer":

            return SentenceTransformerEmbedding(
                model_name
            )


        else:
            raise ValueError(
                f"Unknown embedding provider: {provider}"
            )
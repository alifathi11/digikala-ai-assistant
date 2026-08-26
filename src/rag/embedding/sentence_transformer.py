from sentence_transformers import SentenceTransformer
import torch

from .base import BaseEmbedding


class SentenceTransformerEmbedding(
    BaseEmbedding
):
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
    ):
        self.device = (
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
        )


    def encode(
        self,
        texts: list[str],
        device: str | None = None,
        batch_size: int = 128,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            device=(
                device
                or self.device
            ),
            batch_size=batch_size,
            convert_to_numpy=(
                convert_to_numpy
            ),
            show_progress_bar=(
                show_progress_bar
            ),
        )

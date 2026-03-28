import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import Protocol, runtime_checkable
import config

@runtime_checkable
class EmbedderProtocol(Protocol):
    """Interface contract for embedding models. Implement this to swap embedders in Stage 2."""
    def load_model(self) -> None: ...
    def encode(self, texts: list[str], is_query: bool = False,
               show_progress: bool = True) -> "np.ndarray": ...

class Embedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL,
                 batch_size: int = config.EMBEDDING_BATCH_SIZE,
                 device: str = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None  # loaded lazily via load_model()

    def load_model(self) -> None:
        """Load SentenceTransformer model onto device."""
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: list[str], is_query: bool = False,
               show_progress: bool = True) -> np.ndarray:
        """
        Encode texts with E5 prefix convention.
        is_query=True  → prepends "query: "
        is_query=False → prepends "passage: "
        Returns (N, 1024) float32, explicitly L2-normalized.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before encode()")
        prefix = "query: " if is_query else "passage: "
        prefixed = [prefix + t for t in texts]
        embeddings = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        assert embeddings.shape[1] == config.EMBEDDING_DIM, (
            f"Embedding dim mismatch: model produced {embeddings.shape[1]}, "
            f"expected config.EMBEDDING_DIM={config.EMBEDDING_DIM}"
        )
        return embeddings

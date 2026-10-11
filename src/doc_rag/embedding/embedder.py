import logging
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.doc_rag.config import settings
from src.doc_rag.db.models import EMBEDDING_DIM

logger = logging.getLogger(__name__)


class Embedder:
    """Turns texts into normalized dense vectors. Build once; the model stays in memory."""

    def __init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        started = time.perf_counter()
        self._model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            device=self._device,
            # Half precision halves VRAM use; CPU inference needs full precision.
            model_kwargs={"torch_dtype": torch.float16} if self._device == "cuda" else None,
        )
        elapsed = time.perf_counter() - started
        logger.info(
            f"Embedder {settings.EMBEDDING_MODEL} ready in {elapsed:.1f}s on {self._device}"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one unit-length vector per text, in the same order."""
        if not texts:
            return []

        started = time.perf_counter()
        vectors = self._encode(texts)
        if vectors.shape[1] != EMBEDDING_DIM:
            # The chunks.embedding column is fixed-size; a different model needs a migration.
            raise ValueError(
                f"{settings.EMBEDDING_MODEL} returned {vectors.shape[1]}-dim vectors, "
                f"but the database expects {EMBEDDING_DIM}"
            )
        logger.info(f"Embedded {len(texts)} texts in {time.perf_counter() - started:.1f}s")
        return vectors.astype("float32").tolist()

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode in batches, halving the batch size whenever the GPU runs out of memory."""
        batch_size = settings.EMBEDDING_BATCH_SIZE
        while True:
            try:
                return self._model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise
                batch_size //= 2
                logger.warning(f"GPU out of memory while embedding; retrying with {batch_size=}")

import logging
import threading
import time

import torch
from sentence_transformers import CrossEncoder

from src.doc_rag.config import settings

logger = logging.getLogger(__name__)

# Query and chunk are scored together; chunks are at most CHUNK_MAX_TOKENS plus their heading.
MAX_PAIR_TOKENS = 1024


class Reranker:
    """Scores (query, chunk) pairs with a cross-encoder. Build once; the model stays in memory."""

    def __init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Ingestion and searches run in different threads; one model call at a time.
        self._lock = threading.Lock()
        started = time.perf_counter()
        self._model = CrossEncoder(
            settings.RERANKER_MODEL,
            device=self._device,
            max_length=MAX_PAIR_TOKENS,
            # Sigmoid turns the raw logit into a relevance score between 0 and 1.
            activation_fn=torch.nn.Sigmoid(),
            # Half precision halves VRAM use; CPU inference needs full precision.
            model_kwargs={"torch_dtype": torch.float16} if self._device == "cuda" else None,
        )
        logger.info(
            f"Reranker {settings.RERANKER_MODEL} ready in "
            f"{time.perf_counter() - started:.1f}s on {self._device}"
        )

    def rerank(self, query: str, texts: list[str], top_k: int) -> list[tuple[int, float]]:
        """Return up to `top_k` pairs of (index into `texts`, score), best first."""
        if not texts:
            return []

        started = time.perf_counter()
        with self._lock:
            ranked = self._model.rank(
                query,
                texts,
                top_k=top_k,
                batch_size=settings.RERANKER_BATCH_SIZE,
                show_progress_bar=False,
            )
        logger.debug(f"Reranked {len(texts)} texts in {time.perf_counter() - started:.2f}s")
        return [(int(item["corpus_id"]), float(item["score"])) for item in ranked]

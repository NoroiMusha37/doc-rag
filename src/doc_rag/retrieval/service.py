import asyncio
import logging
import time
import uuid
from collections.abc import Sequence

from src.doc_rag.config import settings
from src.doc_rag.db.models import Chunk, ChunkKind
from src.doc_rag.db.repository import DBRepository
from src.doc_rag.embedding.embedder import Embedder
from src.doc_rag.embedding.reranker import Reranker
from src.doc_rag.retrieval.fusion import reciprocal_rank_fusion
from src.doc_rag.retrieval.schemas import SearchResult

logger = logging.getLogger(__name__)


class RetrievalService:
    """Hybrid search: dense and keyword search, fused by rank, then reranked."""

    def __init__(self, db: DBRepository, embedder: Embedder, reranker: Reranker) -> None:
        self._db = db
        self._embedder = embedder
        self._reranker = reranker

    async def search(
        self, query: str, top_k: int, document_ids: Sequence[uuid.UUID] | None = None
    ) -> list[SearchResult]:
        """The `top_k` chunks that best answer `query`, most relevant first."""
        started = time.perf_counter()
        # Model inference is blocking; run it in a thread to keep the API responsive.
        (embedding,) = await asyncio.to_thread(self._embedder.embed, [query])

        # One session runs one statement at a time, so the two searches run in sequence.
        dense = await self._db.chunks.search_dense(
            embedding, settings.SEARCH_CANDIDATES, document_ids
        )
        lexical = await self._db.chunks.search_lexical(
            query, settings.SEARCH_CANDIDATES, document_ids
        )

        by_id = {chunk.id: chunk for chunk in (*dense, *lexical)}
        candidates = [
            by_id[chunk_id]
            for chunk_id in reciprocal_rank_fusion(
                [[chunk.id for chunk in dense], [chunk.id for chunk in lexical]],
                k=settings.RRF_K,
                limit=settings.RERANK_CANDIDATES,
            )
        ]

        results = [
            self._to_result(candidates[index], score)
            for index, score in await asyncio.to_thread(
                self._reranker.rerank, query, [chunk.embed_text for chunk in candidates], top_k
            )
        ]
        logger.info(
            f"Search {query!r}: dense={len(dense)} keyword={len(lexical)} "
            f"fused={len(candidates)} -> {len(results)} results "
            f"in {time.perf_counter() - started:.2f}s"
        )
        return results

    @staticmethod
    def _to_result(chunk: Chunk, score: float) -> SearchResult:
        return SearchResult(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            filename=chunk.document.filename,
            # The column stores the enum's value as a plain string.
            kind=ChunkKind(chunk.kind),
            text=chunk.text,
            heading_path=chunk.heading_path,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            asset_path=chunk.asset_path,
            score=score,
        )

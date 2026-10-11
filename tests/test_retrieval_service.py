import uuid
from types import SimpleNamespace

import pytest

from src.doc_rag.config import settings
from src.doc_rag.db.models import ChunkKind
from src.doc_rag.retrieval.service import RetrievalService


def _chunk(text: str) -> SimpleNamespace:
    document_id = uuid.uuid4()
    return SimpleNamespace(
        id=uuid.uuid4(),
        document_id=document_id,
        document=SimpleNamespace(filename="paper.pdf"),
        kind=ChunkKind.TEXT,
        text=text,
        embed_text=f"Heading\n{text}",
        heading_path="Heading",
        page_start=1,
        page_end=2,
        asset_path=None,
    )


class FakeChunks:
    def __init__(self, dense: list, lexical: list) -> None:
        self.dense = dense
        self.lexical = lexical
        self.calls: list[tuple] = []

    async def search_dense(self, embedding, limit, document_ids=None) -> list:
        self.calls.append(("dense", embedding, limit, document_ids))
        return self.dense

    async def search_lexical(self, query, limit, document_ids=None) -> list:
        self.calls.append(("lexical", query, limit, document_ids))
        return self.lexical


class FakeDB:
    def __init__(self, dense: list, lexical: list) -> None:
        self.chunks = FakeChunks(dense, lexical)


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.5, 0.5] for _ in texts]


class FakeReranker:
    """Scores a text by its length, so the longest candidate wins."""

    def __init__(self) -> None:
        self.received: list[str] = []

    def rerank(self, query: str, texts: list[str], top_k: int) -> list[tuple[int, float]]:
        self.received = texts
        scored = sorted(enumerate(texts), key=lambda pair: len(pair[1]), reverse=True)
        return [(index, len(text) / 100) for index, text in scored[:top_k]]


def _service(dense: list, lexical: list) -> tuple[RetrievalService, FakeDB, FakeReranker]:
    db, reranker = FakeDB(dense, lexical), FakeReranker()
    return RetrievalService(db, FakeEmbedder(), reranker), db, reranker


async def test_search_reranks_the_fused_candidates_and_returns_the_best() -> None:
    short, medium, long = _chunk("a"), _chunk("a" * 10), _chunk("a" * 50)
    service, _, reranker = _service(dense=[short, long], lexical=[medium, long])

    results = await service.search("question", top_k=2)

    assert [result.chunk_id for result in results] == [long.id, medium.id]
    assert results[0].score > results[1].score
    # Every distinct chunk from either search reached the reranker exactly once.
    assert sorted(reranker.received) == sorted(c.embed_text for c in (short, medium, long))


async def test_search_maps_the_chunk_to_a_result() -> None:
    chunk = _chunk("Delta is the price sensitivity")
    service, _, _ = _service(dense=[chunk], lexical=[])

    (result,) = await service.search("delta", top_k=1)

    assert result.model_dump() == {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "filename": "paper.pdf",
        "kind": ChunkKind.TEXT,
        "text": "Delta is the price sensitivity",
        "heading_path": "Heading",
        "page_start": 1,
        "page_end": 2,
        "asset_path": None,
        "score": result.score,
    }


async def test_search_passes_the_query_embedding_and_filters_to_both_searches() -> None:
    service, db, _ = _service(dense=[], lexical=[])
    document_ids = [uuid.uuid4()]

    await service.search("BLEU score", top_k=3, document_ids=document_ids)

    assert db.chunks.calls == [
        ("dense", [0.5, 0.5], settings.SEARCH_CANDIDATES, document_ids),
        ("lexical", "BLEU score", settings.SEARCH_CANDIDATES, document_ids),
    ]


@pytest.mark.parametrize(
    ("dense", "lexical"),
    [
        pytest.param([], [], id="nothing-found"),
        pytest.param([_chunk("only dense")], [], id="only-dense"),
        pytest.param([], [_chunk("only keyword")], id="only-keyword"),
    ],
)
async def test_search_works_when_one_or_both_searches_find_nothing(
    dense: list, lexical: list
) -> None:
    service, _, _ = _service(dense, lexical)

    results = await service.search("question", top_k=5)

    assert len(results) == len(dense) + len(lexical)


async def test_search_limits_the_reranker_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "RERANK_CANDIDATES", 3)
    chunks = [_chunk(f"chunk {number}") for number in range(10)]
    service, _, reranker = _service(dense=chunks, lexical=[])

    await service.search("question", top_k=2)

    assert len(reranker.received) == 3

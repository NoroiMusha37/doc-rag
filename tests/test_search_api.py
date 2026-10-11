import uuid
from collections.abc import Iterator

import pytest
from httpx import AsyncClient

from src.doc_rag.config import settings
from src.doc_rag.db.models import ChunkKind
from src.doc_rag.dependencies import get_retrieval_service
from src.doc_rag.main import app
from src.doc_rag.retrieval.schemas import SearchResult


class FakeRetrievalService:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.results = [
            SearchResult(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                filename="paper.pdf",
                kind=ChunkKind.TABLE,
                text="| Model | BLEU |",
                heading_path="5 Results",
                page_start=7,
                page_end=7,
                asset_path=None,
                score=0.91,
            )
        ]

    async def search(self, query: str, top_k: int, document_ids: list | None) -> list[SearchResult]:
        self.calls.append((query, top_k, document_ids))
        return self.results


@pytest.fixture
def service() -> Iterator[FakeRetrievalService]:
    fake = FakeRetrievalService()
    app.dependency_overrides[get_retrieval_service] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


async def test_search_returns_the_results(client: AsyncClient, service: FakeRetrievalService):
    response = await client.post("/search", json={"query": "BLEU score"})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "BLEU score"
    assert body["results"][0]["filename"] == "paper.pdf"
    assert body["results"][0]["page_start"] == 7
    assert body["results"][0]["kind"] == "table"
    assert body["results"][0]["score"] == 0.91


async def test_search_uses_the_default_top_k_and_no_document_filter(
    client: AsyncClient, service: FakeRetrievalService
):
    await client.post("/search", json={"query": "BLEU score"})

    assert service.calls == [("BLEU score", settings.SEARCH_TOP_K, None)]


async def test_search_passes_top_k_and_document_filter(
    client: AsyncClient, service: FakeRetrievalService
):
    document_id = uuid.uuid4()

    await client.post(
        "/search", json={"query": "BLEU", "top_k": 3, "document_ids": [str(document_id)]}
    )

    assert service.calls == [("BLEU", 3, [document_id])]


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="no-query"),
        pytest.param({"query": ""}, id="empty-query"),
        pytest.param({"query": "x" * 1001}, id="query-too-long"),
        pytest.param({"query": "x", "top_k": 0}, id="top-k-zero"),
        pytest.param({"query": "x", "top_k": settings.RERANK_CANDIDATES + 1}, id="top-k-too-big"),
        pytest.param({"query": "x", "document_ids": []}, id="empty-document-filter"),
        pytest.param({"query": "x", "document_ids": ["nope"]}, id="malformed-document-id"),
    ],
)
async def test_search_rejects_invalid_requests(
    client: AsyncClient, service: FakeRetrievalService, body: dict
):
    response = await client.post("/search", json=body)

    assert response.status_code == 422
    assert service.calls == []

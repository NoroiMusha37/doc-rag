import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from src.doc_rag.chat.schemas import ChatDone, ChatError, ChatEvent, ChatSources, ChatToken, Source
from src.doc_rag.config import settings
from src.doc_rag.db.models import ChunkKind, DocumentStatus
from src.doc_rag.db.repository import DBRepository
from src.doc_rag.main import app
from src.doc_rag.mcp_server import server
from src.doc_rag.mcp_server.schemas import AskResponse
from src.doc_rag.retrieval.schemas import SearchResult

SOURCE = Source(
    number=1,
    chunk_id=uuid.uuid4(),
    document_id=uuid.uuid4(),
    filename="paper.pdf",
    kind=ChunkKind.TABLE,
    text="| Model | BLEU |",
    heading_path="5 Results",
    page_start=8,
    page_end=8,
    asset_path=None,
    score=0.98,
)


def _ctx(request: object | None = "default") -> SimpleNamespace:
    """A tool context whose request carries the lifespan objects, like a real HTTP call."""
    if request == "default":
        request = SimpleNamespace(
            state=SimpleNamespace(embedder="embedder", reranker="reranker", llm="llm")
        )
    return SimpleNamespace(request_context=SimpleNamespace(request=request))


class FakeDB:
    def __init__(self) -> None:
        self.documents = SimpleNamespace(list_all=self._list_all)

    async def _list_all(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                id=uuid.uuid4(),
                filename="paper.pdf",
                status=DocumentStatus.READY,
                error=None,
                page_count=15,
                created_at=datetime.now(UTC),
            )
        ]


@pytest.fixture
def db(monkeypatch: pytest.MonkeyPatch) -> FakeDB:
    fake = FakeDB()

    @asynccontextmanager
    async def fake_open() -> AsyncIterator[FakeDB]:
        yield fake

    monkeypatch.setattr(DBRepository, "open", fake_open)
    return fake


class FakeRetrieval:
    built_with: list[tuple] = []
    results = [
        SearchResult(**SOURCE.model_dump(exclude={"number"})),
    ]

    def __init__(self, db, embedder, reranker) -> None:
        self.built_with.append((db, embedder, reranker))
        self.searches: list[tuple] = []

    async def search(self, query: str, top_k: int, document_ids=None) -> list[SearchResult]:
        FakeRetrieval.last_search = (query, top_k, document_ids)
        return self.results


class FakePipeline:
    events: list[ChatEvent] = []
    built_with: list[tuple] = []

    def __init__(self, retrieval, llm) -> None:
        FakePipeline.built_with.append((retrieval, llm))

    async def run(self, question: str, top_k: int, document_ids=None) -> AsyncIterator[ChatEvent]:
        FakePipeline.last_run = (question, top_k, document_ids)
        for event in self.events:
            yield event


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch, db: FakeDB) -> None:
    FakeRetrieval.built_with.clear()
    FakePipeline.built_with.clear()
    FakePipeline.events = [
        ChatSources(sources=[SOURCE]),
        ChatToken(text="BLEU is 28.4 [1]."),
        ChatDone(answer="BLEU is 28.4 [1].", citations=[1]),
    ]
    monkeypatch.setattr(server, "RetrievalService", FakeRetrieval)
    monkeypatch.setattr(server, "ChatPipeline", FakePipeline)


async def test_list_documents_returns_the_documents(db: FakeDB) -> None:
    (document,) = await server.list_documents()

    assert document.filename == "paper.pdf"
    assert document.status is DocumentStatus.READY
    assert document.page_count == 15


async def test_search_documents_searches_with_the_lifespan_models(fakes) -> None:
    document_ids = [uuid.uuid4()]

    results = await server.search_documents("BLEU score", _ctx(), 3, document_ids)

    assert results == FakeRetrieval.results
    assert FakeRetrieval.last_search == ("BLEU score", 3, document_ids)
    (_, embedder, reranker) = FakeRetrieval.built_with[0]
    assert (embedder, reranker) == ("embedder", "reranker")


async def test_ask_documents_returns_the_answer_citations_and_sources(fakes) -> None:
    response = await server.ask_documents("What BLEU?", _ctx())

    assert response == AskResponse(answer="BLEU is 28.4 [1].", citations=[1], sources=[SOURCE])
    assert FakePipeline.last_run == ("What BLEU?", settings.SEARCH_TOP_K, None)
    assert FakePipeline.built_with[0][1] == "llm"


async def test_ask_documents_reports_a_failed_answer_as_a_tool_error(fakes) -> None:
    FakePipeline.events = [ChatSources(sources=[]), ChatError(message="Gemini is overloaded.")]

    with pytest.raises(ToolError, match="Gemini is overloaded."):
        await server.ask_documents("What BLEU?", _ctx())


async def test_ask_documents_fails_clearly_when_the_stream_has_no_result(fakes) -> None:
    FakePipeline.events = [ChatSources(sources=[])]

    with pytest.raises(ToolError, match="without a result"):
        await server.ask_documents("What BLEU?", _ctx())


@pytest.mark.parametrize("tool", ["search_documents", "ask_documents"])
async def test_tools_refuse_calls_that_did_not_come_over_http(fakes, tool: str) -> None:
    with pytest.raises(ToolError, match="HTTP"):
        await getattr(server, tool)("question", _ctx(request=None))


async def test_the_three_tools_are_registered_without_exposing_the_context() -> None:
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}

    assert set(tools) == {"list_documents", "search_documents", "ask_documents"}
    for tool in tools.values():
        assert "ctx" not in tool.input_schema["properties"]
        assert tool.description
    assert tools["search_documents"].input_schema["required"] == ["query"]
    assert tools["ask_documents"].input_schema["required"] == ["question"]
    top_k = tools["search_documents"].input_schema["properties"]["top_k"]
    assert (top_k["minimum"], top_k["maximum"]) == (1, settings.RERANK_CANDIDATES)


def test_the_app_serves_mcp_at_exactly_slash_mcp() -> None:
    assert [route.path for route in app.routes if "mcp" in getattr(route, "path", "")] == ["/mcp"]

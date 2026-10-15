import json
import uuid
from collections.abc import AsyncIterator, Iterator

import pytest
from httpx import AsyncClient

from src.doc_rag.chat.schemas import (
    ChatDone,
    ChatError,
    ChatEvent,
    ChatSources,
    ChatStep,
    ChatToken,
    Source,
)
from src.doc_rag.config import settings
from src.doc_rag.db.models import ChunkKind
from src.doc_rag.dependencies import get_chat_graph, get_chat_pipeline
from src.doc_rag.main import app

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


class FakeChatPipeline:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.events: list[ChatEvent] = [
            ChatSources(sources=[SOURCE]),
            ChatToken(text="BLEU is "),
            ChatToken(text="28.4 [1]."),
            ChatDone(answer="BLEU is 28.4 [1].", citations=[1]),
        ]

    async def run(self, question: str, top_k: int, document_ids=None) -> AsyncIterator[ChatEvent]:
        self.calls.append((question, top_k, document_ids))
        for event in self.events:
            yield event


@pytest.fixture
def pipeline() -> Iterator[FakeChatPipeline]:
    fake = FakeChatPipeline()
    app.dependency_overrides[get_chat_pipeline] = lambda: fake
    # A second fake with its own call log, so a test can tell which orchestrator was used.
    app.dependency_overrides[get_chat_graph] = lambda: fake.graph
    fake.graph = FakeChatPipeline()
    yield fake
    app.dependency_overrides.clear()


def _parse(body: str) -> list[tuple[str, dict]]:
    """Split a Server-Sent Events body into (event name, JSON data) pairs."""
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


async def test_chat_streams_the_events_in_order(client: AsyncClient, pipeline: FakeChatPipeline):
    response = await client.post("/chat", json={"question": "What BLEU does it reach?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse(response.text)
    assert [name for name, _ in events] == ["sources", "token", "token", "done"]
    assert events[0][1]["sources"][0]["number"] == 1
    assert events[0][1]["sources"][0]["page_start"] == 8
    assert "".join(data["text"] for name, data in events if name == "token") == "BLEU is 28.4 [1]."
    assert events[-1][1] == {"event": "done", "answer": "BLEU is 28.4 [1].", "citations": [1]}


async def test_chat_passes_the_request_to_the_pipeline(
    client: AsyncClient, pipeline: FakeChatPipeline
):
    document_id = uuid.uuid4()

    await client.post(
        "/chat", json={"question": "Why?", "top_k": 3, "document_ids": [str(document_id)]}
    )
    await client.post("/chat", json={"question": "Why?"})

    assert pipeline.calls == [
        ("Why?", 3, [document_id]),
        ("Why?", settings.SEARCH_TOP_K, None),
    ]


async def test_chat_uses_the_simple_pipeline_unless_graph_mode_is_asked_for(
    client: AsyncClient, pipeline: FakeChatPipeline
):
    await client.post("/chat", json={"question": "Why?"})
    await client.post("/chat", json={"question": "Why?", "mode": "simple"})
    assert len(pipeline.calls) == 2
    assert pipeline.graph.calls == []

    await client.post("/chat", json={"question": "Why?", "mode": "graph", "top_k": 4})

    assert len(pipeline.calls) == 2
    assert pipeline.graph.calls == [("Why?", 4, None)]


async def test_chat_streams_the_steps_of_the_graph_before_the_sources(
    client: AsyncClient, pipeline: FakeChatPipeline
):
    pipeline.graph.events = [
        ChatStep(message="The best result scored 0.12; searching again for: attention heads"),
        *pipeline.graph.events,
    ]

    response = await client.post("/chat", json={"question": "Why?", "mode": "graph"})

    events = _parse(response.text)
    assert [name for name, _ in events] == ["step", "sources", "token", "token", "done"]
    assert events[0][1] == {
        "event": "step",
        "message": "The best result scored 0.12; searching again for: attention heads",
    }


async def test_chat_streams_an_error_event(client: AsyncClient, pipeline: FakeChatPipeline):
    pipeline.events = [ChatSources(sources=[]), ChatError(message="Rate limit reached.")]

    response = await client.post("/chat", json={"question": "Why?"})

    assert response.status_code == 200
    assert _parse(response.text)[-1] == (
        "error",
        {"event": "error", "message": "Rate limit reached."},
    )


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="no-question"),
        pytest.param({"question": ""}, id="empty-question"),
        pytest.param({"question": "x" * 2001}, id="question-too-long"),
        pytest.param({"question": "x", "top_k": 0}, id="top-k-zero"),
        pytest.param({"question": "x", "document_ids": []}, id="empty-document-filter"),
        pytest.param({"question": "x", "mode": "agent"}, id="unknown-mode"),
    ],
)
async def test_chat_rejects_invalid_requests(
    client: AsyncClient, pipeline: FakeChatPipeline, body: dict
):
    response = await client.post("/chat", json=body)

    assert response.status_code == 422
    assert pipeline.calls == []
    assert pipeline.graph.calls == []

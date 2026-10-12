import uuid
from collections.abc import AsyncIterator

import pytest

from src.doc_rag.chat.pipeline import ChatPipeline
from src.doc_rag.chat.schemas import ChatDone, ChatError, ChatEvent, ChatSources, ChatToken
from src.doc_rag.config import settings
from src.doc_rag.db.models import ChunkKind
from src.doc_rag.llm.base import LLMError
from src.doc_rag.llm.prompts import NOT_FOUND_ANSWER
from src.doc_rag.retrieval.schemas import SearchResult


def _result(text: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="paper.pdf",
        kind=ChunkKind.TEXT,
        text=text,
        heading_path="Greeks",
        page_start=3,
        page_end=3,
        asset_path=None,
        score=score,
    )


class FakeRetrieval:
    def __init__(self, results: list[SearchResult], error: Exception | None = None) -> None:
        self.results = results
        self.error = error
        self.calls: list[tuple] = []

    async def search(self, query: str, top_k: int, document_ids=None) -> list[SearchResult]:
        self.calls.append((query, top_k, document_ids))
        if self.error is not None:
            raise self.error
        return self.results


class FakeLLM:
    def __init__(self, pieces: list[str], error: Exception | None = None) -> None:
        self.pieces = pieces
        self.error = error
        self.prompts: list[str] = []

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        self.prompts.append(prompt)
        for piece in self.pieces:
            yield piece
        if self.error is not None:
            raise self.error


async def _run(retrieval: FakeRetrieval, llm: FakeLLM, **kwargs) -> list[ChatEvent]:
    return [
        event async for event in ChatPipeline(retrieval, llm).run("What is delta?", 6, **kwargs)
    ]


async def test_run_sends_sources_then_tokens_then_the_full_answer_with_citations() -> None:
    retrieval = FakeRetrieval([_result("Delta is 0.5.", 0.9), _result("Gamma is 0.1.", 0.7)])
    llm = FakeLLM(["Delta is ", "0.5 [1]."])

    events = await _run(retrieval, llm)

    assert [type(event) for event in events] == [ChatSources, ChatToken, ChatToken, ChatDone]
    assert [source.number for source in events[0].sources] == [1, 2]
    assert [event.text for event in events[1:3]] == ["Delta is ", "0.5 [1]."]
    assert events[3] == ChatDone(answer="Delta is 0.5 [1].", citations=[1])


async def test_run_passes_the_request_to_the_search_and_the_sources_to_the_llm() -> None:
    retrieval = FakeRetrieval([_result("Delta is 0.5.", 0.9)])
    llm = FakeLLM(["ok [1]"])
    document_ids = [uuid.uuid4()]

    await _run(retrieval, llm, document_ids=document_ids)

    assert retrieval.calls == [("What is delta?", 6, document_ids)]
    assert "[1] paper.pdf" in llm.prompts[0]
    assert "Delta is 0.5." in llm.prompts[0]
    assert llm.prompts[0].endswith("Question: What is delta?")


async def test_results_below_the_minimum_score_are_dropped_before_numbering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_MIN_SCORE", 0.5)
    retrieval = FakeRetrieval([_result("relevant", 0.8), _result("noise", 0.2)])
    llm = FakeLLM(["answer [1]"])

    events = await _run(retrieval, llm)

    assert [(source.number, source.text) for source in events[0].sources] == [(1, "relevant")]
    assert "noise" not in llm.prompts[0]


async def test_a_question_with_no_relevant_sources_is_declined_without_calling_the_llm() -> None:
    llm = FakeLLM(["should not be used"])

    events = await _run(FakeRetrieval([_result("off topic", 0.001)]), llm)

    assert llm.prompts == []
    assert events[0] == ChatSources(sources=[])
    assert events[-1] == ChatDone(answer=NOT_FOUND_ANSWER, citations=[])


async def test_an_llm_failure_becomes_an_error_event_with_its_message() -> None:
    llm = FakeLLM(["partial "], error=LLMError("The Gemini free-tier limit was reached."))

    events = await _run(FakeRetrieval([_result("Delta is 0.5.", 0.9)]), llm)

    assert [type(event) for event in events] == [ChatSources, ChatToken, ChatError]
    assert events[-1].message == "The Gemini free-tier limit was reached."


async def test_an_unexpected_failure_becomes_a_generic_error_event() -> None:
    events = await _run(FakeRetrieval([], error=RuntimeError("database is down")), FakeLLM([]))

    assert [type(event) for event in events] == [ChatError]
    assert "database is down" not in events[0].message


@pytest.mark.parametrize(
    ("answer", "source_count", "citations"),
    [
        ("Delta is 0.5 [1].", 2, [1]),
        ("Both hold [2][1], see also [2].", 2, [1, 2]),
        ("It cites [7], which does not exist, and [0].", 2, []),
        ("No citation at all.", 2, []),
        ("A list x[3] and a range [1-2] are not citations of 1 or 2.", 3, [3]),
    ],
)
def test_citations_are_the_existing_source_numbers_found_in_the_answer(
    answer: str, source_count: int, citations: list[int]
) -> None:
    assert ChatPipeline._citations(answer, source_count) == citations

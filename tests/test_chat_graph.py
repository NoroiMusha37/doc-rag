import uuid
from collections.abc import AsyncIterator

import pytest

from src.doc_rag.chat.graph import ChatGraph
from src.doc_rag.chat.schemas import (
    ChatDone,
    ChatError,
    ChatEvent,
    ChatSources,
    ChatStep,
    ChatToken,
)
from src.doc_rag.config import settings
from src.doc_rag.db.models import ChunkKind
from src.doc_rag.llm.base import LLMError
from src.doc_rag.llm.prompts import NOT_FOUND_ANSWER, REWRITE_SYSTEM_PROMPT
from src.doc_rag.retrieval.schemas import SearchResult

QUESTION = "how many heads does it have?"


def _result(text: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="paper.pdf",
        kind=ChunkKind.TEXT,
        text=text,
        heading_path="Model",
        page_start=5,
        page_end=5,
        asset_path=None,
        score=score,
    )


class FakeRetrieval:
    """Returns the results scripted for each query; any other query finds nothing."""

    def __init__(self, by_query: dict[str, list[SearchResult]], error: Exception | None = None):
        self.by_query = by_query
        self.error = error
        self.calls: list[tuple] = []

    async def search(self, query: str, top_k: int, document_ids=None) -> list[SearchResult]:
        self.calls.append((query, top_k, document_ids))
        if self.error is not None:
            raise self.error
        return self.by_query.get(query, [])

    @property
    def queries(self) -> list[str]:
        return [call[0] for call in self.calls]


class FakeLLM:
    """Plays both roles: each rewrite request takes the next scripted rewrite (a string,
    or an error to raise); every other request is an answer."""

    def __init__(
        self,
        rewrites: list[str | Exception] | None = None,
        answer: list[str] | None = None,
        answer_error: Exception | None = None,
    ) -> None:
        self.rewrites = list(rewrites or [])
        self.answer = answer if answer is not None else ["It has 8 [1]."]
        self.answer_error = answer_error
        self.rewrite_prompts: list[str] = []
        self.answer_prompts: list[str] = []

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        if system == REWRITE_SYSTEM_PROMPT:
            self.rewrite_prompts.append(prompt)
            rewrite = self.rewrites.pop(0)
            if isinstance(rewrite, Exception):
                raise rewrite
            yield rewrite
            return
        self.answer_prompts.append(prompt)
        for piece in self.answer:
            yield piece
        if self.answer_error is not None:
            raise self.answer_error


async def _run(retrieval: FakeRetrieval, llm: FakeLLM, **kwargs) -> list[ChatEvent]:
    return [event async for event in ChatGraph(retrieval, llm).run(QUESTION, 6, **kwargs)]


def _types(events: list[ChatEvent]) -> list[type]:
    return [type(event) for event in events]


async def test_good_results_are_answered_at_once_without_rewriting() -> None:
    retrieval = FakeRetrieval({QUESTION: [_result("h = 8 heads", 0.9)]})
    llm = FakeLLM()

    events = await _run(retrieval, llm)

    assert _types(events) == [ChatSources, ChatToken, ChatDone]
    assert events[-1] == ChatDone(answer="It has 8 [1].", citations=[1])
    assert retrieval.queries == [QUESTION]
    assert llm.rewrite_prompts == []
    assert len(llm.answer_prompts) == 1


async def test_weak_results_make_the_graph_rewrite_and_search_again() -> None:
    retrieval = FakeRetrieval(
        {
            QUESTION: [_result("unrelated", 0.12)],
            "attention heads base model": [_result("h = 8 heads", 0.85)],
        }
    )
    llm = FakeLLM(rewrites=["attention heads base model"])

    events = await _run(retrieval, llm)

    assert _types(events) == [ChatStep, ChatSources, ChatToken, ChatDone]
    assert "0.12" in events[0].message
    assert "attention heads base model" in events[0].message
    assert retrieval.queries == [QUESTION, "attention heads base model"]
    assert [source.text for source in events[1].sources] == ["h = 8 heads"]
    # The answer is still about the user's own question, not the search query.
    assert llm.answer_prompts[0].endswith(f"Question: {QUESTION}")
    assert len(llm.answer_prompts) == 1


async def test_the_rewrite_prompt_names_the_question_and_the_queries_already_tried() -> None:
    retrieval = FakeRetrieval({QUESTION: [_result("unrelated", 0.12)]})
    llm = FakeLLM(rewrites=["new words"])

    await _run(retrieval, llm)

    assert f"Question: {QUESTION}" in llm.rewrite_prompts[0]
    assert f"- {QUESTION}" in llm.rewrite_prompts[0]


async def test_the_search_with_the_better_top_result_is_the_one_used() -> None:
    retrieval = FakeRetrieval(
        {QUESTION: [_result("first", 0.2)], "other words": [_result("second", 0.1)]}
    )

    events = await _run(retrieval, FakeLLM(rewrites=["other words"]))

    assert [source.text for source in events[1].sources] == ["first"]


async def test_a_question_nothing_matches_is_declined_after_one_rewrite() -> None:
    retrieval = FakeRetrieval({QUESTION: [_result("noise", 0.002)]})
    llm = FakeLLM(rewrites=["something else"])

    events = await _run(retrieval, llm)

    assert retrieval.queries == [QUESTION, "something else"]
    assert llm.answer_prompts == []  # declined without an answer call
    assert events[-1] == ChatDone(answer=NOT_FOUND_ANSWER, citations=[])
    assert events[-2] == ChatToken(text=NOT_FOUND_ANSWER)


async def test_no_retry_is_made_when_retries_are_turned_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "GRAPH_MAX_RETRIES", 0)
    retrieval = FakeRetrieval({QUESTION: [_result("weak but relevant", 0.2)]})
    llm = FakeLLM()

    events = await _run(retrieval, llm)

    assert _types(events) == [ChatSources, ChatToken, ChatDone]
    assert retrieval.queries == [QUESTION]
    assert llm.rewrite_prompts == []


async def test_retries_stop_at_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "GRAPH_MAX_RETRIES", 2)
    retrieval = FakeRetrieval({QUESTION: [_result("noise", 0.001)]})
    llm = FakeLLM(rewrites=["second try", "third try", "never asked"])

    events = await _run(retrieval, llm)

    assert retrieval.queries == [QUESTION, "second try", "third try"]
    assert len(llm.rewrite_prompts) == 2
    assert [type(e) for e in events].count(ChatStep) == 2


@pytest.mark.parametrize(
    "rewrite",
    [
        pytest.param(QUESTION.upper(), id="same-as-the-question"),
        pytest.param("   ", id="blank"),
        pytest.param('""', id="only-quotes"),
    ],
)
async def test_a_rewrite_that_adds_nothing_does_not_trigger_another_search(rewrite: str) -> None:
    retrieval = FakeRetrieval({QUESTION: [_result("weak but relevant", 0.2)]})

    events = await _run(retrieval, FakeLLM(rewrites=[rewrite]))

    assert retrieval.queries == [QUESTION]
    assert ChatStep not in _types(events)
    assert _types(events)[-1] is ChatDone


async def test_the_rewritten_query_is_cleaned_up() -> None:
    retrieval = FakeRetrieval({QUESTION: [_result("noise", 0.001)]})

    await _run(retrieval, FakeLLM(rewrites=['"attention\nheads  base model"\n']))

    assert retrieval.queries[1] == "attention heads base model"


async def test_a_failed_rewrite_still_answers_from_the_first_results() -> None:
    retrieval = FakeRetrieval({QUESTION: [_result("weak but relevant", 0.2)]})
    llm = FakeLLM(rewrites=[LLMError("The Gemini free-tier limit was reached.")])

    events = await _run(retrieval, llm)

    assert _types(events) == [ChatSources, ChatToken, ChatDone]
    assert retrieval.queries == [QUESTION]


async def test_a_failed_rewrite_does_not_stop_an_off_topic_question_from_being_declined() -> None:
    retrieval = FakeRetrieval({QUESTION: [_result("noise", 0.002)]})
    llm = FakeLLM(rewrites=[LLMError("The Gemini free-tier limit was reached.")])

    events = await _run(retrieval, llm)

    assert events[-1] == ChatDone(answer=NOT_FOUND_ANSWER, citations=[])


async def test_every_search_gets_the_requested_size_and_document_filter() -> None:
    retrieval = FakeRetrieval({QUESTION: [_result("noise", 0.001)]})
    document_ids = [uuid.uuid4()]

    await _run(retrieval, FakeLLM(rewrites=["second try"]), document_ids=document_ids)

    assert retrieval.calls == [(QUESTION, 6, document_ids), ("second try", 6, document_ids)]


async def test_an_llm_failure_while_answering_becomes_an_error_event() -> None:
    retrieval = FakeRetrieval({QUESTION: [_result("h = 8 heads", 0.9)]})
    llm = FakeLLM(answer=["partial "], answer_error=LLMError("Gemini is overloaded right now."))

    events = await _run(retrieval, llm)

    assert _types(events) == [ChatSources, ChatToken, ChatError]
    assert events[-1].message == "Gemini is overloaded right now."


async def test_an_unexpected_failure_becomes_a_generic_error_event() -> None:
    retrieval = FakeRetrieval({}, error=RuntimeError("database is down"))

    events = await _run(retrieval, FakeLLM())

    assert _types(events) == [ChatError]
    assert "database is down" not in events[0].message


async def test_concurrent_questions_do_not_share_state() -> None:
    import asyncio

    async def ask(question: str, text: str) -> list[ChatEvent]:
        retrieval = FakeRetrieval({question: [_result(text, 0.9)]})
        return [event async for event in ChatGraph(retrieval, FakeLLM()).run(question, 6)]

    first, second = await asyncio.gather(ask("one?", "text one"), ask("two?", "text two"))

    assert first[0].sources[0].text == "text one"
    assert second[0].sources[0].text == "text two"

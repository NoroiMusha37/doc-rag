import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Sequence

from src.doc_rag.chat.schemas import ChatDone, ChatError, ChatEvent, ChatSources, ChatToken, Source
from src.doc_rag.config import settings
from src.doc_rag.llm.base import LLMClient, LLMError
from src.doc_rag.llm.prompts import NOT_FOUND_ANSWER, SYSTEM_PROMPT, build_prompt
from src.doc_rag.retrieval.schemas import SearchResult
from src.doc_rag.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

# Matches citations like [3]; a citation list like [1][2] matches twice.
CITATION = re.compile(r"\[(\d+)\]")


def number_sources(results: Sequence[SearchResult]) -> list[Source]:
    """Drop results that are not relevant enough and number the rest from 1."""
    return [
        Source(number=number, **result.model_dump())
        for number, result in enumerate(
            (result for result in results if result.score >= settings.CHAT_MIN_SCORE), start=1
        )
    ]


def extract_citations(answer: str, source_count: int) -> list[int]:
    """The source numbers cited in `answer`, ascending, ignoring numbers that do not exist."""
    return sorted(
        {number for number in map(int, CITATION.findall(answer)) if 1 <= number <= source_count}
    )


async def respond(llm: LLMClient, question: str, sources: list[Source]) -> AsyncIterator[ChatEvent]:
    """The events of an answer from `sources`: `sources`, then tokens, then `done`.

    Shared by both orchestrators, so they cannot drift apart in how they answer. Raises
    `LLMError`; wrap the events in `guard` to turn failures into an `error` event.
    """
    yield ChatSources(sources=sources)
    parts: list[str] = []
    if sources:
        async for text in llm.stream(SYSTEM_PROMPT, build_prompt(question, sources)):
            parts.append(text)
            yield ChatToken(text=text)
    else:
        # Nothing relevant was found: decline without spending an LLM call.
        parts.append(NOT_FOUND_ANSWER)
        yield ChatToken(text=NOT_FOUND_ANSWER)
    answer = "".join(parts)
    yield ChatDone(answer=answer, citations=extract_citations(answer, len(sources)))


async def guard(question: str, events: AsyncIterator[ChatEvent]) -> AsyncIterator[ChatEvent]:
    """Pass the events through; a failure becomes an `error` event instead of an exception.

    The response is already streaming when a failure happens, so it cannot become an HTTP
    error. Also logs each finished answer.
    """
    started = time.perf_counter()
    source_count = 0
    try:
        async for event in events:
            if isinstance(event, ChatSources):
                source_count = len(event.sources)
            elif isinstance(event, ChatDone):
                logger.info(
                    f"Chat {question!r}: {source_count} sources, cited {event.citations} "
                    f"in {time.perf_counter() - started:.2f}s"
                )
            yield event
    except LLMError as exc:
        logger.warning(f"Chat {question!r} failed: {exc}")
        yield ChatError(message=str(exc))
    except Exception:
        logger.exception(f"Chat {question!r} failed unexpectedly")
        yield ChatError(message="Something went wrong while answering. Check the server log.")


class ChatPipeline:
    """Answers a question from the documents: search, keep relevant chunks, ask the LLM."""

    def __init__(self, retrieval: RetrievalService, llm: LLMClient) -> None:
        self._retrieval = retrieval
        self._llm = llm

    async def run(
        self, question: str, top_k: int, document_ids: Sequence[uuid.UUID] | None = None
    ) -> AsyncIterator[ChatEvent]:
        """Yield the events of one answer: sources, then tokens, then done or error.

        Never raises: failures are reported as an `error` event.
        """
        async for event in guard(question, self._events(question, top_k, document_ids)):
            yield event

    async def _events(
        self, question: str, top_k: int, document_ids: Sequence[uuid.UUID] | None
    ) -> AsyncIterator[ChatEvent]:
        sources = number_sources(await self._retrieval.search(question, top_k, document_ids))
        async for event in respond(self._llm, question, sources):
            yield event

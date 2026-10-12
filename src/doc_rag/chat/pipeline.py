import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Sequence

from src.doc_rag.chat.schemas import ChatDone, ChatError, ChatEvent, ChatSources, ChatToken, Source
from src.doc_rag.config import settings
from src.doc_rag.llm.base import LLMClient, LLMError
from src.doc_rag.llm.prompts import NOT_FOUND_ANSWER, SYSTEM_PROMPT, build_prompt
from src.doc_rag.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

# Matches citations like [3]; a citation list like [1][2] matches twice.
CITATION = re.compile(r"\[(\d+)\]")


class ChatPipeline:
    """Answers a question from the documents: search, keep relevant chunks, ask the LLM."""

    def __init__(self, retrieval: RetrievalService, llm: LLMClient) -> None:
        self._retrieval = retrieval
        self._llm = llm

    async def run(
        self, question: str, top_k: int, document_ids: Sequence[uuid.UUID] | None = None
    ) -> AsyncIterator[ChatEvent]:
        """Yield the events of one answer: sources, then tokens, then done or error.

        Never raises: the response is already streaming when a failure happens, so
        the failure is reported as an `error` event.
        """
        started = time.perf_counter()
        try:
            sources = await self._find_sources(question, top_k, document_ids)
            yield ChatSources(sources=sources)

            parts: list[str] = []
            async for text in self._answer(question, sources):
                parts.append(text)
                yield ChatToken(text=text)
            answer = "".join(parts)
            done = ChatDone(answer=answer, citations=self._citations(answer, len(sources)))
        except LLMError as exc:
            logger.warning(f"Chat {question!r} failed: {exc}")
            yield ChatError(message=str(exc))
            return
        except Exception:
            logger.exception(f"Chat {question!r} failed unexpectedly")
            yield ChatError(message="Something went wrong while answering. Check the server log.")
            return

        logger.info(
            f"Chat {question!r}: {len(sources)} sources, cited {done.citations} "
            f"in {time.perf_counter() - started:.2f}s"
        )
        yield done

    async def _find_sources(
        self, question: str, top_k: int, document_ids: Sequence[uuid.UUID] | None
    ) -> list[Source]:
        """Search, drop results that are not relevant enough, and number the rest from 1."""
        return [
            Source(number=number, **result.model_dump())
            for number, result in enumerate(
                (
                    result
                    for result in await self._retrieval.search(question, top_k, document_ids)
                    if result.score >= settings.CHAT_MIN_SCORE
                ),
                start=1,
            )
        ]

    async def _answer(self, question: str, sources: list[Source]) -> AsyncIterator[str]:
        if not sources:
            # Nothing relevant was found: decline without spending an LLM call.
            yield NOT_FOUND_ANSWER
            return
        async for text in self._llm.stream(SYSTEM_PROMPT, build_prompt(question, sources)):
            yield text

    @staticmethod
    def _citations(answer: str, source_count: int) -> list[int]:
        """The source numbers cited in `answer`, ascending, ignoring numbers that do not exist."""
        return sorted(
            {number for number in map(int, CITATION.findall(answer)) if 1 <= number <= source_count}
        )

"""The LangGraph orchestrator: the same answer as `pipeline.py`, with a second chance.

    retrieve -> grade --good--------------------------------> answer
                  |                                             ^
                  +--weak--> rewrite --new query--> retrieve ---+  (at most GRAPH_MAX_RETRIES times)

The question is searched as it was asked. If the best result looks weak, the LLM rewrites
the question into a different search query and the search runs again; whichever search
found the better top result is used to answer. This is the only module that imports
LangGraph. Searching, numbering sources and answering are shared with `ChatPipeline`.
"""

import logging
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from src.doc_rag.chat.pipeline import guard, number_sources, respond
from src.doc_rag.chat.schemas import ChatEvent, ChatStep
from src.doc_rag.config import settings
from src.doc_rag.llm.base import LLMClient, LLMError
from src.doc_rag.llm.prompts import REWRITE_SYSTEM_PROMPT, build_rewrite_prompt
from src.doc_rag.retrieval.schemas import SearchResult
from src.doc_rag.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 300


class GraphState(TypedDict):
    question: str
    top_k: int
    document_ids: Sequence[uuid.UUID] | None
    queries: list[str]
    searches: int
    results: list[SearchResult]
    good: bool


class GraphUpdate(TypedDict, total=False):
    """What a node may return: only the keys it changes. The request fields never change."""

    queries: list[str]
    searches: int
    results: list[SearchResult]
    good: bool


@dataclass
class GraphContext:
    """What the nodes need that differs per request. The compiled graph is built once."""

    retrieval: RetrievalService
    llm: LLMClient


def _top_score(results: Sequence[SearchResult]) -> float:
    return results[0].score if results else 0.0


async def retrieve(state: GraphState, runtime: Runtime[GraphContext]) -> GraphUpdate:
    found = await runtime.context.retrieval.search(
        state["queries"][-1], state["top_k"], state["document_ids"]
    )
    better = found if _top_score(found) > _top_score(state["results"]) else state["results"]
    return {"results": better, "searches": state["searches"] + 1}


def grade(state: GraphState) -> GraphUpdate:
    """Are the results good enough to answer from?

    The reranker is the grader: a cross-encoder trained to judge relevance, free and
    repeatable. Asking the LLM would cost another free-tier request on every question.
    """
    top = _top_score(state["results"])
    good = top >= settings.GRAPH_GOOD_SCORE
    logger.info(
        f"Graded search {state['searches']} of {state['question']!r}: top {top:.3f}, good={good}"
    )
    return {"good": good}


def after_grade(state: GraphState) -> Literal["rewrite", "answer"]:
    if state["good"] or state["searches"] > settings.GRAPH_MAX_RETRIES:
        return "answer"
    return "rewrite"


async def rewrite(state: GraphState, runtime: Runtime[GraphContext]) -> GraphUpdate:
    """Ask the LLM for a differently worded query. Leaves `queries` alone if that fails."""
    try:
        raw = "".join(
            [
                piece
                async for piece in runtime.context.llm.stream(
                    REWRITE_SYSTEM_PROMPT,
                    build_rewrite_prompt(state["question"], state["queries"]),
                )
            ]
        )
    except LLMError as exc:
        logger.warning(f"Could not rewrite {state['question']!r}: {exc}")
        return {}
    query = " ".join(raw.split()).strip("\"'")[:MAX_QUERY_CHARS]
    if not query or query.lower() in {tried.lower() for tried in state["queries"]}:
        return {}  # nothing new to search for
    runtime.stream_writer(
        ChatStep(
            message=f"The best result scored {_top_score(state['results']):.2f}; "
            f"searching again for: {query}"
        )
    )
    return {"queries": [*state["queries"], query]}


def after_rewrite(state: GraphState) -> Literal["retrieve", "answer"]:
    return "retrieve" if len(state["queries"]) > state["searches"] else "answer"


async def answer(state: GraphState, runtime: Runtime[GraphContext]) -> GraphUpdate:
    async for event in respond(
        runtime.context.llm, state["question"], number_sources(state["results"])
    ):
        runtime.stream_writer(event)
    return {}


# PyCharm cannot match a TypedDict or a dataclass to LangGraph's structural state and
# context types (it rejects even the dataclass); pyright accepts them, so only PyCharm's

# noinspection PyTypeChecker
compiled_graph = (
    StateGraph(GraphState, context_schema=GraphContext)
    .add_node("retrieve", retrieve)
    .add_node("grade", grade)
    .add_node("rewrite", rewrite)
    .add_node("answer", answer)
    .add_edge(START, "retrieve")
    .add_edge("retrieve", "grade")
    .add_conditional_edges("grade", after_grade)
    .add_conditional_edges("rewrite", after_rewrite)
    .add_edge("answer", END)
    .compile()
)


class ChatGraph:
    """Same interface as `ChatPipeline`: `run` yields the events of one answer."""

    def __init__(self, retrieval: RetrievalService, llm: LLMClient) -> None:
        self._context = GraphContext(retrieval, llm)

    async def run(
        self, question: str, top_k: int, document_ids: Sequence[uuid.UUID] | None = None
    ) -> AsyncIterator[ChatEvent]:
        """Yield `step` events as the graph works, then sources, tokens and done or error.

        Never raises: failures are reported as an `error` event.
        """
        async for event in guard(question, self._events(question, top_k, document_ids)):
            yield event

    async def _events(
        self, question: str, top_k: int, document_ids: Sequence[uuid.UUID] | None
    ) -> AsyncIterator[ChatEvent]:
        async for event in compiled_graph.astream(
            GraphState(
                question=question,
                top_k=top_k,
                document_ids=document_ids,
                queries=[question],
                searches=0,
                results=[],
                good=False,
            ),
            context=self._context,
            stream_mode="custom",
        ):
            yield cast(ChatEvent, event)

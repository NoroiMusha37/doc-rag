import uuid
from typing import Literal

from pydantic import BaseModel, Field

from src.doc_rag.config import settings
from src.doc_rag.retrieval.schemas import SearchResult


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # How many chunks to search for; fewer may be used if some are not relevant enough.
    top_k: int = Field(default=settings.SEARCH_TOP_K, ge=1, le=settings.RERANK_CANDIDATES)
    # Answer only from these documents; omit to use all of them.
    document_ids: list[uuid.UUID] | None = Field(default=None, min_length=1)


class Source(SearchResult):
    # The number the answer uses to cite this chunk, e.g. [2].
    number: int


# The events of one answer, in the order they are sent: `sources` once, `token` many
# times, then either `done` or `error`.


class ChatSources(BaseModel):
    event: Literal["sources"] = "sources"
    sources: list[Source]


class ChatToken(BaseModel):
    event: Literal["token"] = "token"
    # The next piece of the answer; concatenate all pieces to get the full text.
    text: str


class ChatDone(BaseModel):
    event: Literal["done"] = "done"
    answer: str
    # Numbers of the sources the answer actually cites, ascending.
    citations: list[int]


class ChatError(BaseModel):
    event: Literal["error"] = "error"
    message: str


ChatEvent = ChatSources | ChatToken | ChatDone | ChatError

import uuid
from typing import Annotated

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field
from starlette.datastructures import State
from starlette.requests import Request

from src.doc_rag.chat.pipeline import ChatPipeline
from src.doc_rag.chat.schemas import ChatDone, ChatError, ChatSources
from src.doc_rag.config import settings
from src.doc_rag.db.repository import DBRepository
from src.doc_rag.documents.schemas import DocumentResponse
from src.doc_rag.documents.service import DocumentsService
from src.doc_rag.mcp_server.schemas import AskResponse
from src.doc_rag.retrieval.schemas import SearchResult
from src.doc_rag.retrieval.service import RetrievalService

mcp = MCPServer(
    "doc-rag",
    instructions=(
        "Search and ask questions about the user's own PDF documents (their research "
        "sources). Call list_documents to see what is available, search_documents to read "
        "the relevant passages yourself, or ask_documents to get a short answer with citations."
    ),
)

ToolContext = Context[None, Request]


def _state(ctx: ToolContext) -> State:
    if ctx.request_context.request is None:
        raise ToolError("This server only supports calls over HTTP.")
    return ctx.request_context.request.state


@mcp.tool()
async def list_documents() -> list[DocumentResponse]:
    """List the uploaded documents with their ids and processing status.

    Only documents with status "ready" can be searched. Use the ids to restrict
    search_documents or ask_documents to specific documents.
    """
    async with DBRepository.open() as db:
        return [
            DocumentResponse.model_validate(document)
            for document in await DocumentsService(db).list_all()
        ]


@mcp.tool()
async def search_documents(
    query: Annotated[str, Field(min_length=1, max_length=1000)],
    ctx: ToolContext,
    top_k: Annotated[int, Field(ge=1, le=settings.RERANK_CANDIDATES)] = settings.SEARCH_TOP_K,
    document_ids: Annotated[list[uuid.UUID] | None, Field(min_length=1)] = None,
) -> list[SearchResult]:
    """Find the passages of the documents that best match a question or a few keywords.

    Returns passages (text, tables as Markdown, formulas as LaTeX, figure captions) best
    first, each with its file name, page range, section heading and a relevance score from
    0 to 1. Scores under about 0.05 are noise. Use this to read the sources yourself and
    to quote or cite them with their pages.
    """
    state = _state(ctx)
    async with DBRepository.open() as db:
        return await RetrievalService(db, state.embedder, state.reranker).search(
            query, top_k, document_ids
        )


@mcp.tool()
async def ask_documents(
    question: Annotated[str, Field(min_length=1, max_length=2000)],
    ctx: ToolContext,
    document_ids: Annotated[list[uuid.UUID] | None, Field(min_length=1)] = None,
) -> AskResponse:
    """Answer a question using only the documents, with numbered citations.

    The answer cites its sources as [1], [2]...; `sources` lists those passages with
    their file names and pages. If the documents do not contain the answer, the answer says so.
    Prefer search_documents when you need the exact wording of the sources.
    """
    state = _state(ctx)
    sources = []
    async with DBRepository.open() as db:
        pipeline = ChatPipeline(RetrievalService(db, state.embedder, state.reranker), state.llm)
        async for event in pipeline.run(question, settings.SEARCH_TOP_K, document_ids):
            match event:
                case ChatSources():
                    sources = event.sources
                case ChatDone():
                    return AskResponse(
                        answer=event.answer, citations=event.citations, sources=sources
                    )
                case ChatError():
                    raise ToolError(event.message)
    raise ToolError("The answer ended without a result.")

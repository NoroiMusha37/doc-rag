from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent

from src.doc_rag.chat.pipeline import ChatPipeline
from src.doc_rag.chat.schemas import ChatRequest
from src.doc_rag.dependencies import get_chat_pipeline

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_class=EventSourceResponse)
async def chat(
    body: ChatRequest, pipeline: ChatPipeline = Depends(get_chat_pipeline)
) -> AsyncIterator[ServerSentEvent]:
    """Answer a question from the documents, streamed as Server-Sent Events.

    Events, in order: `sources` (the numbered chunks the answer may cite), `token`
    (repeated; the answer piece by piece), then `done` (the full answer and the source
    numbers it cites) or `error`.
    """
    async for event in pipeline.run(body.question, body.top_k, body.document_ids):
        yield ServerSentEvent(event=event.event, data=event)

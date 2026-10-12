from pydantic import BaseModel

from src.doc_rag.chat.schemas import Source


class AskResponse(BaseModel):
    answer: str
    citations: list[int]
    sources: list[Source]

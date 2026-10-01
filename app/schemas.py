import uuid
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict


class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    filetype: str
    parsing_status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5)
    documents_id: list[uuid.UUID] | None = Field(default=None)


class ChunkResponse(BaseModel):
    id: uuid.UUID
    page_number: int
    text: str
    document_id: uuid.UUID
    similarity_score: float

    model_config = ConfigDict(from_attributes=True)


class SearchResponse(BaseModel):
    answer: str
    sources: list[ChunkResponse]

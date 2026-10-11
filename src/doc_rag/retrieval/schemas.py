import uuid

from pydantic import BaseModel, Field

from src.doc_rag.config import settings
from src.doc_rag.db.models import ChunkKind


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=settings.SEARCH_TOP_K, ge=1, le=settings.RERANK_CANDIDATES)
    # Search only these documents; omit to search all of them.
    document_ids: list[uuid.UUID] | None = Field(default=None, min_length=1)


class SearchResult(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    kind: ChunkKind
    text: str
    heading_path: str
    page_start: int | None
    page_end: int | None
    # Figure image path relative to the storage folder, for figure chunks.
    asset_path: str | None
    # Reranker relevance between 0 and 1; results are sorted by it, best first.
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]

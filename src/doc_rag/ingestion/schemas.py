from pydantic import BaseModel

from src.doc_rag.db.models import ChunkKind


class ParsedChunk(BaseModel):
    ordinal: int
    kind: ChunkKind
    text: str
    embed_text: str
    heading_path: str
    page_start: int | None
    page_end: int | None
    asset_path: str | None = None


class ParsedDocument(BaseModel):
    page_count: int
    chunks: list[ParsedChunk]

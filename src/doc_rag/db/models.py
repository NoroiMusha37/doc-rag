import uuid
from datetime import datetime
from enum import StrEnum

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 1024


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PARSING = "parsing"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"


class ChunkKind(StrEnum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    FORMULA = "formula"


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    status: Mapped[str] = mapped_column(String(16), default=DocumentStatus.PENDING)
    error: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index(
            "ix_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        Index("ix_chunks_document_id", "document_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    ordinal: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(16), default=ChunkKind.TEXT)
    text: Mapped[str] = mapped_column(Text)
    embed_text: Mapped[str] = mapped_column(Text)
    heading_path: Mapped[str] = mapped_column(Text, default="")
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    asset_path: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )

    document: Mapped[Document] = relationship(back_populates="chunks")

import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import Select, cast, delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import REGCONFIG
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer, joinedload

from src.doc_rag.db.models import TEXT_SEARCH_CONFIG, Chunk, Document, DocumentStatus
from src.doc_rag.db.session import SessionLocal

UNFINISHED_STATUSES = (DocumentStatus.PENDING, DocumentStatus.PARSING, DocumentStatus.EMBEDDING)


class DocumentsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, document_id: uuid.UUID) -> Document | None:
        return await self._session.scalar(select(Document).where(Document.id == document_id))

    async def get_by_sha256(self, sha256: str) -> Document | None:
        return await self._session.scalar(select(Document).where(Document.sha256 == sha256))

    async def list_all(self) -> Sequence[Document]:
        return (
            await self._session.scalars(select(Document).order_by(Document.created_at.desc()))
        ).all()

    async def create(self, **values: Any) -> Document:
        return (
            await self._session.scalars(insert(Document).values(**values).returning(Document))
        ).one()

    async def update(self, document_id: uuid.UUID, **values: Any) -> Document | None:
        return await self._session.scalar(
            update(Document).where(Document.id == document_id).values(**values).returning(Document)
        )

    async def delete(self, document_id: uuid.UUID) -> bool:
        return (
            await self._session.scalar(
                delete(Document).where(Document.id == document_id).returning(Document.id)
            )
            is not None
        )

    async def fail_unfinished(self, error: str) -> int:
        """Mark documents left mid-ingestion as failed; return how many there were."""
        return len(
            (
                await self._session.scalars(
                    update(Document)
                    .where(Document.status.in_(UNFINISHED_STATUSES))
                    .values(status=DocumentStatus.FAILED, error=error)
                    .returning(Document.id)
                )
            ).all()
        )


class ChunksRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace(self, document_id: uuid.UUID, rows: list[dict[str, Any]]) -> None:
        await self._session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        if rows:
            await self._session.execute(
                insert(Chunk), [{**row, "document_id": document_id} for row in rows]
            )

    async def search_dense(
        self, embedding: list[float], limit: int, document_ids: Sequence[uuid.UUID] | None = None
    ) -> Sequence[Chunk]:
        # The HNSW index is scanned before the document filter is applied; iterative scan
        # keeps scanning until `limit` rows pass the filter, so filtering never starves it.
        await self._session.execute(
            select(func.set_config("hnsw.iterative_scan", "strict_order", True))
        )
        return (
            await self._session.scalars(
                self._search_statement(document_ids)
                .order_by(Chunk.embedding.cosine_distance(embedding))
                .limit(limit)
            )
        ).all()

    async def search_lexical(
        self, query: str, limit: int, document_ids: Sequence[uuid.UUID] | None = None
    ) -> Sequence[Chunk]:
        """The `limit` chunks matching the words of `query`, best first.

        `query` uses web-search syntax: words are ANDed, "quoted phrases" and `-excluded`
        words are supported. A query made only of stop words matches nothing.
        """
        tsquery = func.websearch_to_tsquery(cast(TEXT_SEARCH_CONFIG, REGCONFIG), query)
        return (
            await self._session.scalars(
                self._search_statement(document_ids)
                .where(Chunk.tsv.op("@@")(tsquery))
                .order_by(func.ts_rank_cd(Chunk.tsv, tsquery).desc())
                .limit(limit)
            )
        ).all()

    @staticmethod
    def _search_statement(document_ids: Sequence[uuid.UUID] | None) -> Select[Chunk]:
        """Chunks with their document loaded; the large vector columns are never read."""
        statement = select(Chunk).options(
            joinedload(Chunk.document), defer(Chunk.embedding), defer(Chunk.tsv)
        )
        if document_ids is not None:
            statement = statement.where(Chunk.document_id.in_(document_ids))
        return statement


class DBRepository:
    """Single entry point to the database: one session, one transaction.

    Nothing is persisted until `commit()` is called, so several operations
    succeed or fail together. Errors are not caught here; callers decide what
    a failure means, and an uncommitted session is rolled back when it closes.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self.documents = DocumentsRepository(session)
        self.chunks = ChunksRepository(session)

    @classmethod
    @asynccontextmanager
    async def open(cls) -> AsyncIterator["DBRepository"]:
        async with SessionLocal() as session:
            yield cls(session)

    async def commit(self) -> None:
        await self._session.commit()

    async def rollback(self) -> None:
        await self._session.rollback()

import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.doc_rag.db.models import Chunk, Document, DocumentStatus
from src.doc_rag.db.session import SessionLocal

UNFINISHED_STATUSES = (DocumentStatus.PENDING, DocumentStatus.PARSING, DocumentStatus.EMBEDDING)


class DocumentsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, document_id: uuid.UUID) -> Document | None:
        statement = select(Document).where(Document.id == document_id)
        return await self._session.scalar(statement)

    async def get_by_sha256(self, sha256: str) -> Document | None:
        statement = select(Document).where(Document.sha256 == sha256)
        return await self._session.scalar(statement)

    async def list_all(self) -> Sequence[Document]:
        statement = select(Document).order_by(Document.created_at.desc())
        return (await self._session.scalars(statement)).all()

    async def create(self, **values: Any) -> Document:
        statement = insert(Document).values(**values).returning(Document)
        return await self._session.scalar(statement)

    async def update(self, document_id: uuid.UUID, **values: Any) -> Document | None:
        statement = (
            update(Document).where(Document.id == document_id).values(**values).returning(Document)
        )
        return await self._session.scalar(statement)

    async def delete(self, document_id: uuid.UUID) -> bool:
        statement = delete(Document).where(Document.id == document_id).returning(Document.id)
        return await self._session.scalar(statement) is not None

    async def fail_unfinished(self, error: str) -> int:
        """Mark documents left mid-ingestion as failed; return how many there were."""
        statement = (
            update(Document)
            .where(Document.status.in_(UNFINISHED_STATUSES))
            .values(status=DocumentStatus.FAILED, error=error)
        )
        return (await self._session.execute(statement)).rowcount


class ChunksRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace(self, document_id: uuid.UUID, rows: list[dict[str, Any]]) -> None:
        await self._session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        if rows:
            await self._session.execute(
                insert(Chunk), [{**row, "document_id": document_id} for row in rows]
            )


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

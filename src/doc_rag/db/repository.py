import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.doc_rag.db.models import Chunk, Document
from src.doc_rag.db.session import SessionLocal


class DocumentsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, document_id: uuid.UUID) -> Document | None:
        statement = select(Document).where(Document.id == document_id)
        return await self._session.scalar(statement)

    async def update(self, document_id: uuid.UUID, **values: Any) -> Document | None:
        """Set the given columns; return the updated document, or None if it does not exist."""
        statement = (
            update(Document).where(Document.id == document_id).values(**values).returning(Document)
        )
        return await self._session.scalar(statement)


class ChunksRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace(self, document_id: uuid.UUID, rows: list[dict[str, Any]]) -> None:
        """Swap all chunks of a document, so re-ingesting never leaves duplicates.

        Each row holds the chunk's column values except `document_id`.
        """
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
        """Open a repository on its own session, for code that runs outside a request
        (background ingestion). Request handlers build one from the `get_session` dependency.
        """
        async with SessionLocal() as session:
            yield cls(session)

    async def commit(self) -> None:
        await self._session.commit()

    async def rollback(self) -> None:
        await self._session.rollback()

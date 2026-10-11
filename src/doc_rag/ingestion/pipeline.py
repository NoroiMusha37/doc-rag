import asyncio
import logging
import uuid
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from src.doc_rag.config import settings
from src.doc_rag.db.models import DocumentStatus
from src.doc_rag.db.repository import DBRepository
from src.doc_rag.embedding.embedder import Embedder
from src.doc_rag.ingestion.chunker import DocumentChunker
from src.doc_rag.ingestion.parser import DocumentParser
from src.doc_rag.ingestion.schemas import ParsedDocument

logger = logging.getLogger(__name__)


def upload_path(document_id: uuid.UUID, filename: str) -> Path:
    return Path(settings.STORAGE_DIR) / "uploads" / f"{document_id}{Path(filename).suffix.lower()}"


def assets_dir(document_id: uuid.UUID) -> Path:
    return Path(settings.STORAGE_DIR) / "assets" / str(document_id)


class IngestionPipeline:
    """Turns a stored upload into embedded chunks, recording progress on the document row."""

    def __init__(self, parser: DocumentParser, chunker: DocumentChunker, embedder: Embedder):
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        # One document at a time: parsing and embedding share 8 GB of VRAM.
        self._lock = asyncio.Lock()

    async def ingest(self, document_id: uuid.UUID) -> None:
        """Never raises: it runs as a background task, so failures go to the row and the log."""
        async with self._lock:
            try:
                await self._run(document_id)
            except Exception as exc:
                logger.exception(f"Ingestion of document {document_id} failed")
                await self._mark_failed(document_id, exc)

    @staticmethod
    async def fail_interrupted() -> None:
        """At startup: documents left mid-ingestion by a restart would otherwise stay stuck."""
        try:
            async with DBRepository.open() as db:
                count = await db.documents.fail_unfinished(
                    "Ingestion was interrupted by a restart; upload the file again to retry"
                )
                await db.commit()
        except SQLAlchemyError as exc:
            # The app must still start when the database is down or not migrated yet.
            logger.warning(f"Could not check for interrupted documents: {exc}")
            return
        if count:
            logger.warning(f"Marked {count} interrupted document(s) as failed")

    async def _run(self, document_id: uuid.UUID) -> None:
        filename = await self._set_status(document_id, DocumentStatus.PARSING, error=None)
        if filename is None:
            logger.info(f"Document {document_id} was deleted before ingestion started")
            return
        logger.info(f"Ingesting document {document_id} ({filename})")

        # Model inference is blocking; run it in a thread to keep the API responsive.
        parsed = await asyncio.to_thread(self._parse, document_id, filename)
        if not parsed.chunks:
            raise ValueError("No text could be extracted from the document")

        await self._set_status(document_id, DocumentStatus.EMBEDDING, page_count=parsed.page_count)
        vectors = await asyncio.to_thread(
            self._embedder.embed, [chunk.embed_text for chunk in parsed.chunks]
        )

        if await self._store(document_id, parsed, vectors):
            logger.info(f"Document {document_id} is ready: {len(parsed.chunks)} chunks")
        else:
            logger.info(f"Document {document_id} was deleted while it was being processed")

    def _parse(self, document_id: uuid.UUID, filename: str) -> ParsedDocument:
        document = self._parser.parse(upload_path(document_id, filename))
        return ParsedDocument(
            page_count=len(document.pages),
            chunks=self._chunker.chunk(document, assets_dir(document_id)),
        )

    @staticmethod
    async def _store(
        document_id: uuid.UUID, parsed: ParsedDocument, vectors: list[list[float]]
    ) -> bool:
        """Save the chunks and mark the document ready in one transaction."""
        async with DBRepository.open() as db:
            if await db.documents.update(document_id, status=DocumentStatus.READY) is None:
                return False
            await db.chunks.replace(
                document_id,
                [
                    {**chunk.model_dump(), "embedding": vector}
                    for chunk, vector in zip(parsed.chunks, vectors, strict=True)
                ],
            )
            await db.commit()
        return True

    @staticmethod
    async def _set_status(
        document_id: uuid.UUID, status: DocumentStatus, **values: object
    ) -> str | None:
        """Update the document; return its filename, or None if it no longer exists."""
        async with DBRepository.open() as db:
            document = await db.documents.update(document_id, status=status, **values)
            await db.commit()
        return document.filename if document else None

    async def _mark_failed(self, document_id: uuid.UUID, error: Exception) -> None:
        try:
            await self._set_status(document_id, DocumentStatus.FAILED, error=str(error))
        except Exception:
            # The database itself may be the problem; the log is all that is left.
            logger.exception(f"Could not record the failure of document {document_id}")

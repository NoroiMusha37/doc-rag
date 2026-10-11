import asyncio
import hashlib
import logging
import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.exc import IntegrityError

from src.doc_rag.config import settings
from src.doc_rag.db.models import Document, DocumentStatus
from src.doc_rag.db.repository import DBRepository
from src.doc_rag.ingestion.pipeline import assets_dir, upload_path

logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".pdf"}
PDF_SIGNATURE = b"%PDF-"
READ_SIZE = 1024 * 1024


class DocumentsService:
    """Upload, list and delete documents: the database rows and the files on disk."""

    def __init__(self, db: DBRepository) -> None:
        self._db = db

    async def upload(self, file: UploadFile) -> tuple[Document, bool]:
        """Store an upload; return the document and whether it must be ingested.

        A file that was uploaded before is not stored again: the existing document
        is returned, and it is ingested again only if its last attempt failed.
        """
        filename = Path(file.filename or "").name
        if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PDF files are supported"
            )

        document_id = uuid.uuid4()
        path = upload_path(document_id, filename)
        # Hashing and writing are blocking disk work; keep them off the event loop.
        sha256 = await asyncio.to_thread(self._save, file.file, path)

        existing = await self._db.documents.get_by_sha256(sha256)
        if existing is not None:
            path.unlink(missing_ok=True)
            return await self._reuse(existing)

        try:
            document = await self._db.documents.create(
                id=document_id, filename=filename, sha256=sha256
            )
            await self._db.commit()
        except IntegrityError as exc:
            # The same file was uploaded twice at the same moment; the other request won.
            await self._db.rollback()
            path.unlink(missing_ok=True)
            raise HTTPException(
                status.HTTP_409_CONFLICT, "This file is already being uploaded"
            ) from exc
        except Exception:
            path.unlink(missing_ok=True)
            raise
        logger.info(f"Stored document {document.id} ({filename})")
        return document, True

    async def list_all(self) -> Sequence[Document]:
        return await self._db.documents.list_all()

    async def get(self, document_id: uuid.UUID) -> Document:
        document = await self._db.documents.get(document_id)
        if document is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
        return document

    async def delete(self, document_id: uuid.UUID) -> None:
        document = await self.get(document_id)
        await self._db.documents.delete(document_id)
        await self._db.commit()
        # Remove the files only after the row is gone, so a failed delete loses nothing.
        await asyncio.to_thread(self._remove_files, document_id, document.filename)
        logger.info(f"Deleted document {document_id} ({document.filename})")

    async def _reuse(self, existing: Document) -> tuple[Document, bool]:
        if existing.status != DocumentStatus.FAILED:
            logger.info(f"Upload matches existing document {existing.id}; nothing to do")
            return existing, False
        document = await self._db.documents.update(
            existing.id, status=DocumentStatus.PENDING, error=None
        )
        if document is None:
            # Deleted by another request between the lookup and this update.
            raise HTTPException(
                status.HTTP_409_CONFLICT, "The document was just deleted; upload the file again"
            )
        await self._db.commit()
        logger.info(f"Upload matches failed document {existing.id}; retrying ingestion")
        return document, True

    @staticmethod
    def _save(source: BinaryIO, path: Path) -> str:
        """Write the upload to `path` and return its SHA-256; the file is removed on error."""
        limit = settings.MAX_UPLOAD_MB * 1024 * 1024
        digest = hashlib.sha256()
        size = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("wb") as target:
                while block := source.read(READ_SIZE):
                    if size == 0 and not block.startswith(PDF_SIGNATURE):
                        raise HTTPException(
                            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "The file is not a valid PDF"
                        )
                    size += len(block)
                    if size > limit:
                        raise HTTPException(
                            status.HTTP_413_CONTENT_TOO_LARGE,
                            f"The file is larger than {settings.MAX_UPLOAD_MB} MB",
                        )
                    digest.update(block)
                    target.write(block)
            if size == 0:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "The file is empty")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return digest.hexdigest()

    @staticmethod
    def _remove_files(document_id: uuid.UUID, filename: str) -> None:
        upload_path(document_id, filename).unlink(missing_ok=True)
        shutil.rmtree(assets_dir(document_id), ignore_errors=True)

import uuid

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.logger import log
from app.models import Document, StatusEnum


class DocumentsRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
            self, document_id: uuid.UUID, filename: str, filetype: str,
            file_path: str, file_metadata: dict | None = None
    ) -> Document:
        document = Document(
            id=document_id, filename=filename, filetype=filetype,
            file_path=file_path, file_metadata=file_metadata,
            parsing_status=StatusEnum.PENDING
        )

        self.session.add(document)
        log.info("Saving document...")
        try:
            await self.session.commit()
            await self.session.refresh(document)
            return document
        except SQLAlchemyError as e:
            log.error("DB query failed while trying to save document", error=e)
            await self.session.rollback()
            raise


class DBRepository:
    def __init__(self, session: AsyncSession):
        self.documents = DocumentsRepository(session)

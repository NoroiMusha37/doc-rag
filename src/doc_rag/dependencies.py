from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.doc_rag.db.repository import DBRepository
from src.doc_rag.db.session import get_db
from src.doc_rag.documents.service import DocumentsService
from src.doc_rag.ingestion.pipeline import IngestionPipeline


def get_db_repo(session: AsyncSession = Depends(get_db)) -> DBRepository:
    return DBRepository(session)


def get_documents_service(db: DBRepository = Depends(get_db_repo)) -> DocumentsService:
    return DocumentsService(db)


def get_pipeline(request: Request) -> IngestionPipeline:
    return request.state.pipeline

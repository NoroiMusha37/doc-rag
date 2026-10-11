import uuid
from collections.abc import Sequence

from fastapi import APIRouter, BackgroundTasks, Depends, Response, UploadFile, status

from src.doc_rag.db.models import Document
from src.doc_rag.dependencies import get_documents_service, get_pipeline
from src.doc_rag.documents.schemas import DocumentResponse
from src.doc_rag.documents.service import DocumentsService
from src.doc_rag.ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    response: Response,
    service: DocumentsService = Depends(get_documents_service),
    pipeline: IngestionPipeline = Depends(get_pipeline),
) -> Document:
    """Store a PDF and start ingesting it in the background.

    Returns 202 while ingestion is pending; poll `GET /documents/{id}` for the status.
    Returns 200 with the existing document when the same file was uploaded before.
    """
    document, needs_ingestion = await service.upload(file)
    if needs_ingestion:
        background_tasks.add_task(pipeline.ingest, document.id)
    else:
        response.status_code = status.HTTP_200_OK
    return document


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    service: DocumentsService = Depends(get_documents_service),
) -> Sequence[Document]:
    return await service.list_all()


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID, service: DocumentsService = Depends(get_documents_service)
) -> Document:
    return await service.get(document_id)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID, service: DocumentsService = Depends(get_documents_service)
) -> None:
    await service.delete(document_id)

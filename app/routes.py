import uuid

from fastapi import APIRouter, status, UploadFile, File

from app.schemas import DocumentResponse, SearchResponse, SearchRequest

router = APIRouter(prefix="/v1/documents", tags=["Documents"])


@router.post(
    "/",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED
)
async def upload_document(document: UploadFile = File(...)):
    ...
    # save
    # chunk
    # change status


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK
)
async def get_document(document_id: uuid.UUID):
    ...
    # return document info


@router.get(
    "/",
    response_model=list[DocumentResponse],
    status_code=status.HTTP_200_OK
)
async def get_documents():
    ...
    # fetch and return all the documents


@router.delete(
    "/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_document(document_id: uuid.UUID):
    ...
    # delete document


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK
)
async def search(data: SearchRequest):
    ...
    # embed the query
    # retrieve top k chunks
    # generate answer

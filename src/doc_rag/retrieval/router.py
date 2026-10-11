from fastapi import APIRouter, Depends

from src.doc_rag.dependencies import get_retrieval_service
from src.doc_rag.retrieval.schemas import SearchRequest, SearchResponse
from src.doc_rag.retrieval.service import RetrievalService

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    body: SearchRequest, service: RetrievalService = Depends(get_retrieval_service)
) -> SearchResponse:
    """Find the chunks that best match a question, with their pages and relevance scores."""
    results = await service.search(body.query, body.top_k, body.document_ids)
    return SearchResponse(query=body.query, results=results)

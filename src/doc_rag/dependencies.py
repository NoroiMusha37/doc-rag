from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.doc_rag.chat.pipeline import ChatPipeline
from src.doc_rag.db.repository import DBRepository
from src.doc_rag.db.session import get_db
from src.doc_rag.documents.service import DocumentsService
from src.doc_rag.embedding.embedder import Embedder
from src.doc_rag.embedding.reranker import Reranker
from src.doc_rag.ingestion.pipeline import IngestionPipeline
from src.doc_rag.llm.base import LLMClient
from src.doc_rag.retrieval.service import RetrievalService


def get_db_repo(session: AsyncSession = Depends(get_db)) -> DBRepository:
    return DBRepository(session)


def get_documents_service(db: DBRepository = Depends(get_db_repo)) -> DocumentsService:
    return DocumentsService(db)


def get_pipeline(request: Request) -> IngestionPipeline:
    return request.state.pipeline


def get_embedder(request: Request) -> Embedder:
    return request.state.embedder


def get_reranker(request: Request) -> Reranker:
    return request.state.reranker


def get_retrieval_service(
    db: DBRepository = Depends(get_db_repo),
    embedder: Embedder = Depends(get_embedder),
    reranker: Reranker = Depends(get_reranker),
) -> RetrievalService:
    return RetrievalService(db, embedder, reranker)


def get_llm(request: Request) -> LLMClient:
    return request.state.llm


def get_chat_pipeline(
    retrieval: RetrievalService = Depends(get_retrieval_service),
    llm: LLMClient = Depends(get_llm),
) -> ChatPipeline:
    return ChatPipeline(retrieval, llm)

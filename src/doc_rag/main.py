import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import torch
from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.doc_rag.chat.router import router as chat_router
from src.doc_rag.db.session import engine, get_db
from src.doc_rag.documents.router import router as documents_router
from src.doc_rag.embedding.embedder import Embedder
from src.doc_rag.embedding.reranker import Reranker
from src.doc_rag.ingestion.chunker import DocumentChunker
from src.doc_rag.ingestion.parser import DocumentParser
from src.doc_rag.ingestion.pipeline import IngestionPipeline
from src.doc_rag.llm.gemini import GeminiClient
from src.doc_rag.logger import setup_logging
from src.doc_rag.mcp_server.server import mcp
from src.doc_rag.retrieval.router import router as search_router
from src.doc_rag.schemas import HealthResponse

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[dict[str, object]]:
    logger.info("Loading models")
    # One embedder serves both ingestion and search, so the model sits in VRAM once.
    embedder = Embedder()
    pipeline = IngestionPipeline(DocumentParser(), DocumentChunker(), embedder)
    reranker = Reranker()
    llm = GeminiClient()
    await pipeline.fail_interrupted()
    logger.info("Application is ready")

    # The MCP session manager must run for as long as the application does.
    async with mcp.session_manager.run():
        yield {
            "pipeline": pipeline,
            "embedder": embedder,
            "reranker": reranker,
            "llm": llm,
        }

    await llm.close()
    await engine.dispose()


app = FastAPI(title="doc-rag", lifespan=lifespan)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(chat_router)
app.routes.extend(mcp.streamable_http_app(stateless_http=True, json_response=True).routes)


@app.get("/health", response_model=HealthResponse)
async def health(session: AsyncSession = Depends(get_db)) -> HealthResponse:
    try:
        await session.execute(text("SELECT 1"))
        database = True
    except (SQLAlchemyError, OSError) as e:
        logger.warning(f"Health check could not reach the database: {e}")
        database = False

    gpu = torch.cuda.is_available()
    return HealthResponse(
        status="ok" if database else "degraded",
        database=database,
        gpu=gpu,
        gpu_name=torch.cuda.get_device_name(0) if gpu else None,
    )

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import torch
from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.doc_rag.db.session import engine, get_session
from src.doc_rag.embedding.embedder import Embedder
from src.doc_rag.ingestion.chunker import DocumentChunker
from src.doc_rag.ingestion.parser import DocumentParser
from src.doc_rag.ingestion.pipeline import IngestionPipeline
from src.doc_rag.logger import setup_logging
from src.doc_rag.schemas import HealthResponse

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[dict[str, IngestionPipeline]]:
    logger.info("Loading ingestion models")
    pipeline = IngestionPipeline(DocumentParser(), DocumentChunker(), Embedder())
    logger.info("Application is ready")

    yield {
        "pipeline": pipeline,
    }

    await engine.dispose()


app = FastAPI(title="doc-rag", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health(session: AsyncSession = Depends(get_session)) -> HealthResponse:
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

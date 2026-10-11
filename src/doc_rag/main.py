from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import torch
from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.doc_rag.db.session import engine, get_session
from src.doc_rag.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await engine.dispose()


app = FastAPI(title="doc-rag", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health(session: AsyncSession = Depends(get_session)) -> HealthResponse:
    try:
        await session.execute(text("SELECT 1"))
        database = True
    except (SQLAlchemyError, OSError):
        database = False
    gpu = torch.cuda.is_available()
    return HealthResponse(
        status="ok" if database else "degraded",
        database=database,
        gpu=gpu,
        gpu_name=torch.cuda.get_device_name(0) if gpu else None,
    )

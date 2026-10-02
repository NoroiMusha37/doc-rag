from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import routes
from app.logger import log
from app.parser import DocumentParser


@asynccontextmanager
async def lifespan(app: FastAPI):
    doc_parser = DocumentParser()

    yield {
        "doc_parser": doc_parser,
    }


app = FastAPI(lifespan=lifespan)
log.info("Starting the application...")

app.include_router(routes.router)


@app.get("/health")
async def health_check():
    log.info("Health check has been called")
    return {
        "status": "ok"
    }

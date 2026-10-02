import uuid

from fastapi import FastAPI

from app import routes
from app.logger import log

app = FastAPI()
log.info("Starting the application...")

app.include_router(routes.router)


@app.get("/health")
async def health_check():
    log.info("Health check has been called")
    return {
        "status": "ok"
    }

from fastapi import FastAPI

from app.logger import log

app = FastAPI()
log.info("Starting the application...")


@app.get("/health")
async def health_check():
    log.info("Health check has been called")
    return {
        "status": "ok"
    }

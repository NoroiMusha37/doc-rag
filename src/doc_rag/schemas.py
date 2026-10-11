from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    database: bool
    gpu: bool
    gpu_name: str | None = None

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from src.doc_rag.db.models import DocumentStatus


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    status: DocumentStatus
    error: str | None
    page_count: int | None
    created_at: datetime

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from httpx import AsyncClient

from src.doc_rag.db.models import DocumentStatus
from src.doc_rag.dependencies import get_documents_service, get_pipeline
from src.doc_rag.main import app

PDF = ("paper.pdf", b"%PDF-1.7\ncontent", "application/pdf")


def _document(status: DocumentStatus = DocumentStatus.PENDING) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        filename="paper.pdf",
        status=status,
        error=None,
        page_count=None,
        created_at=datetime.now(UTC),
    )


class FakeService:
    def __init__(self) -> None:
        self.document = _document()
        self.needs_ingestion = True
        self.deleted: list[uuid.UUID] = []

    async def upload(self, file: object) -> tuple[SimpleNamespace, bool]:
        return self.document, self.needs_ingestion

    async def list_all(self) -> list[SimpleNamespace]:
        return [self.document]

    async def get(self, document_id: uuid.UUID) -> SimpleNamespace:
        if document_id != self.document.id:
            raise HTTPException(404, "Document not found")
        return self.document

    async def delete(self, document_id: uuid.UUID) -> None:
        await self.get(document_id)
        self.deleted.append(document_id)


class FakePipeline:
    def __init__(self) -> None:
        self.ingested: list[uuid.UUID] = []

    async def ingest(self, document_id: uuid.UUID) -> None:
        self.ingested.append(document_id)


@pytest.fixture
def service() -> Iterator[FakeService]:
    fake = FakeService()
    app.dependency_overrides[get_documents_service] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


@pytest.fixture
def pipeline(service: FakeService) -> FakePipeline:
    fake = FakePipeline()
    app.dependency_overrides[get_pipeline] = lambda: fake
    return fake


async def test_upload_returns_202_and_starts_ingestion(
    client: AsyncClient, service: FakeService, pipeline: FakePipeline
) -> None:
    response = await client.post("/documents", files={"file": PDF})

    assert response.status_code == 202
    assert response.json()["id"] == str(service.document.id)
    assert response.json()["status"] == "pending"
    assert pipeline.ingested == [service.document.id]


async def test_upload_of_a_known_file_returns_200_without_ingesting(
    client: AsyncClient, service: FakeService, pipeline: FakePipeline
) -> None:
    service.document = _document(DocumentStatus.READY)
    service.needs_ingestion = False

    response = await client.post("/documents", files={"file": PDF})

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert pipeline.ingested == []


async def test_list_and_get_return_documents(client: AsyncClient, service: FakeService) -> None:
    listed = await client.get("/documents")
    fetched = await client.get(f"/documents/{service.document.id}")

    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [str(service.document.id)]
    assert fetched.json() == listed.json()[0]


async def test_delete_returns_204(client: AsyncClient, service: FakeService) -> None:
    response = await client.delete(f"/documents/{service.document.id}")

    assert response.status_code == 204
    assert service.deleted == [service.document.id]


@pytest.mark.parametrize("method", ["get", "delete"])
async def test_unknown_document_is_404(
    client: AsyncClient, service: FakeService, method: str
) -> None:
    response = await client.request(method, f"/documents/{uuid.uuid4()}")

    assert response.status_code == 404


async def test_malformed_id_is_422(client: AsyncClient, service: FakeService) -> None:
    response = await client.get("/documents/not-a-uuid")

    assert response.status_code == 422

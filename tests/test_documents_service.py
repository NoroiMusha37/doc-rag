import hashlib
import uuid
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile

from src.doc_rag.config import settings
from src.doc_rag.db.models import DocumentStatus
from src.doc_rag.documents.service import DocumentsService
from src.doc_rag.ingestion.pipeline import assets_dir, upload_path

PDF = b"%PDF-1.7\nsome content"


class FakeDocuments:
    def __init__(self) -> None:
        self.rows: dict[uuid.UUID, SimpleNamespace] = {}

    async def get(self, document_id: uuid.UUID) -> SimpleNamespace | None:
        return self.rows.get(document_id)

    async def get_by_sha256(self, sha256: str) -> SimpleNamespace | None:
        return next((row for row in self.rows.values() if row.sha256 == sha256), None)

    async def create(self, **values: object) -> SimpleNamespace:
        row = SimpleNamespace(status=DocumentStatus.PENDING, error=None, **values)
        self.rows[row.id] = row
        return row

    async def update(self, document_id: uuid.UUID, **values: object) -> SimpleNamespace:
        vars(self.rows[document_id]).update(values)
        return self.rows[document_id]

    async def delete(self, document_id: uuid.UUID) -> bool:
        return self.rows.pop(document_id, None) is not None


class FakeDB:
    def __init__(self) -> None:
        self.documents = FakeDocuments()

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass


def _upload(content: bytes = PDF, filename: str = "paper.pdf") -> UploadFile:
    return UploadFile(file=BytesIO(content), filename=filename)


@pytest.fixture
def storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(settings, "STORAGE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def db() -> FakeDB:
    return FakeDB()


@pytest.fixture
def service(db: FakeDB, storage: Path) -> DocumentsService:
    return DocumentsService(db)


async def test_upload_stores_the_file_and_a_pending_document(service: DocumentsService) -> None:
    document, needs_ingestion = await service.upload(_upload())

    assert needs_ingestion is True
    assert document.filename == "paper.pdf"
    assert document.sha256 == hashlib.sha256(PDF).hexdigest()
    assert upload_path(document.id, document.filename).read_bytes() == PDF


async def test_uploading_the_same_file_again_returns_the_existing_document(
    service: DocumentsService, storage: Path
) -> None:
    first, _ = await service.upload(_upload())

    second, needs_ingestion = await service.upload(_upload(filename="renamed.pdf"))

    assert second is first
    assert needs_ingestion is False
    assert len(list((storage / "uploads").iterdir())) == 1


async def test_uploading_a_failed_document_again_retries_it(service: DocumentsService) -> None:
    first, _ = await service.upload(_upload())
    first.status = DocumentStatus.FAILED
    first.error = "boom"

    second, needs_ingestion = await service.upload(_upload())

    assert needs_ingestion is True
    assert second.status is DocumentStatus.PENDING
    assert second.error is None


@pytest.mark.parametrize(
    ("content", "filename", "status_code"),
    [
        pytest.param(PDF, "notes.txt", 415, id="wrong-extension"),
        pytest.param(b"PK\x03\x04 not a pdf", "fake.pdf", 415, id="wrong-content"),
        pytest.param(b"", "empty.pdf", 400, id="empty"),
        pytest.param(PDF + b"x" * (1024 * 1024), "big.pdf", 413, id="too-large"),
    ],
)
async def test_upload_rejects_invalid_files_and_leaves_nothing_behind(
    service: DocumentsService,
    db: FakeDB,
    storage: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: bytes,
    filename: str,
    status_code: int,
) -> None:
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)

    with pytest.raises(HTTPException) as error:
        await service.upload(_upload(content, filename))

    assert error.value.status_code == status_code
    assert db.documents.rows == {}
    assert not list(storage.rglob("*.pdf"))


async def test_delete_removes_the_row_the_upload_and_the_figures(
    service: DocumentsService, db: FakeDB
) -> None:
    document, _ = await service.upload(_upload())
    figures = assets_dir(document.id)
    figures.mkdir(parents=True)
    (figures / "0.png").write_bytes(b"png")

    await service.delete(document.id)

    assert db.documents.rows == {}
    assert not upload_path(document.id, document.filename).exists()
    assert not figures.exists()


async def test_get_and_delete_report_a_missing_document(service: DocumentsService) -> None:
    for call in (service.get, service.delete):
        with pytest.raises(HTTPException) as error:
            await call(uuid.uuid4())
        assert error.value.status_code == 404

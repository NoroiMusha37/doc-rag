import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.doc_rag.db.models import ChunkKind, DocumentStatus
from src.doc_rag.db.repository import DBRepository
from src.doc_rag.ingestion.exceptions import ParsingError
from src.doc_rag.ingestion.pipeline import IngestionPipeline
from src.doc_rag.ingestion.schemas import ParsedChunk

DOCUMENT_ID = uuid.uuid4()


class FakeDocuments:
    def __init__(self, exists: bool) -> None:
        self.document = SimpleNamespace(filename="paper.pdf") if exists else None
        self.statuses: list[DocumentStatus] = []

    async def update(self, document_id: uuid.UUID, **values: object) -> SimpleNamespace | None:
        if self.document is None:
            return None
        self.statuses.append(values["status"])
        vars(self.document).update(values)
        return self.document


class FakeChunks:
    def __init__(self) -> None:
        self.rows: list[dict] | None = None

    async def replace(self, document_id: uuid.UUID, rows: list[dict]) -> None:
        self.rows = rows


class FakeDB:
    def __init__(self, exists: bool = True) -> None:
        self.documents = FakeDocuments(exists)
        self.chunks = FakeChunks()

    async def commit(self) -> None:
        pass


class FakeParser:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def parse(self, path: Path) -> SimpleNamespace:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(pages={1: None, 2: None})


class FakeChunker:
    def __init__(self, count: int = 2) -> None:
        self.count = count

    def chunk(self, document: object, assets_dir: Path) -> list[ParsedChunk]:
        return [
            ParsedChunk(
                ordinal=ordinal,
                kind=ChunkKind.TEXT,
                text=f"chunk {ordinal}",
                embed_text=f"Heading\nchunk {ordinal}",
                heading_path="Heading",
                page_start=1,
                page_end=1,
            )
            for ordinal in range(self.count)
        ]


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(index)] for index, _ in enumerate(texts)]


@pytest.fixture
def use_db(monkeypatch: pytest.MonkeyPatch):
    """Make the pipeline open the given fake instead of a real database session."""

    def _use(db: FakeDB) -> FakeDB:
        @asynccontextmanager
        async def fake_open():
            yield db

        monkeypatch.setattr(DBRepository, "open", fake_open)
        return db

    return _use


async def test_ingest_stores_embedded_chunks_and_marks_the_document_ready(use_db) -> None:
    db = use_db(FakeDB())

    await IngestionPipeline(FakeParser(), FakeChunker(), FakeEmbedder()).ingest(DOCUMENT_ID)

    assert db.documents.statuses == [
        DocumentStatus.PARSING,
        DocumentStatus.EMBEDDING,
        DocumentStatus.READY,
    ]
    assert db.documents.document.page_count == 2
    assert [row["text"] for row in db.chunks.rows] == ["chunk 0", "chunk 1"]
    assert [row["embedding"] for row in db.chunks.rows] == [[0.0], [1.0]]


@pytest.mark.parametrize(
    ("parser", "chunker", "error"),
    [
        pytest.param(
            FakeParser(ParsingError("Could not convert paper.pdf")),
            FakeChunker(),
            "Could not convert paper.pdf",
            id="parser-fails",
        ),
        pytest.param(FakeParser(), FakeChunker(count=0), "No text", id="no-chunks"),
    ],
)
async def test_ingest_records_the_failure_on_the_document(
    use_db, parser: FakeParser, chunker: FakeChunker, error: str
) -> None:
    db = use_db(FakeDB())

    await IngestionPipeline(parser, chunker, FakeEmbedder()).ingest(DOCUMENT_ID)

    assert db.documents.statuses[-1] is DocumentStatus.FAILED
    assert error in db.documents.document.error
    assert db.chunks.rows is None


async def test_ingest_skips_a_document_that_no_longer_exists(use_db) -> None:
    db = use_db(FakeDB(exists=False))
    parser = FakeParser()

    await IngestionPipeline(parser, FakeChunker(), FakeEmbedder()).ingest(DOCUMENT_ID)

    assert parser.calls == 0
    assert db.chunks.rows is None

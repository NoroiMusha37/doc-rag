import os
from collections.abc import AsyncIterator, Callable, Iterator

import pytest
from httpx import ASGITransport, AsyncClient

# Settings are validated at import time; tests never connect to these.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from src.doc_rag.db.session import get_session  # noqa: E402
from src.doc_rag.main import app  # noqa: E402


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        yield http


@pytest.fixture
def override_session() -> Iterator[Callable[[object], None]]:
    """Replace the database session dependency with a stand-in for one test."""

    def _override(session: object) -> None:
        app.dependency_overrides[get_session] = lambda: session

    yield _override
    app.dependency_overrides.clear()

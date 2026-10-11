from httpx import ASGITransport, AsyncClient

from src.doc_rag.db.session import get_session
from src.doc_rag.main import app


class _HealthySession:
    async def execute(self, statement: object) -> None:
        return None


class _BrokenSession:
    async def execute(self, statement: object) -> None:
        raise OSError("connection refused")


async def _get_health(session: object) -> dict:
    app.dependency_overrides[get_session] = lambda: session
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    return response.json()


async def test_health_ok_when_database_responds() -> None:
    body = await _get_health(_HealthySession())
    assert body["status"] == "ok"
    assert body["database"] is True
    assert isinstance(body["gpu"], bool)


async def test_health_degraded_when_database_is_down() -> None:
    body = await _get_health(_BrokenSession())
    assert body["status"] == "degraded"
    assert body["database"] is False

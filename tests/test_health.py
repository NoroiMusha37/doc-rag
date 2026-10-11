from collections.abc import Callable

import pytest
from httpx import AsyncClient
from sqlalchemy.exc import OperationalError


class FakeSession:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def execute(self, statement: object) -> None:
        if self.error is not None:
            raise self.error


@pytest.mark.parametrize(
    ("error", "status", "database"),
    [
        pytest.param(None, "ok", True, id="database-up"),
        pytest.param(OSError("connection refused"), "degraded", False, id="connection-refused"),
        pytest.param(
            OperationalError("SELECT 1", {}, Exception("server closed the connection")),
            "degraded",
            False,
            id="sqlalchemy-error",
        ),
    ],
)
async def test_health_reports_database_state(
    client: AsyncClient,
    override_session: Callable[[object], None],
    error: Exception | None,
    status: str,
    database: bool,
) -> None:
    override_session(FakeSession(error))

    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == status
    assert body["database"] is database


async def test_health_reports_gpu(
    client: AsyncClient,
    override_session: Callable[[object], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    override_session(FakeSession())

    body = (await client.get("/health")).json()

    assert body["gpu"] is False
    assert body["gpu_name"] is None

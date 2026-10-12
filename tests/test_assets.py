from pathlib import Path

import pytest
from httpx import AsyncClient

from src.doc_rag.main import app


@pytest.fixture
def assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the /assets route at a temporary folder instead of the real storage."""
    files = next(route for route in app.routes if getattr(route, "path", None) == "/assets").app
    monkeypatch.setattr(files, "directory", tmp_path)
    monkeypatch.setattr(files, "all_directories", [tmp_path])
    return tmp_path


async def test_a_figure_is_served_at_its_asset_path(client: AsyncClient, assets: Path) -> None:
    (assets / "7f3c").mkdir()
    (assets / "7f3c" / "8.png").write_bytes(b"\x89PNG fake image")

    response = await client.get("/assets/7f3c/8.png")

    assert response.status_code == 200
    assert response.content == b"\x89PNG fake image"
    assert response.headers["content-type"] == "image/png"


async def test_a_missing_figure_is_not_found(client: AsyncClient, assets: Path) -> None:
    assert (await client.get("/assets/7f3c/8.png")).status_code == 404


async def test_files_outside_the_assets_folder_are_not_served(
    client: AsyncClient, assets: Path
) -> None:
    (assets.parent / "secret.txt").write_text("not an asset")

    assert (await client.get("/assets/%2e%2e/secret.txt")).status_code == 404

# src/doc_rag

## Purpose

The FastAPI application: settings, database access, and (as phases land) ingestion, retrieval, LLM and chat modules.

## Ownership

- `main.py` - the `app` object, lifespan, and `GET /health`
- `config.py` - `Settings` and the module-level `settings` instance
- `schemas.py` - DTOs for routes defined in `main.py`
- `db/session.py` - module-level `engine`, `SessionLocal`, and the `get_session` dependency
- `db/models.py` - `Base`, `Document`, `Chunk`, and the `DocumentStatus` / `ChunkKind` string enums

## Local Contracts

- Imports use the full path from the repo root: `from src.doc_rag.db.session import get_session`. Everything (uvicorn, pytest, alembic) runs with the repo root as the working directory; the project is not installed as a package.
- `__init__.py` files stay empty.
- `Settings` fields are UPPER_CASE and match their environment variable names. Required values (`DATABASE_URL`, `GEMINI_API_KEY`) have no default, so importing `config.py` fails when one is missing or empty.
- `app` is defined at module level in `main.py`; there is no `create_app` factory.
- `engine` is created once at import and disposed in the lifespan; request handlers get sessions only through `get_session`.
- Primary keys are UUIDs generated in Python (`default=uuid.uuid4`), so an id is known before the row is inserted. Enum-like columns (`status`, `kind`) are plain `String` columns holding `StrEnum` values, not native Postgres enums.
- `Chunk.embedding` is `Vector(EMBEDDING_DIM)` (1024); `Chunk.tsv` is a generated column and is never written by the application.
- `GET /health` always returns 200; `status` is `ok` or `degraded` depending on the database check.

## Work Guidance

- Prefer the simplest structure that is still correct: module-level objects over lazy singletons and factories.
- DTOs go in the `schemas.py` of the module that owns the route.
- Never hardcode connection strings, credentials or keys; read them from `settings`.

## Verification

- `uv run ruff check` and `uv run ruff format --check`
- `uv run pytest`
- `docker compose up --build -d`, then `curl http://localhost:8000/health` returns `"status":"ok"` with `"gpu":true`

## Child DOX Index

None yet. `ingestion/`, `retrieval/`, `llm/` and `chat/` get their own AGENTS.md when created.
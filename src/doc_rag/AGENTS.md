# src/doc_rag

## Purpose

The FastAPI application: settings, database access, and (as phases land) ingestion, retrieval, LLM and chat modules.

## Ownership

- `main.py` - the `app` object, lifespan (loads the models and the LLM client once and yields `pipeline`, `embedder`, `reranker`, `llm`), and `GET /health`
- `config.py` - `Settings` and the module-level `settings` instance
- `logger.py` - `setup_logging()`, called once at the top of `main.py`
- `schemas.py` - DTOs for routes defined in `main.py`
- `dependencies.py` - FastAPI dependencies: `get_db_repo`, `get_documents_service`, `get_retrieval_service`, `get_chat_pipeline`, and `get_pipeline` / `get_embedder` / `get_reranker` / `get_llm` (lifespan objects)
- `db/session.py` - module-level `engine`, `SessionLocal`, and `get_db` (the per-request session dependency)
- `db/models.py` - `Base`, `Document`, `Chunk`, and the `DocumentStatus` / `ChunkKind` string enums
- `db/repository.py` - `DBRepository` with `documents` (`DocumentsRepository`) and `chunks` (`ChunksRepository`)
- `embedding/embedder.py` - `Embedder` (bge-m3, normalized 1024-dim vectors)
- `embedding/reranker.py` - `Reranker` (bge-reranker-v2-m3 cross-encoder, scores 0-1)

## Local Contracts

- Imports use the full path from the repo root: `from src.doc_rag.dependencies import get_db_repo`. Everything (uvicorn, pytest, alembic) runs with the repo root as the working directory; the project is not installed as a package.
- `__init__.py` files stay empty.
- `Settings` fields are UPPER_CASE and match their environment variable names. Required values (`DATABASE_URL`, `GEMINI_API_KEY`) have no default, so importing `config.py` fails when one is missing or empty.
- `app` is defined at module level in `main.py`; there is no `create_app` factory.
- `engine` is created once at import and disposed in the lifespan. Only `/health` uses a raw session (`get_db`); everything else goes through `DBRepository`.
- Primary keys are UUIDs generated in Python (`default=uuid.uuid4`), so an id is known before the row is inserted. Enum-like columns (`status`, `kind`) are plain `String` columns holding `StrEnum` values, not native Postgres enums.
- `Chunk.embedding` is `Vector(EMBEDDING_DIM)` (1024); `Chunk.tsv` is a generated column and is never written by the application.
- All queries live in `db/repository.py`; nothing else builds SQL or touches the session, except the `SELECT 1` in `/health`. Repositories use SQLAlchemy 2.0 statements (`select`, `update(...).returning(...)`, `delete`, bulk `insert` with a list of rows), do not commit, and do not catch errors; the caller calls `DBRepository.commit()` and decides what a failure means.
- Dependency chain: `get_db` (in `db/session.py`, yields the request's session) -> `get_db_repo` (builds `DBRepository(session)`) -> `get_<feature>_service` (builds the service from the repository). All except `get_db` are defined in `dependencies.py`, never in routers.
- Services are built per request from the request's repository; they are not singletons and never open sessions themselves.
- Only objects that are expensive or stateful across requests (the GPU models: `Embedder`, `Reranker`, the ingestion pipeline that owns the parser and chunker, and the `GeminiClient` with its HTTP connection pool) are built in the lifespan, yielded in its state dict, and returned from `request.state` by a dependency. One `Embedder` is shared by ingestion and search. Model wrappers hold a lock so only one thread uses a model at a time.
- `DBRepository.open()` is only for code that runs outside any request (the ingestion background task, startup), where no request session exists.
- Routers are registered in `main.py` with `app.include_router(...)`. Each feature folder has `router.py` (thin), `service.py` (logic, may raise `HTTPException`) and `schemas.py`; in `chat/` the logic file is `pipeline.py`, as named in `docs/plan.md`.
- Streaming routes use FastAPI's own SSE support: `response_class=EventSourceResponse` and `yield ServerSentEvent(event=..., data=...)` from the route (`fastapi.sse`). A streaming route cannot change its status code after the first event, so failures during the stream are sent as an `error` event.
- The language model is used only through `LLMClient` from `llm/base.py`; no module outside `llm/` imports `google.genai`.
- Objects built at startup are passed to handlers through the lifespan state (`yield {...}` -> `request.state`), not assigned to `app.state`. Startup work in the lifespan is plain synchronous code, no `asyncio.to_thread`.
- Logging: `logging.getLogger(__name__)` in each module, f-strings in log calls. `LOG_LEVEL` applies to `src.doc_rag` loggers; third-party libraries log at WARNING and above; uvicorn's lines go through the same handler.
- `GET /health` always returns 200; `status` is `ok` or `degraded` depending on the database check.

## Work Guidance

- Prefer the simplest structure that is still correct: module-level objects over lazy singletons and factories.
- DTOs go in the `schemas.py` of the module that owns the route.
- Never hardcode connection strings, credentials or keys; read them from `settings`.
- Components that hold models or state are classes; keep their structure flat and readable (a short public method, small private helpers).
- Use custom exceptions sparingly; built-in exceptions are the default.
- No single-use variables that only name a value; inline them. A variable earns its place by being used more than once, by fixing the moment something is computed, or by being mutated.

## Verification

- `uv run ruff check` and `uv run ruff format --check`
- `uv run pytest`
- `docker compose up --build -d`, then `curl http://localhost:8000/health` returns `"status":"ok"` with `"gpu":true`

## Child DOX Index

- `ingestion/AGENTS.md` - parser, chunker and pipeline contracts, storage paths, Docling chunk facts
- `documents/AGENTS.md` - `/documents` routes, upload validation and deduplication, delete behaviour
- `retrieval/AGENTS.md` - hybrid search flow, `/search`, keyword-search behaviour, scoring
- `llm/AGENTS.md` - `LLMClient` interface, Gemini client, retries, error contract, prompts
- `chat/AGENTS.md` - answering flow, `/chat` SSE events, declining, citations
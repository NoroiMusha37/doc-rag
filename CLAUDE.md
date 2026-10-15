# CLAUDE.md

This file provides guidance to agents when working with code in this repository.

Read `AGENTS.md` (DOX) and every child `AGENTS.md` on the path before editing. Roadmap and fixed decisions are in `docs/plan.md`.

## Project
RAG service over complex PDFs (tables, figures, formulas) that answers questions with page citations, to support a bachelor paper in financial mathematics.
FastAPI + Postgres/pgvector + Docker; local Docling, bge-m3 and bge-reranker on GPU; Gemini API free tier as the LLM.

## Commands
Run everything from the repo root:
- `uv sync` - install dependencies; `uv add <pkg>` to add one
- `docker compose up --build -d` - start Postgres, the API and the web page (needs `.env`, see `.env.example`)
- `uv run alembic revision --autogenerate -m "<msg>"` - generate a migration from the models
- `uv run alembic upgrade head` (or `docker compose exec api alembic upgrade head`) - apply migrations
- `uv run pytest` - tests; `uv run --group ui pytest` also runs the web page tests (skipped without Streamlit)
- `uv run ruff check` / `uv run ruff format` - lint and format
- `uv run python eval/run_eval.py` - evaluate a running stack against `eval/questions.jsonl` (retrieval only; `--answers` also asks `/chat` and spends Gemini requests)

## Architecture
Target layout under `src/doc_rag/` (built phase by phase):
- `main.py`, `config.py`, `schemas.py` - module-level `app` with `/health`, settings, health DTO
- `db/` - async session and SQLAlchemy models (`documents`, `chunks`)
- `documents/` - upload, list, status, delete
- `ingestion/` - Docling parsing, chunking, optional figure captioning
- `embedding/` - bge-m3 embedder and bge reranker
- `retrieval/` - dense + full-text search, RRF fusion, rerank
- `llm/` - `LLMClient` protocol with Gemini and Ollama implementations
- `chat/` - `pipeline.py` (hand-written) and `graph.py` (LangGraph)
- `mcp_server/` - MCP tools mounted in the FastAPI app
- `ui/` (repo root) - Streamlit page (`app.py`) and its API client (`client.py`), in their own container
- `eval/` (repo root) - hand-written question set and the script that scores retrieval and answers against the running API

## Facts
Every time you get new information about this project, or new way of interaction with 
library/models/services, or specific instructions from user on how to test/implement new feature,
add short description of acquired knowledge here in bulleted list:
- After work is done in worktrees - merge worktree branch into main locally (`git merge`); do NOT push worktree branches to remote; Delete worktree after successful merge;
- If needed DTO model - create pydantic schema in separate schemas.py file related to working module
- Never add anything to `__init__.py`. Use full-path imports.
- Imports start with `src.`: `from src.doc_rag.retrieval.service import ...`. The project is not installed as a package; run uvicorn, pytest and alembic from the repo root
- Models are local except the LLM; call the LLM only through `src.doc_rag.llm.base.LLMClient`
- `Settings` fields are UPPER_CASE; required ones (`DATABASE_URL`, `GEMINI_API_KEY`) have no default so the app fails at import when they are missing. Never hardcode URLs or credentials
- `main.py` defines `app` at module level - no `create_app` factory. Keep modules simple (module-level engine/sessionmaker) unless there is a concrete need for more
- Do not hand-write migrations; generate them with `alembic revision --autogenerate` (new files are auto-formatted by ruff hooks). Only `CREATE EXTENSION` is added by hand, in the first revision
- Ids of `Document` and `Chunk` are UUIDs (`uuid.uuid4` in Python); `status`/`kind` are `String` columns with `StrEnum` values
- Add dependencies with `uv add`, not by editing `pyproject.toml`. `uv add <pkg> --index NAME=URL` on an index that already exists in `pyproject.toml` re-registers it as a normal (non-explicit) index and breaks resolution (`certifi` is then looked up there), so it cannot be used to pin more packages to the cu128 index
- The API container runs `uvicorn src.doc_rag.main:app` directly and has a Dockerfile `HEALTHCHECK` polling `/health`; migrations are applied explicitly, not on container start
- In `compose.yaml` write `healthcheck.test` as a single string, not a list
- torch 2.11 cu128 wheels work on Python 3.13 in `python:3.13-slim` with GPU passthrough (Docker Desktop + NVIDIA runtime); no CUDA base image is needed. The API image is about 11 GB
- Postgres only applies `POSTGRES_USER`/`POSTGRES_PASSWORD` when the `pgdata` volume is first created; changing them later needs a new volume
- Feature folders follow `router.py` (thin) + `service.py` (class, raises `HTTPException`) + `schemas.py`
- Dependencies follow the user's usual pattern (as in `D:\Projects\url-shortener\app\dependencies.py`): `get_db` in `db/session.py` yields the session; `dependencies.py` has `get_db_repo(session=Depends(get_db)) -> DBRepository`, `get_documents_service(db=Depends(get_db_repo))`, and `get_pipeline(request)` for the lifespan object. No dependencies in routers. Tests override them, because the test client does not run the lifespan
- Services are built per request from `DBRepository`; do not make them singletons or put them in the lifespan (tried once and reverted: a shared service cannot hold a session, which forced it to open its own sessions). Only the GPU pipeline lives in the lifespan. `DBRepository.open()` is for background/startup code only
- Push back before implementing when the user proposes something that harms the design or correctness: say what breaks and why, offer the alternative, and only then proceed with what they decide
- Uploads: PDF only, deduplicated by SHA-256, `MAX_UPLOAD_MB` limit; files live under `STORAGE_DIR/uploads` and `STORAGE_DIR/assets` (paths from `upload_path()` / `assets_dir()` in `ingestion/pipeline.py`)
- Tests never need a GPU, network or database: fake the repository, models and service (see `tests/test_pipeline.py`, `tests/test_documents_*.py`). Real behaviour is checked against the running compose stack with curl and psql
- Database access only through `DBRepository` (`src/doc_rag/db/repository.py`) with its `documents` and `chunks` sub-repositories, written in SQLAlchemy 2.0 statement style. Repositories do not commit or catch errors; callers do
- SQLAlchemy is 2.1: annotate statements as `Select[Chunk]`, not the 2.0-style `Select[tuple[Chunk]]` (2.1 reads that as one tuple-valued column and `scalars().all()` becomes `Sequence[tuple[Chunk]]`). Use `(await session.scalars(stmt)).one()` when a row is guaranteed (`scalar()` is typed `T | None`), and count affected rows with `.returning(Model.id)` instead of `.rowcount`. Type-check with `uvx pyright <file> --pythonpath .venv/Scripts/python.exe`
- Do not create a variable that is used only once just to name a value (`limit = settings.X`, `elapsed = ...`, `statement = select(...)` followed by one use): inline it. Keep a variable only when it is used more than once, must be computed before something else happens (a timer start, a dict built once and indexed in a loop), or is mutated
- `Chunk.kind` / `Document.status` are `Mapped[str]`; convert with `ChunkKind(chunk.kind)` when a DTO field is typed as the enum
- Logs use f-strings (`logger.info(f"...")`) and `logging.getLogger(__name__)`; logging is configured in `src/doc_rag/logger.py`
- Components holding models or state are classes (`DocumentParser`, `DocumentChunker`, `Embedder`, `IngestionPipeline`); keep custom exceptions rare (`ParsingError` is the only one)
- Lifespan: build startup objects with plain calls (no `asyncio.to_thread`) and pass them via `yield {"pipeline": ..., "embedder": ..., "reranker": ...}` -> `request.state.<name>` through a dependency each; no `app.state` assignments and no extra state classes
- The Write tool's batch is cancelled as a whole if one file in it is rejected, and writing an empty file (`__init__.py`) is reported as a rejection: create empty files from the shell and send other files one at a time
- `torch` and `torchvision` must both come from the cu128 index (`[tool.uv.sources]` has a line for each); the PyPI `torchvision` fails in the container with `operator torchvision::nms does not exist`. The user approved hand-editing that sources line; a package must also be a direct dependency (`uv add torchvision`) for its source to apply
- Measured on the RTX 4060 (15-page paper with formulas): ingestion ~79s, peak VRAM 2.9 GB, models at rest 2.2 GB; API startup ~50s with cached models (bge-m3 load is the slow part)
- RapidOCR (Docling's OCR) downloads its models into `site-packages` on the first parse, not into the `models` volume, so every new container re-downloads ~30 MB; its INFO lines bypass our logging config
- Docling formula enrichment returns LaTeX with spaced letters (`A t t e n t i o n ( Q , K , V )`); valid for display but weak for keyword search
- Docling 2.137: chunk `doc_items` are generic `DocItem`s (match by `self_ref`); pictures reach chunks only through their captions; HybridChunker merges items under one heading; tables need `MarkdownTableSerializer` to come out as Markdown. Details in `src/doc_rag/ingestion/AGENTS.md`
- Search (`POST /search`): dense (bge-m3, cosine, HNSW) + keyword (`websearch_to_tsquery`) -> RRF -> bge-reranker-v2-m3 cross-encoder; SQL lives in `ChunksRepository.search_dense/search_lexical`, not in `retrieval/`. Details and measured behaviour in `src/doc_rag/retrieval/AGENTS.md`
- Keyword search is strict AND, so it only contributes for short exact-term queries (0-2 hits on natural questions); dense carries long queries. Revisit with the phase 6 evaluation rather than guessing
- Reranker scores are sigmoid 0-1 and usable for thresholding (on-topic 0.5-1.0, off-topic < 0.05); `/search` has no cutoff, the answering step must decide "not in the documents"
- pgvector in the `pgvector/pgvector:pg17` image is 0.8.7: iterative HNSW scan is available (`hnsw.iterative_scan`), and `hnsw.*` settings are only recognised after the extension has been used in the session
- With bge-m3 + reranker + Docling loaded, the container uses about 3.8 GB of the 8 GB VRAM at rest; API startup is about 65 s with cached models; a warm search takes 0.4-0.9 s; searching during ingestion works (no OOM)
- Ukrainian queries find the right English passages (bge-m3 is cross-lingual) but the keyword search does not help for them. Windows consoles garble Cyrillic in script output; that is display only
- Chat (`POST /chat`, SSE): `ChatPipeline.run` yields `sources` -> `token`* -> `done` | `error`; it searches, drops results under `CHAT_MIN_SCORE`, numbers the rest, and streams the LLM's answer. With no relevant source it answers `NOT_FOUND_ANSWER` itself without calling the LLM. Details in `src/doc_rag/chat/AGENTS.md` and `src/doc_rag/llm/AGENTS.md`
- google-genai 2.29: `genai.Client(api_key=..., http_options=types.HttpOptions(timeout=<ms>, retry_options=types.HttpRetryOptions(attempts=..., initial_delay=...)))`; it retries nothing unless `retry_options` is set, then 408/429/5xx with backoff. Streaming is `async for chunk in await client.aio.models.generate_content_stream(model=, contents=, config=types.GenerateContentConfig(system_instruction=, temperature=))`; text is `chunk.text` (may be `None`); errors are `google.genai.errors.APIError` with `.code` and `.message`; close with `await client.aio.aclose()`. An invalid key comes back as HTTP 400, not 401
- Free-tier Gemini text models on 2026-10-12 (from ai.google.dev/gemini-api/docs/pricing): `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`; 2.0 models are shut down. The user's free daily limits (from https://aistudio.google.com/rate-limit, not published elsewhere) are 20, 20 and 500 requests; they reset at midnight Pacific time. Re-check before changing `GEMINI_MODELS`
- `GeminiClient` tries the models of `GEMINI_MODELS` in order and falls back to the next on 429, 5xx or a slow start; a rate-limited model is skipped until the `retryDelay` Gemini sent. 429 is never retried on the same model (the daily quota is tiny and every request counts, including each SDK retry). A 429 body has `error.details[]` with a `QuotaFailure` (`violations[].quotaId` such as `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, `quotaValue`) and a `RetryInfo` (`retryDelay` like `40022s`), available as `APIError.details`
- FastAPI 0.143 has built-in SSE: `from fastapi.sse import EventSourceResponse, ServerSentEvent`, route declared with `response_class=EventSourceResponse` and written as an async generator that yields `ServerSentEvent(event=, data=<pydantic model>)`; `data` is JSON-encoded automatically
- Adding a dependency invalidates the Docker dependency layer: the next `docker compose up --build` reinstalls everything (about 15 minutes including image export). Docker Desktop is installed at `D:\Apps\Docker` and must be running first; a compose command run while it is closed fails immediately with "failed to connect to the docker API"
- A live check of `/chat` answers needs a real `GEMINI_API_KEY` (set in `.env` since 2026-10-12); without one only declining and the error path can be verified
- The Gemini free tier is unreliable under load: `gemini-3.8-flash` answered in ~4 s when healthy but also returned 503 "high demand" or hung for the full timeout. The SDK's retries are silent, so a slow answer usually means retries, not a slow model. `GeminiClient` therefore puts one deadline per model (`LLM_START_TIMEOUT_SECONDS`, 20 s; healthy starts were 4-8 s) on "until the answer starts", retries included, and then falls back to the next model. `HttpOptions.timeout` (`LLM_ANSWER_TIMEOUT_SECONDS`, 180 s) is also sent to Gemini as `X-Server-Timeout`, a hard cap on the whole answer (504 "Deadline expired"), so it must not be set to a short value. Do not hold `asyncio.timeout` across a `yield` in an async generator
- To time or probe Gemini without rebuilding the image, run a script on the host that uses `GeminiClient` directly (the host has `.env`); pass `LLM_MAX_ATTEMPTS = 1` to see raw behaviour without retries. Live tests spend the daily quota: about 20 chat questions exhausted `gemini-3.8-flash` on 2026-10-12, so keep live `/chat` checks to a handful
- MCP SDK is 2.x (`mcp` 2.3): `FastMCP` was renamed `MCPServer` (`from mcp.server.mcpserver import MCPServer, Context`; `ToolError` in `mcp.server.mcpserver.exceptions`); `mcp.server.fastmcp` raises on import. Client: `from mcp import Client`, `async with Client("http://host/mcp") as c: await c.list_tools(); await c.call_tool(name, args)`. Tools are `@mcp.tool()` functions; a `ctx: Context` parameter is injected and hidden from the schema; `ctx.request_context.request` is the call's Starlette request (`None` over stdio), so `request.state` holds the FastAPI lifespan objects
- `/mcp` is mounted by extending `app.routes` with `mcp.streamable_http_app(stateless_http=True, json_response=True).routes` and running `mcp.session_manager.run()` inside the lifespan (once per instance); never mount it at `/` (catch-all shadows later routes). The SDK enables DNS-rebinding protection for localhost only (421 for other hosts)
- Register the server in Claude Code with `claude mcp add --transport http doc-rag http://localhost:8000/mcp`; tools appear after `/mcp` reconnects or a restart
- Neither the REST API nor `/mcp` is authenticated, and compose publishes port 8000 (and Postgres 5432) on all interfaces; keep the machine off untrusted networks or bind the ports to `127.0.0.1`
- The web page (`ui/`) is a client of the REST API only: no `src.doc_rag` imports, plain dicts instead of DTOs, all HTTP in `ui/client.py` (`ApiClient`, `ApiError`). Its dependencies are the `ui` group (`uv add --group ui <pkg>`), installed in `ui/Dockerfile` with `uv sync --frozen --only-group ui`, so the image has no torch. Details in `ui/AGENTS.md`
- Figure images are served by `app.mount("/assets", StaticFiles(directory=STORAGE_DIR/assets, check_dir=False))`; a chunk's `asset_path` (`assets/<document id>/<n>.png`) is the URL path. The page fetches them server-side, because the browser cannot reach `http://api:8000`
- Streamlit 1.65: `st.write_stream(generator)` streams and returns the text; `st.fragment(run_every=seconds)(function)()` makes a part refresh on its own, and `st.rerun()` inside a fragment reruns the whole page; `st.form(clear_on_submit=True)` is what empties a `file_uploader`; `st.cache_resource` is shared by all sessions. Tests use `streamlit.testing.v1.AppTest.from_file(path)` (path is relative to the test file, hence `../ui/app.py`) and replace `ui.client.ApiClient` with a fake, clearing `st.cache_resource` per test
- Evaluation (`eval/`): an HTTP client of the running API, no `src` imports; 30 questions about `attention.pdf` (arXiv 1706.03762v7) and `bert.pdf` (arXiv 1810.04805v2) with page labels read from the PDFs. Retrieval is scored with `/search` (hit@1/3/6, MRR), so it costs no Gemini quota; `--answers` calls `/chat`. Baseline (2026-10-15): hit@1 92.3%, hit@3 and hit@6 100%, MRR 0.955; relevant results score at least 0.319, unanswerable ones at most 0.070 (the GPT-4 question passes `CHAT_MIN_SCORE` 0.05 and is declined by the LLM). Label pages by reading every page of the PDF: the first labels missed pages that also state the answer and made the numbers look worse. Details in `eval/AGENTS.md`
- A game running in the background (seen: `eu5.exe`) starves the container's GPU work: a search that takes 0.5 s then takes 50 s to several minutes, the API sits at 100% CPU, and a tiny matmul in the container does not finish in 100 s. Only timings are affected, not scores. Ask the user to close it before ingesting, rebuilding or evaluating
- Embedding dimension is 1024 (bge-m3); changing the embedder requires a migration and re-embedding all chunks
- `chat/pipeline.py` and `chat/graph.py` share `retrieval/`; do not fork retrieval logic into either
- GPU models (Docling, embedder, reranker) are loaded once at startup and reused; never instantiate per request
- Gemini app subscriptions (Pro/Plus) give no API tokens; the API free tier is separate and uses an AI Studio key
- Default branch is `main`
- Commit messages are `[verb] the rest of the text` (e.g. `[add] models`). Never commit on your own: at each checkpoint stop with the work uncommitted so the user can review the diff, and commit only after they approve
- Phases of `docs/plan.md` are implemented with the `build-phase` skill, any other change with the `change` skill (`.claude/skills/*/SKILL.md`); both pin `model: sonnet`. A skill's model only applies while the skill runs, so `.claude/settings.json` (`"model": "sonnet"`) is what makes every session in this project start on Sonnet


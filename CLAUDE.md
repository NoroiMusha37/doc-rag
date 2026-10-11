# CLAUDE.md

This file provides guidance to agents when working with code in this repository.

Read `AGENTS.md` (DOX) and every child `AGENTS.md` on the path before editing. Roadmap and fixed decisions are in `docs/plan.md`.

## Project
RAG service over complex PDFs (tables, figures, formulas) that answers questions with page citations, to support a bachelor paper in financial mathematics.
FastAPI + Postgres/pgvector + Docker; local Docling, bge-m3 and bge-reranker on GPU; Gemini API free tier as the LLM.

## Commands
Run everything from the repo root:
- `uv sync` - install dependencies; `uv add <pkg>` to add one
- `docker compose up --build -d` - start Postgres and the API (needs `.env`, see `.env.example`)
- `uv run alembic revision --autogenerate -m "<msg>"` - generate a migration from the models
- `uv run alembic upgrade head` (or `docker compose exec api alembic upgrade head`) - apply migrations
- `uv run pytest` - tests
- `uv run ruff check` / `uv run ruff format` - lint and format

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
- `ui/app.py` (repo root) - Streamlit client of the API

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
- Database access only through `DBRepository` (`src/doc_rag/db/repository.py`) with its `documents` and `chunks` sub-repositories, written in SQLAlchemy 2.0 statement style. Repositories do not commit or catch errors; callers do
- Logs use f-strings (`logger.info(f"...")`) and `logging.getLogger(__name__)`; logging is configured in `src/doc_rag/logger.py`
- Components holding models or state are classes (`DocumentParser`, `DocumentChunker`, `Embedder`, `IngestionPipeline`); keep custom exceptions rare (`ParsingError` is the only one)
- Lifespan: build startup objects with plain calls (no `asyncio.to_thread`) and pass them via `yield {"pipeline": ...}` -> `request.state.pipeline`; no `app.state` assignments and no extra state classes
- The Write tool's batch is cancelled as a whole if one file in it is rejected, and writing an empty file (`__init__.py`) is reported as a rejection: create empty files from the shell and send other files one at a time
- `torch` and `torchvision` must both come from the cu128 index (`[tool.uv.sources]` has a line for each); the PyPI `torchvision` fails in the container with `operator torchvision::nms does not exist`. The user approved hand-editing that sources line; a package must also be a direct dependency (`uv add torchvision`) for its source to apply
- Measured on the RTX 4060 (15-page paper with formulas): ingestion ~79s, peak VRAM 2.9 GB, models at rest 2.2 GB; API startup ~50s with cached models (bge-m3 load is the slow part)
- RapidOCR (Docling's OCR) downloads its models into `site-packages` on the first parse, not into the `models` volume, so every new container re-downloads ~30 MB; its INFO lines bypass our logging config
- Docling formula enrichment returns LaTeX with spaced letters (`A t t e n t i o n ( Q , K , V )`); valid for display but weak for keyword search
- Docling 2.137: chunk `doc_items` are generic `DocItem`s (match by `self_ref`); pictures reach chunks only through their captions; HybridChunker merges items under one heading; tables need `MarkdownTableSerializer` to come out as Markdown. Details in `src/doc_rag/ingestion/AGENTS.md`
- Embedding dimension is 1024 (bge-m3); changing the embedder requires a migration and re-embedding all chunks
- `chat/pipeline.py` and `chat/graph.py` share `retrieval/`; do not fork retrieval logic into either
- GPU models (Docling, embedder, reranker) are loaded once at startup and reused; never instantiate per request
- Gemini app subscriptions (Pro/Plus) give no API tokens; the API free tier is separate and uses an AI Studio key
- Default branch is `main`
- Commit messages are `[verb] the rest of the text` (e.g. `[add] models`); stop at each checkpoint to commit before continuing
- Phases of `docs/plan.md` are implemented with the `build-phase` skill (`.claude/skills/build-phase/SKILL.md`), which runs on Sonnet


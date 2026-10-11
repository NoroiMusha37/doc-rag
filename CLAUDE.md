# CLAUDE.md

This file provides guidance to agents when working with code in this repository.

Read `AGENTS.md` (DOX) and every child `AGENTS.md` on the path before editing. Roadmap and fixed decisions are in `docs/plan.md`.

## Project
RAG service over complex PDFs (tables, figures, formulas) that answers questions with page citations, to support a bachelor paper in financial mathematics.
FastAPI + Postgres/pgvector + Docker; local Docling, bge-m3 and bge-reranker on GPU; Gemini API free tier as the LLM.

## Commands
Available from phase 1 of `docs/plan.md`:
- `uv sync` - install dependencies
- `docker compose up --build` - start Postgres, API and UI
- `uv run alembic upgrade head` - apply migrations
- `uv run pytest` - tests
- `uv run ruff check` / `uv run ruff format` - lint and format

## Architecture
Target layout under `src/doc_rag/` (built phase by phase):
- `main.py`, `config.py` - app factory and settings
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
- Package is `src/doc_rag`; imports look like `from doc_rag.retrieval.service import ...`
- Models are local except the LLM; call the LLM only through `doc_rag.llm.base.LLMClient`
- Embedding dimension is 1024 (bge-m3); changing the embedder requires a migration and re-embedding all chunks
- `chat/pipeline.py` and `chat/graph.py` share `retrieval/`; do not fork retrieval logic into either
- GPU models (Docling, embedder, reranker) are loaded once at startup and reused; never instantiate per request
- Gemini app subscriptions (Pro/Plus) give no API tokens; the API free tier is separate and uses an AI Studio key
- Default branch is `main`
- Commit messages are `[verb] the rest of the text` (e.g. `[add] models`); stop at each checkpoint to commit before continuing
- Phases of `docs/plan.md` are implemented with the `build-phase` skill (`.claude/skills/build-phase/SKILL.md`), which runs on Sonnet


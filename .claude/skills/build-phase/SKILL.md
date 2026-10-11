---
name: build-phase
description: Implement the next phase (or a given phase) of the doc-rag roadmap in docs/plan.md. Use when asked to implement, build or continue a phase of this project, e.g. "do phase 2" or "continue the build".
argument-hint: "[phase number]"
model: sonnet
---

# Build a doc-rag phase

Implement exactly one phase of `docs/plan.md`, verify it, update the docs, commit, and stop.

Phase to build: `$ARGUMENTS`. If empty, use the first unticked phase in `docs/plan.md`.

## 1. Read first

- `AGENTS.md` at the repo root, then every child `AGENTS.md` on the path to each folder you will touch.
- `CLAUDE.md`.
- `docs/plan.md`: Decisions, Target layout, Data model, and the chosen phase.

State which phase you are building and which files you expect to create or change before editing.

## 2. Implement only that phase

- Follow the target layout in `docs/plan.md`; do not invent new top-level modules.
- `__init__.py` files stay empty. Imports are full-path: `from doc_rag.retrieval.service import ...`.
- DTOs are pydantic models in the module's own `schemas.py`.
- The LLM is called only through `doc_rag.llm.base.LLMClient`.
- GPU models (Docling, embedder, reranker) are loaded once at startup and reused, never per request.
- Retrieval logic lives in `retrieval/`; `chat/pipeline.py` and `chat/graph.py` call it and do not reimplement it.
- Add dependencies with `uv add`; never edit the lockfile by hand.
- Check current library APIs (Docling, google-genai, mcp, LangGraph) in their docs or installed source before using them; do not rely on remembered signatures.
- Decisions in `docs/plan.md` (models, schema, embedding dimension, stack) are fixed. If one looks wrong or blocks the phase, stop and ask the user.
- Stay inside the phase. Note ideas for later phases in the final report instead of building them.

## 3. Verify

- The phase's "done when" check from `docs/plan.md`, actually run.
- `uv run ruff check` and `uv run pytest`.
- `uv run python eval/run_eval.py` once it exists (phase 6 onward), compared with the recorded baseline.

If a check fails, fix the cause. If you cannot, report the failure with its output; do not weaken or skip the check.

## 4. DOX pass

- Create a child `AGENTS.md` for each new durable folder (sections: Purpose, Ownership, Local Contracts, Work Guidance, Verification, Child DOX Index) and update existing ones whose contracts changed.
- Refresh the Child DOX Index in the parent `AGENTS.md`.
- Add anything newly learned about libraries, models or services to the Facts list in `CLAUDE.md`.
- Tick the phase in `docs/plan.md` and in the README roadmap.

## 5. Commit and stop

- Show `git status` and a one-paragraph summary of what was built and how it was verified.
- Commit with a subject of the form `[verb] the rest of the text`, lowercase verb in brackets, for example `[add] ingestion pipeline` or `[fix] chunk page ranges`. Large phases may be split into several such commits at self-contained points (for example models and migration first).
- Do not push. Do not start the next phase; wait for the user.
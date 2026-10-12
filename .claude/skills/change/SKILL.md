---
name: change
description: Make a change to doc-rag that is not a whole roadmap phase - a bug fix, a refactor, review feedback, a small feature, a dependency or docs update. Use for any such request in this project, e.g. "fix the upload error", "rename this", "apply my review comments".
argument-hint: "[what to change]"
model: sonnet
---

# Change doc-rag

Make one requested change, verify it, update the docs, and stop for review.

Change to make: `$ARGUMENTS`. If empty, use the request in the conversation.

## 1. Read first

- `AGENTS.md` at the repo root, then every child `AGENTS.md` on the path to each file you will touch.
- `CLAUDE.md`, especially the Facts list.
- The code you are about to change and its tests.

If the request would harm the design, correctness or robustness, say what breaks and offer the alternative before writing anything; then follow the user's decision.

## 2. Make only that change

- Follow the contracts in the `AGENTS.md` chain and the Facts in `CLAUDE.md`; they hold the project's conventions (imports, settings, repository, dependencies, lifespan, logging, no single-use variables).
- Add dependencies with `uv add` (`uv add --group ui` for the web page); never edit dependencies or the lockfile by hand.
- Generate migrations with `alembic revision --autogenerate`; never hand-write them.
- Check a library's current API in its docs or installed source before using it.
- Decisions in `docs/plan.md` are fixed. If the change needs one of them to move, stop and ask.
- Do not widen the scope. Note related problems you find in the final report instead of fixing them.

## 3. Verify

- `uv run ruff check` and `uv run --group ui pytest`; add or update tests for the changed behaviour.
- When the change affects running behaviour, check it against the compose stack as well (`docker compose up --build -d`). Live `/chat` questions spend the Gemini free quota: keep them to a handful.
- `uv run python eval/run_eval.py` once it exists, for any retrieval or prompt change.

If a check fails, fix the cause. If you cannot, report the failure with its output; do not weaken or skip the check.

## 4. DOX pass

- Update the nearest owning `AGENTS.md` when a contract, structure or workflow changed; refresh any affected Child DOX Index.
- Add anything newly learned about libraries, models or services to the Facts list in `CLAUDE.md`.
- When explaining the change, explain the idea in plain language before the syntax.

## 5. Stop for review, commit only after approval

- Do not commit. Leave the changes in the working tree, show `git status`, summarize what changed and how it was verified (and what was not verified), and propose a commit message.
- Commit only after the user explicitly approves. The subject has the form `[verb] the rest of the text`, for example `[fix] chunk page ranges`.
- Do not push.

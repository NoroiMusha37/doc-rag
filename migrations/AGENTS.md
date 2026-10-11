# migrations

## Purpose

Alembic migrations for the Postgres schema (async template).

## Ownership

- `env.py` - reads the database URL from `src.doc_rag.config.settings`; `alembic.ini` at the repo root holds no URL
- `versions/` - revision files (`60e9ff6f7f24` creates `documents` and `chunks` and enables the `vector` extension)
- `script.py.mako` - revision template; imports `pgvector.sqlalchemy` so `VECTOR` columns render correctly

## Local Contracts

- Revisions are generated with `uv run alembic revision --autogenerate -m "<message>"` after the SQLAlchemy models change, then reviewed; they are not written from scratch.
- `target_metadata` in `env.py` is `Base.metadata` from `src.doc_rag.db.models`; new models must be defined there so autogenerate sees them.
- `alembic.ini` runs `ruff check --fix` and `ruff format` on every new revision (post-write hooks), so generated files already match the project style and unused template imports are dropped.
- Autogenerate does not emit `CREATE EXTENSION`; the first revision adds `CREATE EXTENSION IF NOT EXISTS vector` by hand and downgrades never drop the extension.
- Autogenerate needs a reachable database: run it on the host against the compose `db` service (`DATABASE_URL` in `.env` points at `localhost:5432`).
- Migrations are applied explicitly; the API container does not run them on start.

## Work Guidance

## Verification

- `uv run alembic upgrade head` on the host, or `docker compose exec api alembic upgrade head`
- `alembic current` shows the head revision
- `uv run alembic check` reports "No new upgrade operations detected" when the models and the migrations agree

## Child DOX Index

None.
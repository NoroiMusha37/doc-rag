# migrations

## Purpose

Alembic migrations for the Postgres schema (async template).

## Ownership

- `env.py` - reads the database URL from `src.doc_rag.config.settings`; `alembic.ini` at the repo root holds no URL
- `versions/` - revision files

## Local Contracts

- Revisions are generated with `uv run alembic revision --autogenerate -m "<message>"` after the SQLAlchemy models change, then reviewed; they are not written from scratch.
- `versions/` is empty until the first models exist. The first revision must also run `CREATE EXTENSION IF NOT EXISTS vector` before creating tables, because autogenerate does not emit it.
- For autogenerate, `target_metadata` in `env.py` must point at the models' `Base.metadata` (currently `None`).
- Migrations are applied explicitly; the API container does not run them on start.

## Work Guidance

## Verification

- `uv run alembic upgrade head` on the host, or `docker compose exec api alembic upgrade head`
- `alembic current` shows the head revision

## Child DOX Index

None.
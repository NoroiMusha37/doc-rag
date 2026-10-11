# src/doc_rag/documents

## Purpose

HTTP API for documents: upload, list, status, delete.

## Ownership

- `router.py` - `/documents` routes
- `service.py` - `DocumentsService`: validation, files on disk, document rows; built per request from the request's `DBRepository`
- `schemas.py` - `DocumentResponse`

## Local Contracts

- `POST /documents` only stores the file and a `pending` row, then schedules `IngestionPipeline.ingest` as a background task. It returns 202 when ingestion was scheduled and 200 when the same file (by SHA-256) already exists.
- Re-uploading a file whose document is `failed` resets it to `pending` and ingests it again; any other existing status is returned untouched.
- Only `.pdf` files that start with `%PDF-` are accepted (415 otherwise), up to `MAX_UPLOAD_MB` (413), and not empty (400). A rejected upload leaves no file and no row behind.
- `DELETE /documents/{id}` deletes the row first (chunks cascade), then the upload and the figures folder.
- The service raises `fastapi.HTTPException` directly; routes contain no logic beyond calling the service and scheduling ingestion.
- `DocumentsService(db)` receives the request's repository and commits through it; it never opens a session or a repository itself.
- Routes receive the service and the pipeline through `get_documents_service` and `get_pipeline` from `src/doc_rag/dependencies.py`; tests override those, because the lifespan does not run under the test client.
- File paths come from `upload_path()` and `assets_dir()` in `ingestion/pipeline.py`; do not build them here.
- Blocking disk work in the service runs in `asyncio.to_thread`.

## Work Guidance

- To accept another file type, extend `ALLOWED_SUFFIXES` and the signature check in `service.py`, and register the format in `DocumentParser`.

## Verification

- `uv run pytest tests/test_documents_service.py tests/test_documents_api.py` - service against a fake repository and a temp folder; routes against a fake service and pipeline
- With the stack running: `curl -F "file=@paper.pdf" http://localhost:8000/documents`, then poll `GET /documents/{id}` until `ready`

## Child DOX Index

None.

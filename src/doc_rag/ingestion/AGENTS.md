# src/doc_rag/ingestion

## Purpose

Turns a stored upload into embedded chunks: parse with Docling, chunk, embed, store.

## Ownership

- `parser.py` - `DocumentParser`: owns the Docling `DocumentConverter` and its models
- `chunker.py` - `DocumentChunker`: Docling `HybridChunker` with the embedder's tokenizer; builds `ParsedChunk`s and saves figure images
- `pipeline.py` - `IngestionPipeline`: orchestrates parse -> chunk -> embed -> store and records status on the document row; `upload_path()` and `assets_dir()` define where files live under `STORAGE_DIR`
- `schemas.py` - `ParsedChunk`, `ParsedDocument`
- `exceptions.py` - `ParsingError`, the only custom exception

The embedder lives in `src/doc_rag/embedding/embedder.py` (`Embedder`).

## Local Contracts

- `DocumentParser`, `DocumentChunker` and `Embedder` are built once, in the lifespan in `main.py`, and passed to `IngestionPipeline`; the same `Embedder` also serves search. Handlers read `request.state.pipeline` through `get_pipeline`.
- `IngestionPipeline.ingest(document_id)` expects the `documents` row and the uploaded file to exist already; it never creates the row. It never raises: failures set `status = failed` and `error` on the row and are logged.
- `IngestionPipeline.fail_interrupted()` runs once in the lifespan: documents left in `pending`/`parsing`/`embedding` by a restart are marked `failed` (re-uploading the file retries them). It only logs a warning when the database is unreachable or not migrated, so the app still starts.
- Status flow: `pending` -> `parsing` -> `embedding` -> `ready`, or `failed`. One document is processed at a time (an `asyncio.Lock`), because parsing and embedding share the GPU.
- All database access goes through `DBRepository` (`db/repository.py`); background code opens it with `DBRepository.open()`. No sessions or SQL in this folder.
- Chunks and the `ready` status are written in one transaction; re-ingesting replaces a document's chunks.
- Files: uploads at `STORAGE_DIR/uploads/<document_id><suffix>`, figure images at `STORAGE_DIR/assets/<document_id>/<ordinal>.png`. `Chunk.asset_path` is relative to `STORAGE_DIR`.
- Chunk facts that come from Docling and are easy to get wrong:
  - `chunk.meta.doc_items` are generic `DocItem` references, never `PictureItem`/`TableItem`; resolve them by `self_ref`.
  - A picture is not in a chunk's items; only its caption is. Pictures are linked to chunks through `picture.captions`, so a picture without a caption has no chunk.
  - Items under the same heading are merged into one chunk, so `kind` is the richest content present: table, then figure, then formula (only if nothing else), then text.
  - Tables are serialized as Markdown and formulas as `$$latex$$` inside `text`.
- A partially converted document (timeout or failed pages) is kept and logged as a warning; only a full conversion failure raises `ParsingError`.

## Work Guidance

- Classes for components that hold models or state (parser, chunker, embedder, pipeline).
- Log with f-strings via `logging.getLogger(__name__)`: model load times, per-document timings and counts at INFO, recoverable problems at WARNING, failures with `logger.exception`.
- Keep custom exceptions rare; use built-in ones unless callers need to tell the case apart.
- Handle errors where the outcome is decided (the pipeline), not in the repository or helpers.

## Verification

- `uv run pytest tests/test_chunker.py tests/test_pipeline.py` - runs the real chunker on an in-memory document with a fake tokenizer, and the pipeline with fake models and a fake repository; no GPU, network or database needed

## Child DOX Index

None.
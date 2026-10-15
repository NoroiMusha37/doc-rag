# doc-rag plan

Source of truth for the roadmap and for decisions that are fixed. Change a decision here only after agreeing it with the user.

## Goal

Ingest complex PDFs (tables, figures, formulas, slides) and answer questions about them with page-level citations, to support writing a bachelor paper in financial mathematics. Free models and services only. Hardware: RTX 4060, 8 GB VRAM.

## Decisions

| Concern         <br | Choice                                                                                         | Why                                                                                                                                         |
|---------------------|------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| API                 | FastAPI                                                                                        | Fixed by the user.                                                                                                                          |
| Parser              | Docling (local, GPU)                                                                           | Layout-aware; tables -> Markdown, formulas -> LaTeX, figure image export, OCR. Also reads PPTX/DOCX/XLSX/HTML.                              |
| Embedder            | `BAAI/bge-m3` (local, 1024-dim)                                                                | Free, ~1 GB VRAM in fp16, multilingual (English and Ukrainian).                                                                             |
| Reranker            | `BAAI/bge-reranker-v2-m3` (local)                                                              | Largest quality gain after hybrid search; multilingual.                                                                                     |
| Vector store        | Postgres 17 + pgvector (HNSW, cosine)                                                          | Fixed by the user. The same table holds a `tsvector` column.                                                                                |
| Retrieval           | Dense + Postgres full-text, fused with RRF, then rerank                                        | Dense-only retrieval misses exact terms and symbols.                                                                                        |
| LLM                 | Gemini API free tier via `google-genai`, behind `LLMClient`; Ollama as a second implementation | App subscriptions give no API tokens; the API free tier is separate, rate-limited, and inputs may be used for training. Model id is config. |
| Figure captions     | Gemini vision at ingest, optional flag, off by default                                         | Makes charts retrievable by text without spending quota by default.                                                                         |
| DB access           | SQLAlchemy 2 async + asyncpg + `pgvector` + Alembic                                            |                                                                                                                                             |
| MCP                 | Official `mcp` SDK (FastMCP), Streamable HTTP mounted in the FastAPI app                       |                                                                                                                                             |
| UI                  | Streamlit, separate container, talks to the API only                                           |                                                                                                                                             |
| Tooling             | uv, ruff, pytest                                                                               |                                                                                                                                             |

### Two chat orchestrators, one retrieval layer

- Ingestion, embedding and retrieval are hand-written once.
- `chat/pipeline.py`: hand-written linear flow (retrieve -> rerank -> prompt -> answer).
- `chat/graph.py`: LangGraph flow (rewrite query -> retrieve -> grade relevance -> re-retrieve or answer).
- `/chat` selects one per request with `mode: "simple" | "graph"`; the eval set compares them.

## Target layout

```
src/doc_rag/
  main.py                 module-level app; mounts routers and MCP
  config.py               pydantic-settings
  db/                     session.py, models.py
  documents/              router.py, schemas.py, service.py
  ingestion/              parser.py, chunker.py, captioner.py, pipeline.py, schemas.py
  embedding/              embedder.py, reranker.py
  retrieval/              fusion.py, service.py, schemas.py, router.py   (dense and keyword SQL live in db/repository.py)
  llm/                    base.py, gemini.py, ollama.py, prompts.py
  chat/                   router.py, schemas.py, pipeline.py, graph.py
  mcp_server/             server.py   (not named `mcp`, to avoid shadowing the SDK)
ui/                      app.py, client.py, Dockerfile
migrations/
eval/                     questions.jsonl, run_eval.py
tests/
Dockerfile, compose.yaml, .env.example
```

## Data model

- `documents`: id, filename, sha256 (unique; dedupes re-uploads), status (`pending|parsing|embedding|ready|failed`), error, page_count, created_at.
- `chunks`: id, document_id (cascade delete), ordinal, kind (`text|table|figure|formula`), text (shown and sent to the LLM), embed_text (heading path + text), heading_path, page_start, page_end, asset_path (figure image), `embedding vector(1024)` with an HNSW index, `tsv tsvector` generated from text with a GIN index.

## Phases

Tick a phase when its commit lands.

- [x] **1. Skeleton.** Compose with `pgvector/pgvector:pg17` and the app (NVIDIA device reservation, Hugging Face model cache as a volume); config; DB session; Alembic configured (no revisions yet); `GET /health`.
  *Done when:* `docker compose up` gives a healthy API and `torch.cuda.is_available()` is true inside the container.
- [x] **2. Ingestion.** SQLAlchemy models and the first autogenerated migration (with `CREATE EXTENSION vector` added to it). `POST /documents` stores the file and a `pending` row; a background task runs Docling (table structure, formula enrichment, picture images) -> chunker (Docling HybridChunker with the bge-m3 tokenizer; tables kept whole with heading context) -> embed in batches -> insert. `GET /documents`, `GET /documents/{id}`, `DELETE /documents/{id}`.
  *Done when:* a PDF with tables reaches `ready` and its chunks show intact tables and LaTeX.
- [x] **3. Retrieval.** Dense kNN, lexical `websearch_to_tsquery`, RRF fusion, rerank top-30 -> top-6. `POST /search` returns chunks with scores and pages.
  *Done when:* searching an exact table header and a paraphrased concept both return the right page.
- [x] **4. Answering (hand-written).** `LLMClient` protocol; Gemini implementation with retry on 429; prompt that requires `[n]` citations and says so when the documents do not contain the answer. `POST /chat` streams over SSE and returns the answer plus cited chunks.
  *Done when:* a question answered by a table gets a correct answer citing that page, and an off-topic question is declined.
- [x] **5. UI and MCP.** Streamlit page (upload, document status, chat, expandable sources with page numbers and figure images). MCP tools `search_documents`, `ask_documents`, `list_documents`.
  *Done when:* the whole flow works from the browser, and Claude Code lists and calls the MCP tools.
- [x] **6. Evaluation.** 20-30 hand-written questions with expected document and page; `eval/run_eval.py` reports hit@k and MRR and dumps answers for manual review.
  *Done when:* the script runs end to end and its numbers are recorded as the baseline.
- [x] **7. LangGraph orchestrator.** `chat/graph.py` with rewrite, grade and retry nodes; `mode` switch on `/chat`.
  *Done when:* both modes run on the eval set and the comparison is recorded.
- [ ] **8. Later, as needed.** Figure captioning via Gemini vision; conversation history; Ollama client; PPTX/DOCX upload (only the upload whitelist changes); a separate worker if background ingestion blocks the API.

## Known risks

- Image size: the API image is ~11 GB with CUDA torch alone and grows with Docling; first start downloads ~5 GB of models into the cache volume.
- VRAM: Docling models plus bge-m3 and the reranker fit in 8 GB, but ingesting and querying at the same time may not. Ingestion runs one document at a time.
- Gemini free-tier limits change; retry/backoff and the Ollama client exist for that.
- torch with CUDA on Python 3.13 is verified; Docling on 3.13 is verified in phase 2, with 3.12 in the image as the fallback.

## Verification

- Per phase: its "done when" check, `uv run ruff check`, `uv run pytest`.
- From phase 6: `uv run python eval/run_eval.py` before and after any retrieval or prompt change.
- End to end: upload a real course PDF, ask one question answered by a table and one answered by a formula, and open the cited pages.

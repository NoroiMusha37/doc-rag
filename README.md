# doc-rag

A retrieval-augmented question answering service over your own documents. It parses complex PDFs (tables, figures, formulas, slides), indexes them in Postgres with pgvector, and answers questions with citations to the source pages. Built to support writing a bachelor paper in financial mathematics, using only free models and services.

**Status:** in progress. The roadmap below tracks what works so far; details are in [docs/plan.md](docs/plan.md).

## How it works

```
Ingest:  PDF -> Docling (layout, tables, formulas, figures) -> chunks
             -> bge-m3 embeddings -> Postgres (pgvector + full-text)

Query:   question -> dense search + full-text search -> RRF fusion
                  -> bge reranker -> top chunks -> Gemini -> answer with [n] citations
```

The answering step has two interchangeable implementations over the same retrieval layer: a hand-written linear pipeline and a LangGraph flow that can rewrite the query and retry.

## Stack

| Concern       | Choice                                         |
|---------------|------------------------------------------------|
| API           | FastAPI                                        |
| Parser        | Docling, local on GPU                          |
| Embedder      | `BAAI/bge-m3`, local, 1024-dim, multilingual   |
| Reranker      | `BAAI/bge-reranker-v2-m3`, local               |
| Store         | Postgres 17 + pgvector (HNSW) + `tsvector`     |
| LLM           | Gemini API free tier; Ollama as an alternative |
| Orchestration | Hand-written pipeline and LangGraph            |
| Interfaces    | REST API, MCP server, Streamlit UI             |
| Tooling       | Docker Compose, uv, Alembic, ruff, pytest      |

## Quickstart

Prerequisites:

- Docker Desktop with the WSL2 backend and NVIDIA GPU support
- [uv](https://docs.astral.sh/uv/) for local development
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/)

```
cp .env.example .env      # set GEMINI_API_KEY and the Postgres password
docker compose up --build -d
docker compose exec api alembic upgrade head
```

| Service               | URL                        |
|-----------------------|----------------------------|
| API docs              | http://localhost:8000/docs |
| MCP (Streamable HTTP) | http://localhost:8000/mcp  |
| Web UI                | http://localhost:8501      |

The first start downloads about 5 GB of models into a Docker volume.

## API

| Method | Path              | Purpose                                                        |
|--------|-------------------|----------------------------------------------------------------|
| GET    | `/health`         | Liveness, DB and GPU status                                    |
| POST   | `/documents`      | Upload a file and start ingestion                              |
| GET    | `/documents`      | List documents with ingestion status                           |
| GET    | `/documents/{id}` | One document's status and metadata                             |
| DELETE | `/documents/{id}` | Remove a document and its chunks                               |
| POST   | `/search`         | Retrieve ranked chunks for a query                             |
| POST   | `/chat`           | Answer a question with citations (`mode`: `simple` or `graph`) |

MCP tools: `search_documents`, `ask_documents`, `list_documents`.

## Roadmap

- [x] 1. Skeleton: Compose, config, database, migrations, health check
- [x] 2. Ingestion: upload, Docling parsing, chunking, embedding
- [ ] 3. Retrieval: hybrid search, RRF fusion, reranking
- [ ] 4. Answering: hand-written pipeline with Gemini and citations
- [ ] 5. Streamlit UI and MCP server
- [ ] 6. Evaluation set and script
- [ ] 7. LangGraph orchestrator
- [ ] 8. Later: figure captioning, chat history, Ollama, more file types

## A note on the Gemini free tier

The Gemini API free tier is separate from Gemini app subscriptions, is rate-limited, and Google may use free-tier inputs to improve its products. Only document excerpts that are retrieved for a question are sent. For unpublished material you would rather keep private, switch the LLM to the local Ollama client.

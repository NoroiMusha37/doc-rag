# Doc-RAG Backend

A production-grade Retrieval-Augmented Generation (RAG) backend system designed to parse complex PDFs, generate vector embeddings, and answer queries using the Google Gemini API.

## Technology Stack
- **Language:** Python
- **API Framework:** FastAPI
- **Database:** PostgreSQL with `pgvector`
- **LLM & Embeddings:** Google Gemini API
- **Dependency Management:** `uv`
- **Infrastructure:** Docker & Docker Compose

## High-Level Architecture
1. **Upload & Parse:** Users upload PDFs (papers, presentations) via an asynchronous FastAPI endpoint. The system extracts text handling complex layouts.
2. **Semantic Chunking:** Text is split into semantic chunks (500-1000 tokens) with overlap to preserve context.
3. **Vectorization:** Chunks are sent to the Gemini API to generate semantic embeddings.
4. **Storage:** Vectors and metadata are stored in PostgreSQL using the `pgvector` extension.
5. **Retrieval & Synthesis:** User queries are embedded, similar chunks are retrieved from the DB, and the Gemini LLM synthesizes a final, context-aware answer.

## Folder Structure
```text
doc-rag/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration (Pydantic BaseSettings)
│   ├── logger.py               # Centralized robust logging setup
│   ├── exceptions.py           # Custom exception definitions & handlers
│   ├── database.py             # SQLAlchemy engine & session setup
│   ├── models.py               # SQLAlchemy ORM models (pgvector)
│   ├── schemas.py              # Pydantic V2 models for API validation
│   ├── routes.py               # API endpoints
│   ├── dependencies.py         # FastAPI dependencies (e.g., get_db)
│   └── services/               # Core business and RAG logic
│       ├── pdf_parser.py
│       ├── chunker.py
│       ├── llm_service.py
│       └── rag_engine.py
├── pyproject.toml              # uv dependencies
├── Dockerfile                  # FastAPI container
└── docker-compose.yml          # Services (PostgreSQL + pgvector + App)
```

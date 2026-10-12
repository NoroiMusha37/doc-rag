# ui

## Purpose

The Streamlit web page: upload PDFs, watch their ingestion status, ask questions, and read the answer with its sources (pages, text, figure images).

## Ownership

- `app.py` - the page. Streamlit reruns it top to bottom on every interaction
- `client.py` - `ApiClient` (every HTTP call to the API, including reading the `/chat` event stream) and `ApiError`
- `Dockerfile` - the `ui` image; built with the repository root as context

## Local Contracts

- The page is a client of the REST API and nothing else: it never imports `src.doc_rag`, never opens the database and never calls a model. Whatever it needs is added to the API first.
- `API_URL` is read from the environment with no default (`os.environ["API_URL"]`), so the page fails at once when it is missing. Compose sets it to `http://api:8000`.
- All HTTP goes through `ApiClient`; `app.py` does not import `httpx`. The client turns every failure into `ApiError` with a message meant for the user (the API's `detail` text, or that the API did not respond), and the page shows it with `st.error` instead of crashing.
- One `ApiClient` is shared by all browser sessions (`st.cache_resource`). The conversation lives in `st.session_state.messages`, per browser tab, and is lost on reload; the API keeps no chat history.
- API responses stay plain dicts; there are no DTO classes on this side.
- Figure images are fetched by the page's server through `ApiClient.asset()` and handed to `st.image` as bytes, because the browser cannot reach `http://api:8000`. `asset_path` from a source is the URL path as it is (`assets/<document id>/<n>.png`).
- The document list is a fragment that re-runs every `POLL_SECONDS` only while a document is being ingested (or the API is unreachable). When ingestion ends it reruns the whole page once, but never while `st.session_state.answering` is set, because a rerun would cut off the answer being streamed.
- Dependencies live in the `ui` dependency group (`uv add --group ui <pkg>`), installed in the image with `uv sync --only-group ui`; nothing from the API's dependencies is available here.
- Imports are full-path from the repo root (`from ui.client import ApiClient`); the image sets `PYTHONPATH=/app` for that. `__init__.py` stays empty.

## Work Guidance

- Keep `app.py` to drawing and `client.py` to HTTP; logic worth testing without a browser belongs in `client.py`.
- Source text is rendered as Markdown so tables and `$...$` formulas display; a text with two plain dollar amounts on one line can therefore be shown as a formula.

## Verification

- `uv run --group ui pytest tests/test_ui_client.py tests/test_ui_app.py` - the page tests use Streamlit's `AppTest` with a fake client and are skipped when Streamlit is not installed (plain `uv run pytest` on a fresh environment)
- `docker compose up --build -d`, then open http://localhost:8501: upload a PDF, wait for `ready`, ask a question, open a source
- Run the page outside Docker: `API_URL=http://localhost:8000 uv run --group ui python -m streamlit run ui/app.py` (`python -m` puts the repo root on the import path)

## Child DOX Index

None.

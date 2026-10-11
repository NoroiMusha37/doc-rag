FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/models \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so source edits do not invalidate the (large) torch layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY alembic.ini ./
COPY migrations migrations
COPY src src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=10 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"]

CMD ["uvicorn", "src.doc_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
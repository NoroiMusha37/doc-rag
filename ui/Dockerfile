# Built from the repository root (see compose.yaml), which holds pyproject.toml and uv.lock.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/

# PYTHONPATH lets the page import `ui.client` the same way the tests do.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Only the `ui` dependency group: Streamlit and httpx, none of the API's models.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --only-group ui

COPY ui ui

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"]

CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]

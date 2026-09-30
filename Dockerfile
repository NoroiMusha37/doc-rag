# Use Python 3.13 slim image
FROM python:3.13-slim-bookworm

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Enable bytecode compilation to optimize performance
ENV UV_COMPILE_BYTECODE=1

# Copy pyproject.toml and optional uv.lock
COPY pyproject.toml uv.lock* /app/

# Install the project's dependencies
RUN if [ -f uv.lock ]; then uv sync --frozen --no-install-project; else uv sync --no-install-project; fi

# Copy the rest of the application code
COPY . /app

# Complete the sync to install the project itself (if applicable)
RUN if [ -f uv.lock ]; then uv sync --frozen; else uv sync; fi

# Add healthcheck
HEALTHCHECK --interval=300s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Set default execution command
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--workers", "4"]

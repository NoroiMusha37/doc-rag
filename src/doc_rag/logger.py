import logging.config

from src.doc_rag.config import settings

APP_LOGGER = "src.doc_rag"


def setup_logging() -> None:
    """Configure logging once, at application start.

    - Application loggers (`logging.getLogger(__name__)` under `src.doc_rag`) log at LOG_LEVEL.
    - Third-party libraries (Docling, transformers, httpx, ...) only log warnings and errors.
    - uvicorn's own handlers are removed so its lines go through the same handler and format.
    """
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {"level": "WARNING", "handlers": ["console"]},
            "loggers": {
                APP_LOGGER: {"level": settings.LOG_LEVEL},
                "uvicorn": {"level": "INFO", "handlers": [], "propagate": True},
                "uvicorn.access": {"level": "INFO", "handlers": [], "propagate": True},
            },
        }
    )

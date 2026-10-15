from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = Field(min_length=1)
    GEMINI_API_KEY: str = Field(min_length=1)
    STORAGE_DIR: str = "data"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    MAX_UPLOAD_MB: int = 100

    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_BATCH_SIZE: int = 16
    CHUNK_MAX_TOKENS: int = 512

    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_BATCH_SIZE: int = 16
    # Chunks fetched by each of the two searches (dense and keyword).
    SEARCH_CANDIDATES: int = 30
    # Best fused chunks handed to the reranker.
    RERANK_CANDIDATES: int = 30
    # Chunks returned when a request does not set top_k.
    SEARCH_TOP_K: int = 6
    # Reciprocal rank fusion constant; 60 is the value from the original paper.
    RRF_K: int = 60

    # Comma-separated, best first. When a model has no quota left or is overloaded, the
    # next one answers. All must have a free tier: https://ai.google.dev/gemini-api/docs/pricing
    # Free daily limits seen on 2026-10-12: 20, 20 and 500 requests.
    GEMINI_MODELS: str = "gemini-3.8-flash,gemini-3.6-flash,gemini-3.5-flash-lite"
    # Low values keep answers close to the sources instead of creative.
    LLM_TEMPERATURE: float = 0.2
    # Tries per LLM call, the first included; rate limits and server errors are retried.
    LLM_MAX_ATTEMPTS: int = 5
    # Longest wait for one model to start answering, retries included; after that the
    # next model is tried. Healthy calls started in 4-8 seconds when this was measured.
    LLM_START_TIMEOUT_SECONDS: int = 20
    # Longest total time for one answer. Gemini itself stops generating at this point.
    LLM_ANSWER_TIMEOUT_SECONDS: int = 180
    # Search results scoring below this are not shown to the LLM. If none remain, the
    # question is declined without calling the LLM at all.
    CHAT_MIN_SCORE: float = 0.05

    # Graph mode: when the best search result scores below this, the question is rewritten
    # and searched again. In the phase 6 evaluation every relevant result scored at least
    # 0.319, and questions the documents cannot answer scored at most 0.07.
    GRAPH_GOOD_SCORE: float = 0.3
    # Extra searches graph mode may make with a rewritten question; each costs one LLM call.
    GRAPH_MAX_RETRIES: int = 1

    # LaTeX for formulas; slow and VRAM-heavy, turn off for text-only documents.
    DOCLING_FORMULA_ENRICHMENT: bool = True
    # A document that takes longer is cut off and ingested with the pages parsed so far.
    DOCLING_TIMEOUT_SECONDS: float = 1800


settings = Settings()

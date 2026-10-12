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

    # Must be a model with a free tier: https://ai.google.dev/gemini-api/docs/pricing
    GEMINI_MODEL: str = "gemini-3.8-flash"
    # Low values keep answers close to the sources instead of creative.
    LLM_TEMPERATURE: float = 0.2
    # Tries per LLM call, the first included; rate limits and server errors are retried.
    LLM_MAX_ATTEMPTS: int = 5
    # Longest wait for the answer to start, retries included. A healthy call starts in
    # about 5 seconds; the rest is room for retries when Gemini is overloaded.
    LLM_START_TIMEOUT_SECONDS: int = 45
    # Longest total time for one answer. Gemini itself stops generating at this point.
    LLM_ANSWER_TIMEOUT_SECONDS: int = 180
    # Search results scoring below this are not shown to the LLM. If none remain, the
    # question is declined without calling the LLM at all.
    CHAT_MIN_SCORE: float = 0.05

    # LaTeX for formulas; slow and VRAM-heavy, turn off for text-only documents.
    DOCLING_FORMULA_ENRICHMENT: bool = True
    # A document that takes longer is cut off and ingested with the pages parsed so far.
    DOCLING_TIMEOUT_SECONDS: float = 1800


settings = Settings()

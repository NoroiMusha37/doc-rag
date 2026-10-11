from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = Field(min_length=1)
    GEMINI_API_KEY: str = Field(min_length=1)
    STORAGE_DIR: str = "data"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_BATCH_SIZE: int = 16
    CHUNK_MAX_TOKENS: int = 512
    # LaTeX for formulas; slow and VRAM-heavy, turn off for text-only documents.
    DOCLING_FORMULA_ENRICHMENT: bool = True
    # A document that takes longer is cut off and ingested with the pages parsed so far.
    DOCLING_TIMEOUT_SECONDS: float = 1800


settings = Settings()

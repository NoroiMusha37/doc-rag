from pydantic import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    DATABASE_URL: PostgresDsn
    DATABASE_ECHO: bool = False

    GEMINI_API_KEY: str

    EMBEDDING_MODEL: str = "text-embedding-004"
    GENERATION_MODEL: str = "gemini-2.0-flash"

    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 200
    RETRIEVAL_TOP_K: int = 5

    LOG_LEVEL: str = "INFO"


settings = Settings()

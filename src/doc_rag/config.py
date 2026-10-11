from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = Field(min_length=1)
    GEMINI_API_KEY: str = Field(min_length=1)
    STORAGE_DIR: str = "data"


settings = Settings()

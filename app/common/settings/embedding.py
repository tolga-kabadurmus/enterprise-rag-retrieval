"""Validated embedding-service HTTP client settings."""

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    """Embedding API settings read from ``EMBEDDING_SERVICE_`` variables."""

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_SERVICE_",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    base_url: AnyHttpUrl
    timeout_seconds: float = Field(default=15.0, gt=0, le=120)

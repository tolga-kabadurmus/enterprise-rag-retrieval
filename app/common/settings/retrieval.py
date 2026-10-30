"""Validated retrieval behavior settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalSettings(BaseSettings):
    """Retrieval limits read from ``RAG_`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
        frozen=True,
    )

    top_k: int = Field(default=5, ge=1, le=50)
    score_threshold: float | None = Field(default=None, ge=0, le=1)
    max_context_characters: int = Field(default=20_000, ge=1, le=200_000)

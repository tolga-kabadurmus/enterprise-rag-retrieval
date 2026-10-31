"""Validated retrieval behavior settings."""

from pydantic import Field, model_validator
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
    candidate_k: int = Field(default=20, ge=1, le=100)
    hybrid_prefetch_limit: int = Field(default=20, ge=1, le=200)
    score_threshold: float | None = Field(default=None, ge=0, le=1)
    max_context_characters: int = Field(default=20_000, ge=1, le=200_000)

    @model_validator(mode="after")
    def validate_prefetch_limit(self) -> "RetrievalSettings":
        """Ensure each hybrid branch can supply at least the final result count."""

        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be at least top_k")
        if self.hybrid_prefetch_limit < self.candidate_k:
            raise ValueError("hybrid_prefetch_limit must be at least candidate_k")
        return self

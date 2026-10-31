"""Validated local reranking settings."""

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RerankingProvider(str, Enum):
    """Supported reranking repository provider keys."""

    FLASHRANK = "flashrank"


class RerankingSettings(BaseSettings):
    """FlashRank settings read from ``RERANKING_`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="RERANKING_",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
        frozen=True,
    )

    provider: RerankingProvider = RerankingProvider.FLASHRANK
    model_name: str = Field(default="ms-marco-TinyBERT-L-2-v2", min_length=1)
    cache_dir: str = Field(default="/tmp/flashrank", min_length=1)
    max_length: int = Field(default=128, ge=32, le=512)
    max_concurrency: int = Field(default=1, ge=1, le=16)

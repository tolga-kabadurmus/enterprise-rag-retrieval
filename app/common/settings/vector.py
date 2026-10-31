"""Validated vector-store client settings."""

from enum import Enum

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorProvider(str, Enum):
    """Supported vector repository provider keys."""

    QDRANT = "qdrant"


class VectorSettings(BaseSettings):
    """Vector-store settings read from ``QDRANT_`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
        frozen=True,
    )

    provider: VectorProvider = VectorProvider.QDRANT
    url: AnyHttpUrl
    api_key: SecretStr | None = None
    collection_name: str = Field(min_length=1)
    dense_vector_name: str = Field(default="dense", min_length=1)
    sparse_vector_name: str = Field(default="sparse", min_length=1)
    vector_size: int = Field(default=384, ge=1)
    timeout_seconds: int = Field(default=10, ge=1, le=120)

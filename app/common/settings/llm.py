"""Validated configurable LLM client settings."""

from enum import Enum

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM repository provider keys."""

    OPENAI_COMPATIBLE = "openai_compatible"


class LLMSettings(BaseSettings):
    """LLM settings read from ``LLM_`` environment variables.

    Provider, endpoint, and model are required so the application does not
    silently select an LLM deployment.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
        frozen=True,
    )

    provider: LLMProvider
    base_url: AnyHttpUrl
    model: str = Field(min_length=1)
    api_key: SecretStr | None = None
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_output_tokens: int = Field(default=1_024, ge=1, le=32_768)

"""Validated configurable LLM client settings."""

from enum import Enum

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM repository provider keys."""

    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"


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
    system_prompt: str = Field(
        default=(
            "Answer using only the supplied context. "
            "If the context is insufficient, say so clearly."
        ),
        min_length=1,
    )
    api_key: SecretStr | None = None
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    verify_tls: bool = True
    max_connections: int = Field(default=20, ge=1)
    max_keepalive_connections: int = Field(default=10, ge=0)
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_output_tokens: int = Field(default=1_024, ge=1, le=32_768)

    @field_validator("system_prompt")
    @classmethod
    def normalize_system_prompt(cls, value: str) -> str:
        """Trim and reject a blank configured system prompt."""

        value = value.strip()
        if not value:
            raise ValueError("LLM_SYSTEM_PROMPT must not be blank")
        return value

    @model_validator(mode="after")
    def require_provider_credentials(self) -> "LLMSettings":
        """Require credentials for native hosted-provider SDKs."""

        if self.provider in {LLMProvider.ANTHROPIC, LLMProvider.COHERE}:
            if self.api_key is None or not self.api_key.get_secret_value().strip():
                raise ValueError(f"LLM_API_KEY is required for {self.provider.value}")
        return self

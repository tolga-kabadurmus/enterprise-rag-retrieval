"""Validated HTTP API process settings."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """Runtime settings read from ``QUERY_API_`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="QUERY_API_",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    host: str = Field(default="0.0.0.0", min_length=1)
    port: int = Field(default=8082, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_question_characters: int = Field(default=4_000, ge=1, le=100_000)

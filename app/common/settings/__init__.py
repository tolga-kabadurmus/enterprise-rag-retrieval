"""Validated settings package reserved for the configuration phase."""
"""Validated query-service configuration."""

from pydantic import BaseModel, ConfigDict

from app.common.settings.api import ApiSettings
from app.common.settings.embedding import EmbeddingProvider, EmbeddingServiceSettings
from app.common.settings.llm import LLMProvider, LLMSettings
from app.common.settings.reranking import RerankingProvider, RerankingSettings
from app.common.settings.retrieval import RetrievalSettings
from app.common.settings.vector import VectorProvider, VectorSettings


class ApplicationSettings(BaseModel):
    """Immutable aggregate of all independently prefixed setting groups."""

    model_config = ConfigDict(frozen=True)

    api: ApiSettings
    embedding: EmbeddingServiceSettings
    vector: VectorSettings
    retrieval: RetrievalSettings
    reranking: RerankingSettings
    llm: LLMSettings


def load_settings() -> ApplicationSettings:
    """Load and validate the complete query-service environment.

    Returns:
        An immutable aggregate suitable for dependency composition.

    Raises:
        pydantic.ValidationError: If required values are missing or invalid.
    """

    return ApplicationSettings(
        api=ApiSettings(),
        embedding=EmbeddingServiceSettings(),
        vector=VectorSettings(),
        retrieval=RetrievalSettings(),
        reranking=RerankingSettings(),
        llm=LLMSettings(),
    )


__all__ = [
    "ApiSettings",
    "ApplicationSettings",
    "EmbeddingProvider",
    "EmbeddingServiceSettings",
    "LLMProvider",
    "LLMSettings",
    "RetrievalSettings",
    "RerankingProvider",
    "RerankingSettings",
    "VectorProvider",
    "VectorSettings",
    "load_settings",
]

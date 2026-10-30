"""Embedding repository contract, factory, and HTTP implementation."""

from app.repository.embedding_service.base import EmbeddingRepository
from app.repository.embedding_service.enterprise_embedding_service_repository import (
    EmbeddingRequest,
    EmbeddingResponse,
    EnterpriseEmbeddingServiceRepository,
)
from app.repository.embedding_service.exceptions import EmbeddingServiceRequestError
from app.repository.embedding_service.factory import EmbeddingRepositoryFactory

__all__ = [
    "EmbeddingRepository",
    "EmbeddingRepositoryFactory",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingServiceRequestError",
    "EnterpriseEmbeddingServiceRepository",
]

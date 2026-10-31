"""HTTPX adapter for the shared embedding service."""

from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.common.settings.embedding import EmbeddingServiceSettings
from app.repository.embedding_service.base import (
    EmbeddingRepository,
    EmbeddingVector,
    SparseEmbedding,
)
from app.repository.embedding_service.exceptions import EmbeddingServiceRequestError


class EmbeddingRequest(BaseModel):
    """Payload accepted by the embedding service."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_type: Literal["document", "query"]
    texts: tuple[str, ...] = Field(min_length=1)


class EmbeddingResponse(BaseModel):
    """Validated response returned by the embedding service."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    model: str
    revision: str
    dimension: int = Field(ge=1)
    normalized: bool
    embeddings: tuple[tuple[float, ...], ...]
    sparse_model: str = Field(min_length=1)
    sparse_embeddings: tuple[SparseEmbedding, ...]

    @model_validator(mode="after")
    def validate_vector_counts(self) -> "EmbeddingResponse":
        """Require one aligned dense and sparse representation per input."""

        if len(self.embeddings) != len(self.sparse_embeddings):
            raise ValueError("dense and sparse vector counts must match")
        return self


class EnterpriseEmbeddingServiceRepository(EmbeddingRepository):
    """Call the embedding API using an injected pooled HTTPX client."""

    def __init__(
        self,
        httpx_client: httpx.AsyncClient,
        settings: EmbeddingServiceSettings,
    ) -> None:
        self._httpx_client = httpx_client
        self._settings = settings
        self._closed = False

    @property
    def max_batch_size(self) -> int:
        """Return the configured maximum number of texts per request."""

        return self._settings.max_batch_size

    @property
    def max_text_characters(self) -> int:
        """Return the maximum aggregate characters per request."""

        return self._settings.max_text_characters

    async def embed(
        self,
        texts: tuple[str, ...],
        input_type: Literal["document", "query"] = "document",
    ) -> tuple[EmbeddingVector, ...]:
        """Vectorize a bounded batch and validate its model contract."""

        if self._closed:
            raise RuntimeError("embedding service repository is closed")
        if not texts:
            return ()
        if len(texts) > self._settings.max_batch_size:
            raise ValueError(
                "embedding request exceeds max batch size "
                f"({self._settings.max_batch_size})"
            )
        character_count = sum(len(text) for text in texts)
        if character_count > self._settings.max_text_characters:
            raise ValueError(
                "embedding request exceeds max text characters "
                f"({self._settings.max_text_characters})"
            )

        payload = EmbeddingRequest(input_type=input_type, texts=texts)
        response = await self._httpx_client.post(
            self._settings.api_path,
            json=payload.model_dump(mode="json"),
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            detail = "request rejected"
            try:
                response_body = response.json()
                if isinstance(response_body, dict):
                    response_detail = response_body.get("detail")
                    if isinstance(response_detail, str) and response_detail.strip():
                        detail = response_detail.strip()
            except ValueError:
                pass
            raise EmbeddingServiceRequestError(response.status_code, detail) from error

        try:
            result = EmbeddingResponse.model_validate(response.json())
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "embedding service returned an invalid response"
            ) from error
        if result.model != self._settings.expected_model:
            raise RuntimeError(
                f"embedding service returned unexpected model {result.model!r}"
            )
        if result.revision != self._settings.expected_revision:
            raise RuntimeError(
                "embedding service returned unexpected revision "
                f"{result.revision!r}"
            )
        if result.dimension != self._settings.expected_dimension:
            raise RuntimeError(
                "embedding service returned unexpected dimension "
                f"{result.dimension}; expected {self._settings.expected_dimension}"
            )
        if result.normalized is not self._settings.expected_normalized:
            raise RuntimeError(
                "embedding service returned unexpected normalization status"
            )
        if result.sparse_model != self._settings.expected_sparse_model:
            raise RuntimeError(
                "embedding service returned unexpected sparse model "
                f"{result.sparse_model!r}"
            )
        if len(result.embeddings) != len(texts):
            raise RuntimeError("embedding service returned the wrong vector count")
        if any(len(vector) != result.dimension for vector in result.embeddings):
            raise RuntimeError("embedding service returned a wrong vector length")
        return tuple(
            EmbeddingVector(dense=dense, sparse=sparse)
            for dense, sparse in zip(
                result.embeddings,
                result.sparse_embeddings,
                strict=True,
            )
        )

    async def close(self) -> None:
        """Close the injected client exactly once."""

        if self._closed:
            return
        self._closed = True
        await self._httpx_client.aclose()


__all__ = [
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EnterpriseEmbeddingServiceRepository",
]

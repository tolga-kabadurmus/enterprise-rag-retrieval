"""Provider-neutral reranking contract and models."""

from abc import abstractmethod
import math

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.repository.base import Repository


class RerankDocument(BaseModel):
    """One candidate document supplied to a reranking provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class RerankResult(BaseModel):
    """One reranked document identifier and provider relevance score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(min_length=1)
    score: float

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        """Reject non-finite provider scores."""

        if not math.isfinite(value):
            raise ValueError("reranking score must be finite")
        return value


class RerankingRepositoryBase(Repository):
    """Score and reorder retrieved candidates for one user query."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: tuple[RerankDocument, ...],
    ) -> tuple[RerankResult, ...]:
        """Return every candidate once in descending relevance order."""

"""Provider-neutral vector retrieval contract and result model."""

from abc import abstractmethod

from pydantic import BaseModel, ConfigDict, Field

from app.repository.base import Repository


class VectorSearchResult(BaseModel):
    """One scored vector match with its citation payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    point_id: str = Field(min_length=1)
    score: float
    payload: dict[str, object]


class VectorRepositoryBase(Repository):
    """Retrieve vectors without exposing the selected provider."""

    @abstractmethod
    async def search(
        self,
        query_vector: tuple[float, ...],
        limit: int,
        score_threshold: float | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Return the best dense-vector matches in descending relevance order."""

"""Use case for vectorizing a user question."""

from app.repository.embedding_service.base import EmbeddingRepository
from app.usecase.base import UseCase


class QueryEmbeddingUseCase(UseCase[str, tuple[float, ...]]):
    """Generate one query-intent vector through an injected repository."""

    def __init__(self, repo: EmbeddingRepository) -> None:
        self._repo = repo

    async def execute(self, value: str) -> tuple[float, ...]:
        """Return the single embedding generated for a non-blank question."""

        question = value.strip()
        if not question:
            raise ValueError("question must not be blank")
        vectors = await self._repo.embed((question,), input_type="query")
        if len(vectors) != 1:
            raise RuntimeError("embedding repository returned the wrong vector count")
        return vectors[0]

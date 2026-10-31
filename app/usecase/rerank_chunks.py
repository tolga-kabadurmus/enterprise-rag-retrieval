"""Use case for reranking retrieved chunks."""

from pydantic import BaseModel, ConfigDict, Field

from app.domain import RetrievedChunk
from app.repository.reranking import RerankDocument, RerankingRepositoryBase
from app.usecase.base import UseCase


class RerankChunksInput(BaseModel):
    """Question and hybrid candidates supplied to the reranking use case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str = Field(min_length=1)
    chunks: tuple[RetrievedChunk, ...]


class RerankChunksUseCase(
    UseCase[RerankChunksInput, tuple[RetrievedChunk, ...]]
):
    """Rerank hybrid candidates and return the configured final result count."""

    def __init__(self, repo: RerankingRepositoryBase, top_k: int) -> None:
        self._repo = repo
        self._top_k = top_k

    async def execute(
        self,
        value: RerankChunksInput,
    ) -> tuple[RetrievedChunk, ...]:
        """Replace hybrid scores with reranker scores and truncate to top-k."""

        if not value.chunks:
            return ()
        documents = tuple(
            RerankDocument(document_id=str(index), text=chunk.text)
            for index, chunk in enumerate(value.chunks)
        )
        results = await self._repo.rerank(value.question, documents)
        chunks_by_id = {
            str(index): chunk for index, chunk in enumerate(value.chunks)
        }
        return tuple(
            RetrievedChunk(
                text=chunks_by_id[result.document_id].text,
                score=result.score,
                source=chunks_by_id[result.document_id].source,
            )
            for result in results[: self._top_k]
        )

"""Controller for the query embedding and chunk retrieval flow."""

from app.domain import RetrievedChunk, UserQuestion
from app.usecase import (
    QueryEmbeddingUseCase,
    RerankChunksInput,
    RerankChunksUseCase,
    RetrieveChunksUseCase,
)


class RetrievalController:
    """Coordinate question vectorization and relevant-chunk retrieval."""

    def __init__(
        self,
        query_embedding_usecase: QueryEmbeddingUseCase,
        retrieve_chunks_usecase: RetrieveChunksUseCase,
        rerank_chunks_usecase: RerankChunksUseCase,
    ) -> None:
        self._query_embedding_usecase = query_embedding_usecase
        self._retrieve_chunks_usecase = retrieve_chunks_usecase
        self._rerank_chunks_usecase = rerank_chunks_usecase

    async def execute(
        self,
        question: UserQuestion,
    ) -> tuple[RetrievedChunk, ...]:
        """Embed one validated question and retrieve its relevant chunks."""

        vector = await self._query_embedding_usecase.execute(question.question)
        chunks = await self._retrieve_chunks_usecase.execute(vector)
        return await self._rerank_chunks_usecase.execute(
            RerankChunksInput(question=question.question, chunks=chunks)
        )

"""Controller for the query embedding and chunk retrieval flow."""

from app.domain import RetrievedChunk, UserQuestion
from app.usecase import QueryEmbeddingUseCase, RetrieveChunksUseCase


class RetrievalController:
    """Coordinate question vectorization and relevant-chunk retrieval."""

    def __init__(
        self,
        query_embedding_usecase: QueryEmbeddingUseCase,
        retrieve_chunks_usecase: RetrieveChunksUseCase,
    ) -> None:
        self._query_embedding_usecase = query_embedding_usecase
        self._retrieve_chunks_usecase = retrieve_chunks_usecase

    async def execute(
        self,
        question: UserQuestion,
    ) -> tuple[RetrievedChunk, ...]:
        """Embed one validated question and retrieve its relevant chunks."""

        vector = await self._query_embedding_usecase.execute(question.question)
        return await self._retrieve_chunks_usecase.execute(vector)

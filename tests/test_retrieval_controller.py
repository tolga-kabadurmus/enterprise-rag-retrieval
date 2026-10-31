"""Unit tests for retrieval orchestration."""

import unittest

from app.controller import RetrievalController
from app.domain import RetrievedChunk, SourceReference, UserQuestion
from app.repository.embedding_service import EmbeddingVector, SparseEmbedding
from app.usecase import RerankChunksInput


class FakeQueryEmbeddingUseCase:
    """Return a fixed query vector."""

    async def execute(self, question: str) -> EmbeddingVector:
        self.question = question
        return EmbeddingVector(
            dense=(0.1, 0.2),
            sparse=SparseEmbedding(indices=(5,), values=(1.0,)),
        )


class FakeRetrieveChunksUseCase:
    """Return fixed chunks for a query vector."""

    async def execute(
        self,
        vector: EmbeddingVector,
    ) -> tuple[RetrievedChunk, ...]:
        self.vector = vector
        return (
            RetrievedChunk(
                text="Leave policy context",
                score=0.9,
                source=SourceReference(
                    document_id="doc-1",
                    source_bucket="documents",
                    object_key="handbook.pdf",
                    chunk_id="chunk-1",
                    page_number=2,
                ),
            ),
        )


class FakeRerankChunksUseCase:
    """Return candidates after recording the reranking input."""

    async def execute(
        self,
        value: RerankChunksInput,
    ) -> tuple[RetrievedChunk, ...]:
        self.value = value
        return value.chunks


class RetrievalControllerTests(unittest.IsolatedAsyncioTestCase):
    """Verify embedding happens before vector retrieval."""

    async def test_coordinates_embedding_and_retrieval(self) -> None:
        embedding = FakeQueryEmbeddingUseCase()
        retrieval = FakeRetrieveChunksUseCase()
        reranking = FakeRerankChunksUseCase()
        controller = RetrievalController(embedding, retrieval, reranking)

        chunks = await controller.execute(UserQuestion(question="Leave policy?"))

        self.assertEqual(embedding.question, "Leave policy?")
        self.assertEqual(retrieval.vector.dense, (0.1, 0.2))
        self.assertEqual(reranking.value.question, "Leave policy?")
        self.assertEqual(chunks[0].source.object_key, "handbook.pdf")

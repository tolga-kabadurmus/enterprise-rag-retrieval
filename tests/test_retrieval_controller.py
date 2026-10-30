"""Unit tests for retrieval orchestration."""

import unittest

from app.controller import RetrievalController
from app.domain import RetrievedChunk, SourceReference, UserQuestion


class FakeQueryEmbeddingUseCase:
    """Return a fixed query vector."""

    async def execute(self, question: str) -> tuple[float, ...]:
        self.question = question
        return (0.1, 0.2)


class FakeRetrieveChunksUseCase:
    """Return fixed chunks for a query vector."""

    async def execute(
        self,
        vector: tuple[float, ...],
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


class RetrievalControllerTests(unittest.IsolatedAsyncioTestCase):
    """Verify embedding happens before vector retrieval."""

    async def test_coordinates_embedding_and_retrieval(self) -> None:
        embedding = FakeQueryEmbeddingUseCase()
        retrieval = FakeRetrieveChunksUseCase()
        controller = RetrievalController(embedding, retrieval)

        chunks = await controller.execute(UserQuestion(question="Leave policy?"))

        self.assertEqual(embedding.question, "Leave policy?")
        self.assertEqual(retrieval.vector, (0.1, 0.2))
        self.assertEqual(chunks[0].source.object_key, "handbook.pdf")

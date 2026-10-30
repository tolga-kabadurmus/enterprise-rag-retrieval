"""Unit tests for chunk reranking application behavior."""

import unittest

from app.domain import RetrievedChunk, SourceReference
from app.repository.reranking import (
    RerankDocument,
    RerankingRepositoryBase,
    RerankResult,
)
from app.usecase import RerankChunksInput, RerankChunksUseCase


class FakeRerankingRepository(RerankingRepositoryBase):
    """Return the second candidate first with deterministic scores."""

    async def rerank(
        self,
        query: str,
        documents: tuple[RerankDocument, ...],
    ) -> tuple[RerankResult, ...]:
        self.query = query
        self.documents = documents
        return (
            RerankResult(document_id="1", score=0.95),
            RerankResult(document_id="0", score=0.4),
        )

    async def close(self) -> None:
        return None


def chunk(text: str, chunk_id: str) -> RetrievedChunk:
    """Build one candidate fixture with citation metadata."""

    return RetrievedChunk(
        text=text,
        score=0.1,
        source=SourceReference(
            document_id="doc-1",
            source_bucket="documents",
            object_key="handbook.pdf",
            chunk_id=chunk_id,
        ),
    )


class RerankChunksUseCaseTests(unittest.IsolatedAsyncioTestCase):
    """Verify score replacement, ordering, truncation, and empty handling."""

    async def test_reranks_and_truncates_candidates(self) -> None:
        repository = FakeRerankingRepository()
        usecase = RerankChunksUseCase(repo=repository, top_k=1)

        results = await usecase.execute(
            RerankChunksInput(
                question="leave policy",
                chunks=(chunk("first", "chunk-1"), chunk("best", "chunk-2")),
            )
        )

        self.assertEqual(results[0].text, "best")
        self.assertEqual(results[0].score, 0.95)
        self.assertEqual(results[0].source.chunk_id, "chunk-2")
        self.assertEqual(repository.query, "leave policy")

    async def test_returns_empty_without_calling_repository(self) -> None:
        repository = FakeRerankingRepository()
        usecase = RerankChunksUseCase(repo=repository, top_k=5)

        results = await usecase.execute(
            RerankChunksInput(question="question", chunks=())
        )

        self.assertEqual(results, ())
        self.assertFalse(hasattr(repository, "query"))

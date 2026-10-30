"""Unit tests for vector retrieval application behavior."""

import unittest

from app.common.settings.retrieval import RetrievalSettings
from app.repository.vector import VectorRepositoryBase, VectorSearchResult
from app.usecase import RetrieveChunksUseCase


class FakeVectorRepository(VectorRepositoryBase):
    """Return configured vector results while recording search arguments."""

    def __init__(self, results: tuple[VectorSearchResult, ...]) -> None:
        self.results = results
        self.search_arguments: dict[str, object] = {}

    async def search(
        self,
        query_vector: tuple[float, ...],
        limit: int,
        score_threshold: float | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        self.search_arguments = {
            "query_vector": query_vector,
            "limit": limit,
            "score_threshold": score_threshold,
        }
        return self.results

    async def close(self) -> None:
        """Satisfy the repository lifecycle contract for this in-memory fake."""


class RetrieveChunksUseCaseTests(unittest.IsolatedAsyncioTestCase):
    """Verify configured search delegation and payload mapping."""

    async def test_retrieves_and_maps_indexed_chunk_payload(self) -> None:
        repository = FakeVectorRepository(
            (
                VectorSearchResult(
                    point_id="point-1",
                    score=0.91,
                    payload={
                        "text": "Relevant text",
                        "document_id": "doc-1",
                        "source_bucket": "documents",
                        "object_key": "handbook.pdf",
                        "chunk_id": "chunk-1",
                        "page_number": 2,
                        "chunk_index": 4,
                    },
                ),
            )
        )
        usecase = RetrieveChunksUseCase(
            repo=repository,
            settings=RetrievalSettings(top_k=3, score_threshold=0.5),
        )

        chunks = await usecase.execute((0.1, 0.2))

        self.assertEqual(chunks[0].text, "Relevant text")
        self.assertEqual(chunks[0].source.object_key, "handbook.pdf")
        self.assertEqual(repository.search_arguments["limit"], 3)
        self.assertEqual(repository.search_arguments["score_threshold"], 0.5)

    async def test_rejects_invalid_retrieval_payload(self) -> None:
        repository = FakeVectorRepository(
            (
                VectorSearchResult(
                    point_id="point-1",
                    score=0.91,
                    payload={"text": "Missing citation metadata"},
                ),
            )
        )
        usecase = RetrieveChunksUseCase(
            repo=repository,
            settings=RetrievalSettings(),
        )

        with self.assertRaisesRegex(RuntimeError, "point-1"):
            await usecase.execute((0.1,))

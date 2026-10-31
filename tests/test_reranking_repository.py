"""Unit tests for the FlashRank reranking adapter and factory."""

import unittest

from app.common.settings.reranking import RerankingProvider, RerankingSettings
from app.repository.reranking import (
    FlashRankRerankingRepository,
    RerankDocument,
    RerankingError,
    RerankingRepositoryFactory,
)


class FakeRanker:
    """Return deterministic FlashRank-compatible dictionaries."""

    def __init__(self, results: object) -> None:
        self.results = results
        self.request = None

    def rerank(self, request: object) -> object:
        self.request = request
        return self.results


class FlashRankRepositoryTests(unittest.IsolatedAsyncioTestCase):
    """Verify mapping, validation, factory selection, and cleanup."""

    async def test_reranks_documents_and_maps_scores(self) -> None:
        ranker = FakeRanker(
            [
                {"id": "1", "text": "best", "score": 0.9},
                {"id": "0", "text": "other", "score": 0.2},
            ]
        )
        repository = FlashRankRerankingRepository(
            ranker,
            RerankingSettings(),
        )

        results = await repository.rerank(
            "leave policy",
            (
                RerankDocument(document_id="0", text="other"),
                RerankDocument(document_id="1", text="best"),
            ),
        )

        self.assertEqual([result.document_id for result in results], ["1", "0"])
        self.assertEqual(results[0].score, 0.9)
        self.assertEqual(ranker.request.query, "leave policy")
        await repository.close()

    async def test_rejects_incomplete_results(self) -> None:
        repository = FlashRankRerankingRepository(
            FakeRanker([{"id": "0", "score": 0.9}]),
            RerankingSettings(),
        )

        with self.assertRaises(RerankingError):
            await repository.rerank(
                "query",
                (
                    RerankDocument(document_id="0", text="first"),
                    RerankDocument(document_id="1", text="second"),
                ),
            )

    async def test_factory_creates_flashrank_repository(self) -> None:
        factory = RerankingRepositoryFactory()
        factory.register(
            RerankingProvider.FLASHRANK,
            FlashRankRerankingRepository,
        )

        repository = factory.create(
            RerankingProvider.FLASHRANK,
            FakeRanker([]),
            RerankingSettings(),
        )

        self.assertIsInstance(repository, FlashRankRerankingRepository)
        await repository.close()

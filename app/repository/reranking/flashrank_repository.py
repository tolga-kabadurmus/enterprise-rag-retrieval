"""FlashRank implementation of local candidate reranking."""

import asyncio
from collections.abc import Mapping

from flashrank import Ranker, RerankRequest

from app.common.settings.reranking import RerankingSettings
from app.repository.reranking.base import (
    RerankDocument,
    RerankingRepositoryBase,
    RerankResult,
)
from app.repository.reranking.exceptions import RerankingError


class FlashRankRerankingRepository(RerankingRepositoryBase):
    """Run one shared FlashRank model behind a bounded thread boundary."""

    def __init__(self, ranker: Ranker, settings: RerankingSettings) -> None:
        self._ranker = ranker
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._closed = False

    async def rerank(
        self,
        query: str,
        documents: tuple[RerankDocument, ...],
    ) -> tuple[RerankResult, ...]:
        """Rerank candidates without blocking the FastAPI event loop."""

        if self._closed:
            raise RuntimeError("reranking repository is closed")
        if not documents:
            return ()

        passages = [
            {"id": document.document_id, "text": document.text}
            for document in documents
        ]
        request = RerankRequest(query=query, passages=passages)
        try:
            async with self._semaphore:
                raw_results = await asyncio.to_thread(self._ranker.rerank, request)
            return self._validate_results(raw_results, documents)
        except RerankingError:
            raise
        except Exception as error:
            raise RerankingError("FlashRank inference failed") from error

    @staticmethod
    def _validate_results(
        raw_results: object,
        documents: tuple[RerankDocument, ...],
    ) -> tuple[RerankResult, ...]:
        """Reject missing, duplicate, unknown, or malformed provider results."""

        if not isinstance(raw_results, list):
            raise RerankingError("FlashRank returned an invalid result")
        expected_ids = {document.document_id for document in documents}
        results: list[RerankResult] = []
        for raw_result in raw_results:
            if not isinstance(raw_result, Mapping):
                raise RerankingError("FlashRank returned an invalid result")
            try:
                result = RerankResult(
                    document_id=str(raw_result["id"]),
                    score=raw_result["score"],
                )
            except (KeyError, TypeError, ValueError) as error:
                raise RerankingError("FlashRank returned an invalid result") from error
            results.append(result)

        result_ids = [result.document_id for result in results]
        if len(result_ids) != len(set(result_ids)) or set(result_ids) != expected_ids:
            raise RerankingError("FlashRank returned an incomplete result")
        return tuple(results)

    async def close(self) -> None:
        """Mark the local repository closed; FlashRank owns no closeable handle."""

        self._closed = True

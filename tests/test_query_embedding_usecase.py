"""Unit tests for query embedding application behavior."""

import unittest
from typing import Literal

from app.repository.embedding_service.base import EmbeddingRepository
from app.usecase.query_embedding import QueryEmbeddingUseCase


class FakeEmbeddingRepository(EmbeddingRepository):
    """Record embedding calls without external connectivity."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str]] = []

    @property
    def max_batch_size(self) -> int:
        return 1

    @property
    def max_text_characters(self) -> int:
        return 100

    async def embed(
        self,
        texts: tuple[str, ...],
        input_type: Literal["document", "query"] = "document",
    ) -> tuple[tuple[float, ...], ...]:
        self.calls.append((texts, input_type))
        return ((0.25, 0.75),)

    async def close(self) -> None:
        return None


class QueryEmbeddingUseCaseTests(unittest.IsolatedAsyncioTestCase):
    """Verify query intent and input validation."""

    async def test_embeds_one_trimmed_question_as_query(self) -> None:
        repository = FakeEmbeddingRepository()
        usecase = QueryEmbeddingUseCase(repo=repository)

        vector = await usecase.execute("  What is RAG?  ")

        self.assertEqual(vector, (0.25, 0.75))
        self.assertEqual(repository.calls, [(("What is RAG?",), "query")])

    async def test_rejects_blank_question(self) -> None:
        usecase = QueryEmbeddingUseCase(repo=FakeEmbeddingRepository())

        with self.assertRaisesRegex(ValueError, "must not be blank"):
            await usecase.execute("   ")

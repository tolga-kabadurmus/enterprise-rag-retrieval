"""Unit tests for grounded complete and streamed answer generation."""

import unittest
from collections.abc import AsyncIterator

from app.domain import (
    LLMGenerationInput,
    RetrievedChunk,
    SourceReference,
    UserQuestion,
)
from app.repository.llm import LLMMessage, LLMRepositoryBase
from app.usecase import GenerateAnswerUseCase


class FakeLLMRepository(LLMRepositoryBase):
    """Capture generated messages without external calls."""

    def __init__(self) -> None:
        super().__init__("Configured system prompt")
        self.messages: tuple[LLMMessage, ...] = ()

    async def _generate(self, messages: tuple[LLMMessage, ...]) -> str:
        self.messages = messages
        return "Use the documented policy."

    async def _stream(
        self,
        messages: tuple[LLMMessage, ...],
    ) -> AsyncIterator[str]:
        self.messages = messages
        for delta in ("Use the ", "documented policy."):
            yield delta

    async def close(self) -> None:
        return None


class GenerateAnswerUseCaseTests(unittest.IsolatedAsyncioTestCase):
    """Verify prompt grounding, source propagation, and streaming."""

    def setUp(self) -> None:
        source = SourceReference(
            document_id="doc-1",
            source_bucket="documents",
            object_key="policy.pdf",
            chunk_id="chunk-1",
            page_number=1,
        )
        self.input = LLMGenerationInput(
            question=UserQuestion(question="What is the policy?"),
            chunks=(
                RetrievedChunk(text="The policy requires approval.", score=0.9, source=source),
                RetrievedChunk(text="Approval is documented.", score=0.8, source=source),
            ),
        )

    async def test_generates_answer_with_unique_sources(self) -> None:
        repository = FakeLLMRepository()
        usecase = GenerateAnswerUseCase(repo=repository)

        answer = await usecase.execute(self.input)

        self.assertEqual(answer.answer, "Use the documented policy.")
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(repository.messages[0].content, "Configured system prompt")
        self.assertIn("What is the policy?", repository.messages[1].content)
        self.assertIn("The policy requires approval.", repository.messages[1].content)

    async def test_streams_repository_deltas(self) -> None:
        usecase = GenerateAnswerUseCase(repo=FakeLLMRepository())

        deltas = [delta async for delta in usecase.stream(self.input)]

        self.assertEqual(deltas, ["Use the ", "documented policy."])

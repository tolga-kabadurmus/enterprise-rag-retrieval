"""Unit tests for complete and streamed answer orchestration."""

import unittest
from collections.abc import AsyncIterator

from app.controller import AnswerQuestionController
from app.domain import (
    CitationReference,
    GeneratedAnswer,
    LLMGenerationInput,
    QueryDeltaEvent,
    QueryDoneEvent,
    QueryRequest,
    QuerySourcesEvent,
    RetrievedChunk,
    SourceReference,
)


class FakeRetrievalController:
    """Return one ranked chunk and count retrieval executions."""

    def __init__(self, chunk: RetrievedChunk) -> None:
        self.chunk = chunk
        self.calls = 0

    async def execute(
        self,
        question: QueryRequest,
    ) -> tuple[RetrievedChunk, ...]:
        self.calls += 1
        self.question = question
        return (self.chunk,)


class FakeGenerateAnswerUseCase:
    """Provide deterministic complete and streamed LLM output."""

    async def execute(self, value: LLMGenerationInput) -> GeneratedAnswer:
        self.complete_input = value
        return GeneratedAnswer(
            answer="Employees receive 20 days of leave.",
            sources=self.source_references(value),
        )

    async def stream(
        self,
        value: LLMGenerationInput,
    ) -> AsyncIterator[str]:
        self.stream_input = value
        for delta in ("Employees receive ", "20 days of leave."):
            yield delta

    def source_references(
        self,
        value: LLMGenerationInput,
    ) -> tuple[CitationReference, ...]:
        return tuple(
            CitationReference(
                **chunk.source.model_dump(),
                citation_id=f"S{index}",
            )
            for index, chunk in enumerate(value.chunks, start=1)
        )


class AnswerQuestionControllerTests(unittest.IsolatedAsyncioTestCase):
    """Verify both modes share one retrieval result per request."""

    def setUp(self) -> None:
        self.source = SourceReference(
            document_id="employee-handbook",
            source_bucket="documents",
            object_key="hr/employee-handbook.pdf",
            chunk_id="employee-handbook-page-12-chunk-0",
            page_number=12,
            chunk_index=0,
        )
        self.chunk = RetrievedChunk(
            text="Employees receive 20 days of annual leave.",
            score=0.91,
            source=self.source,
        )
        self.request = QueryRequest(question="How much leave do employees get?")

    async def test_complete_answer_retrieves_once(self) -> None:
        retrieval = FakeRetrievalController(self.chunk)
        generation = FakeGenerateAnswerUseCase()
        controller = AnswerQuestionController(retrieval, generation)

        response = await controller.execute(self.request)

        self.assertEqual(retrieval.calls, 1)
        self.assertEqual(generation.complete_input.chunks, (self.chunk,))
        self.assertEqual(response.sources[0].citation_id, "S1")
        self.assertEqual(response.sources[0].object_key, self.source.object_key)

    async def test_streamed_answer_retrieves_once_before_deltas(self) -> None:
        retrieval = FakeRetrievalController(self.chunk)
        generation = FakeGenerateAnswerUseCase()
        controller = AnswerQuestionController(retrieval, generation)

        events = [event async for event in controller.stream(self.request)]

        self.assertEqual(retrieval.calls, 1)
        self.assertEqual(generation.stream_input.chunks, (self.chunk,))
        self.assertEqual(
            events,
            [
                QueryDeltaEvent(delta="Employees receive "),
                QueryDeltaEvent(delta="20 days of leave."),
                QuerySourcesEvent(
                    sources=(
                        CitationReference(
                            **self.source.model_dump(),
                            citation_id="S1",
                        ),
                    )
                ),
                QueryDoneEvent(),
            ],
        )


if __name__ == "__main__":
    unittest.main()

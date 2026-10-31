"""Controller for complete and streamed retrieval-and-answering flows."""

from collections.abc import AsyncIterator

from app.controller.retrieval import RetrievalController
from app.domain import (
    LLMGenerationInput,
    QueryDeltaEvent,
    QueryDoneEvent,
    QueryRequest,
    QueryResponse,
    QuerySourcesEvent,
    QueryStreamEvent,
)
from app.usecase import GenerateAnswerUseCase


class AnswerQuestionController:
    """Coordinate ranked-context retrieval and grounded answer generation."""

    def __init__(
        self,
        retrieval_controller: RetrievalController,
        generate_answer_usecase: GenerateAnswerUseCase,
    ) -> None:
        self._retrieval_controller = retrieval_controller
        self._generate_answer_usecase = generate_answer_usecase

    async def execute(self, request: QueryRequest) -> QueryResponse:
        """Return one complete answer with deduplicated source references."""

        generation_input = await self._generation_input(request)
        generated = await self._generate_answer_usecase.execute(
            generation_input
        )
        return QueryResponse(
            answer=generated.answer,
            sources=generated.sources,
        )

    async def stream(
        self,
        request: QueryRequest,
    ) -> AsyncIterator[QueryStreamEvent]:
        """Yield answer deltas, final citations, and a completion marker."""

        generation_input = await self._generation_input(request)
        async for delta in self._generate_answer_usecase.stream(
            generation_input
        ):
            if delta:
                yield QueryDeltaEvent(delta=delta)

        yield QuerySourcesEvent(
            sources=self._generate_answer_usecase.source_references(
                generation_input
            )
        )
        yield QueryDoneEvent()

    async def _generation_input(
        self,
        request: QueryRequest,
    ) -> LLMGenerationInput:
        """Retrieve ranked chunks and create the shared generation input."""

        chunks = await self._retrieval_controller.execute(request)
        return LLMGenerationInput(question=request, chunks=chunks)

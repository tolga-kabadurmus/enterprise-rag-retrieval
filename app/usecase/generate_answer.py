"""Use case for grounded complete and streamed answer generation."""

from collections.abc import AsyncIterator

from app.domain import GeneratedAnswer, LLMGenerationInput, SourceReference
from app.repository.llm import LLMMessage, LLMRepositoryBase
from app.usecase.base import UseCase


class GenerateAnswerUseCase(UseCase[LLMGenerationInput, GeneratedAnswer]):
    """Build grounded messages and delegate generation to one LLM repository."""

    def __init__(self, repo: LLMRepositoryBase) -> None:
        self._repo = repo

    async def execute(self, value: LLMGenerationInput) -> GeneratedAnswer:
        """Generate a complete answer with references to supplied chunks."""

        answer = await self._repo.generate(self._messages(value))
        return GeneratedAnswer(answer=answer, sources=self._sources(value))

    async def stream(self, value: LLMGenerationInput) -> AsyncIterator[str]:
        """Yield answer deltas while preserving the same grounded prompt."""

        async for delta in self._repo.stream(self._messages(value)):
            yield delta

    def _messages(self, value: LLMGenerationInput) -> tuple[LLMMessage, ...]:
        context = "\n\n".join(
            f"[Source {index}]\n{chunk.text}"
            for index, chunk in enumerate(value.chunks, start=1)
        )
        if not context:
            context = "No relevant context was retrieved."
        return (
            LLMMessage(
                role="user",
                content=f"Question:\n{value.question.question}\n\nContext:\n{context}",
            ),
        )

    def _sources(self, value: LLMGenerationInput) -> tuple[SourceReference, ...]:
        unique: dict[tuple[str, str], SourceReference] = {}
        for chunk in value.chunks:
            key = (chunk.source.document_id, chunk.source.chunk_id)
            unique.setdefault(key, chunk.source)
        return tuple(unique.values())

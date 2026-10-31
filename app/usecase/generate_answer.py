"""Use case for grounded complete and streamed answer generation."""

from collections.abc import AsyncIterator

from app.domain import (
    CitationReference,
    GeneratedAnswer,
    LLMGenerationInput,
    RetrievedChunk,
)
from app.repository.llm import LLMMessage, LLMRepositoryBase
from app.usecase.base import UseCase


class GenerateAnswerUseCase(UseCase[LLMGenerationInput, GeneratedAnswer]):
    """Build grounded messages and delegate generation to one LLM repository."""

    def __init__(self, repo: LLMRepositoryBase) -> None:
        self._repo = repo

    async def execute(self, value: LLMGenerationInput) -> GeneratedAnswer:
        """Buffer streamed provider deltas into one answer with references."""

        deltas = [delta async for delta in self.stream(value)]
        answer = "".join(deltas).strip()
        if not answer:
            raise RuntimeError("LLM provider returned an empty answer")
        return GeneratedAnswer(
            answer=answer,
            sources=self.source_references(value),
        )

    async def stream(self, value: LLMGenerationInput) -> AsyncIterator[str]:
        """Yield answer deltas while preserving the same grounded prompt."""

        async for delta in self._repo.stream(self._messages(value)):
            yield delta

    def source_references(
        self,
        value: LLMGenerationInput,
    ) -> tuple[CitationReference, ...]:
        """Return deduplicated citations for complete or streamed output."""

        return tuple(citation for citation, _ in self._cited_chunks(value))

    def _messages(self, value: LLMGenerationInput) -> tuple[LLMMessage, ...]:
        cited_chunks = self._cited_chunks(value)
        context = "\n\n".join(
            f"[{citation.citation_id}]\n{chunk.text}"
            for citation, chunk in cited_chunks
        )
        if not context:
            context = "No relevant context was retrieved."
        citation_instruction = (
            "Cite supporting claims with the matching labels such as [S1]. "
            "Do not invent citation labels."
            if cited_chunks
            else "State that the supplied context cannot answer the question."
        )
        return (
            LLMMessage(
                role="user",
                content=(
                    f"Question:\n{value.question.question}\n\n"
                    f"Context:\n{context}\n\n"
                    "Answer only from the supplied context. "
                    f"{citation_instruction}"
                ),
            ),
        )

    def _cited_chunks(
        self,
        value: LLMGenerationInput,
    ) -> tuple[tuple[CitationReference, RetrievedChunk], ...]:
        """Assign stable labels to unique chunks in retrieval order."""

        unique: dict[tuple[str, str], RetrievedChunk] = {}
        for chunk in value.chunks:
            key = (chunk.source.document_id, chunk.source.chunk_id)
            unique.setdefault(key, chunk)

        return tuple(
            (
                CitationReference(
                    **chunk.source.model_dump(),
                    citation_id=f"S{index}",
                ),
                chunk,
            )
            for index, chunk in enumerate(unique.values(), start=1)
        )

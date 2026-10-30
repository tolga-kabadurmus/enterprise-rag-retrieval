"""Immutable Pydantic models for the minimum RAG query flow."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DomainModel(BaseModel):
    """Shared strict and immutable configuration for domain records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class UserQuestion(DomainModel):
    """A validated natural-language question submitted by a user."""

    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        """Trim surrounding whitespace and reject blank questions."""

        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class SourceReference(DomainModel):
    """Citation identity derived from indexed chunk metadata."""

    document_id: str = Field(min_length=1)
    source_bucket: str = Field(min_length=1)
    object_key: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    chunk_index: int | None = Field(default=None, ge=0)


class RetrievedChunk(DomainModel):
    """One relevant text chunk and its source citation."""

    text: str = Field(min_length=1)
    score: float
    source: SourceReference


class GeneratedAnswer(DomainModel):
    """An LLM-generated answer and the sources used to ground it."""

    answer: str = Field(min_length=1)
    sources: tuple[SourceReference, ...] = ()


class LLMGenerationInput(DomainModel):
    """Question and retrieved evidence supplied to answer generation."""

    question: UserQuestion
    chunks: tuple[RetrievedChunk, ...]


class QueryRequest(UserQuestion):
    """REST request payload for the future query endpoint."""


class QueryResponse(GeneratedAnswer):
    """REST response payload for the future query endpoint."""

"""Immutable Pydantic models for the minimum RAG query flow."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DomainModel(BaseModel):
    """Shared strict and immutable configuration for domain records."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class UserQuestion(DomainModel):
    """A validated natural-language question submitted by a user."""

    question: str = Field(
        min_length=1,
        description="Natural-language question to embed and retrieve evidence for.",
        examples=["What is the annual leave policy?"],
    )

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

    document_id: str = Field(
        min_length=1,
        description="Deterministic identifier of the indexed source document.",
    )
    source_bucket: str = Field(
        min_length=1,
        description="S3-compatible bucket containing the source document.",
    )
    object_key: str = Field(
        min_length=1,
        description="Object key of the source document within its bucket.",
    )
    chunk_id: str = Field(
        min_length=1,
        description="Deterministic identifier of the retrieved chunk.",
    )
    page_number: int | None = Field(
        default=None,
        ge=1,
        description="One-based PDF page number when available.",
    )
    chunk_index: int | None = Field(
        default=None,
        ge=0,
        description="Zero-based chunk position when available.",
    )


class CitationReference(SourceReference):
    """Response-scoped citation label paired with indexed source metadata."""

    citation_id: str = Field(
        pattern=r"^S[1-9][0-9]*$",
        description="Stable label used to cite this source within one answer.",
        examples=["S1"],
    )


class RetrievedChunk(DomainModel):
    """One relevant text chunk and its source citation."""

    text: str = Field(
        min_length=1,
        description="Indexed source text supplied as potential answer context.",
    )
    score: float = Field(
        description="Relevance score after retrieval and reranking."
    )
    source: SourceReference = Field(
        description="Citation metadata for the source text."
    )


class GeneratedAnswer(DomainModel):
    """An LLM-generated answer and the sources used to ground it."""

    answer: str = Field(
        min_length=1,
        description="Complete LLM answer grounded in the retrieved context.",
        examples=["Employees receive 20 days of annual leave [S1]."],
    )
    sources: tuple[CitationReference, ...] = Field(
        default=(),
        description=(
            "Deduplicated sources whose citation IDs are valid for this answer."
        ),
    )


class LLMGenerationInput(DomainModel):
    """Question and retrieved evidence supplied to answer generation."""

    question: UserQuestion
    chunks: tuple[RetrievedChunk, ...]


class QueryRequest(UserQuestion):
    """REST request payload for retrieval operations."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [{"question": "What is the annual leave policy?"}]
        },
    )


class QueryResponse(GeneratedAnswer):
    """Complete REST response containing a grounded answer and citations."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "answer": "Employees receive 20 days of annual leave [S1].",
                    "sources": [
                        {
                            "citation_id": "S1",
                            "document_id": "employee-handbook",
                            "source_bucket": "documents",
                            "object_key": "hr/employee-handbook.pdf",
                            "chunk_id": "employee-handbook-page-12-chunk-0",
                            "page_number": 12,
                            "chunk_index": 0,
                        }
                    ],
                }
            ]
        },
    )


class QueryDeltaEvent(DomainModel):
    """One incremental answer fragment in a streamed query response."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"event": "delta", "delta": "Employees receive "}
            ]
        },
    )

    event: Literal["delta"] = "delta"
    delta: str = Field(
        min_length=1,
        description="Next non-empty fragment of generated answer text.",
    )


class QuerySourcesEvent(DomainModel):
    """Final source references emitted after all answer fragments."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "event": "sources",
                    "sources": [
                        {
                            "citation_id": "S1",
                            "document_id": "employee-handbook",
                            "source_bucket": "documents",
                            "object_key": "hr/employee-handbook.pdf",
                            "chunk_id": "employee-handbook-page-12-chunk-0",
                            "page_number": 12,
                            "chunk_index": 0,
                        }
                    ],
                }
            ]
        },
    )

    event: Literal["sources"] = "sources"
    sources: tuple[CitationReference, ...] = Field(
        default=(),
        description="Final citations for the streamed answer.",
    )


class QueryErrorEvent(DomainModel):
    """Reserved sanitized error payload for a stream that already started."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "event": "error",
                    "code": "stream_failed",
                    "detail": "answer streaming could not be completed",
                }
            ]
        },
    )

    event: Literal["error"] = "error"
    code: Literal["stream_failed"] = "stream_failed"
    detail: str = Field(
        min_length=1,
        description="Sanitized client-safe explanation without internal details.",
    )


class QueryDoneEvent(DomainModel):
    """Terminal marker indicating that a streamed query is complete."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"event": "done"}]},
    )

    event: Literal["done"] = "done"


QueryStreamEvent = (
    QueryDeltaEvent | QuerySourcesEvent | QueryErrorEvent | QueryDoneEvent
)

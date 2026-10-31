"""Unit tests for core RAG domain and API payload models."""

import unittest

from pydantic import ValidationError

from app.domain import (
    CitationReference,
    GeneratedAnswer,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    SourceReference,
)


class DomainModelTests(unittest.TestCase):
    """Verify normalization, nesting, validation, and transport shape."""

    def setUp(self) -> None:
        self.source = SourceReference(
            document_id="doc-1",
            source_bucket="documents",
            object_key="policies/travel.pdf",
            chunk_id="chunk-3",
            page_number=2,
            chunk_index=3,
        )

    def test_query_request_normalizes_question(self) -> None:
        request = QueryRequest(question="  What is the travel policy?  ")

        self.assertEqual(request.question, "What is the travel policy?")
        self.assertEqual(
            request.model_dump(mode="json"),
            {"question": "What is the travel policy?"},
        )

    def test_blank_question_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            QueryRequest(question="   ")

    def test_retrieved_chunk_preserves_citation_metadata(self) -> None:
        chunk = RetrievedChunk(text="Travel is pre-approved.", score=0.92, source=self.source)

        self.assertEqual(chunk.source.object_key, "policies/travel.pdf")
        self.assertEqual(chunk.source.page_number, 2)

    def test_generated_answer_and_api_response_share_contract(self) -> None:
        cited_source = CitationReference(
            **self.source.model_dump(),
            citation_id="S1",
        )
        answer = GeneratedAnswer(
            answer="Travel requires approval [S1].",
            sources=(cited_source,),
        )
        response = QueryResponse(**answer.model_dump())

        self.assertEqual(response.answer, answer.answer)
        self.assertEqual(response.sources, answer.sources)

    def test_citation_id_uses_response_scoped_label(self) -> None:
        citation = CitationReference(
            **self.source.model_dump(),
            citation_id="S1",
        )

        self.assertEqual(citation.citation_id, "S1")

        with self.assertRaises(ValidationError):
            CitationReference(
                **self.source.model_dump(),
                citation_id="source-one",
            )

    def test_source_page_is_optional_but_one_based_when_present(self) -> None:
        without_page = self.source.model_copy(update={"page_number": None})
        self.assertIsNone(without_page.page_number)

        with self.assertRaises(ValidationError):
            SourceReference(
                document_id="doc-1",
                source_bucket="documents",
                object_key="policy.txt",
                chunk_id="chunk-1",
                page_number=0,
            )

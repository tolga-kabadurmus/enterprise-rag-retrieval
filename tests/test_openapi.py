"""Contract tests for generated OpenAPI and interactive documentation routes."""

import unittest

from app.main import create_app


class OpenApiContractTests(unittest.TestCase):
    """Keep the retrieval-service contract complete and discoverable."""

    def setUp(self) -> None:
        self.application = create_app()
        self.schema = self.application.openapi()

    def test_exposes_swagger_redoc_and_openapi_routes(self) -> None:
        """Expose both interactive renderers and the machine-readable schema."""

        paths = {route.path for route in self.application.routes}

        self.assertTrue({"/docs", "/redoc", "/openapi.json"} <= paths)

    def test_documents_service_metadata_and_tags(self) -> None:
        """Identify the renamed service and group its public operations."""

        self.assertEqual(
            self.schema["info"]["title"],
            "Enterprise RAG Retrieval Service",
        )
        self.assertIn("hybrid Qdrant", self.schema["info"]["description"])
        self.assertEqual(
            {tag["name"] for tag in self.schema["tags"]},
            {"query", "retrieval", "embeddings"},
        )
        self.assertNotIn("securitySchemes", self.schema["components"])

    def test_documents_retrieval_operation_and_failures(self) -> None:
        """Publish stable IDs, summaries, schemas, and dependency failures."""

        operation = self.schema["paths"]["/v1/retrieval"]["post"]

        self.assertEqual(operation["operationId"], "retrieve_chunks_v1")
        self.assertEqual(operation["tags"], ["retrieval"])
        self.assertTrue(operation["summary"])
        self.assertTrue(operation["description"])
        self.assertEqual(
            set(operation["responses"]),
            {"200", "422", "429", "502", "503", "504"},
        )

    def test_documents_query_embedding_operation(self) -> None:
        """Describe the diagnostic embedding endpoint separately from retrieval."""

        operation = self.schema["paths"]["/v1/embeddings"]["post"]

        self.assertEqual(operation["operationId"], "create_query_embedding_v1")
        self.assertEqual(operation["tags"], ["embeddings"])
        self.assertIn("diagnostic", operation["description"])

    def test_documents_complete_and_streamed_query_operations(self) -> None:
        """Publish separate JSON and SSE answer contracts."""

        complete = self.schema["paths"]["/v1/query"]["post"]
        streamed = self.schema["paths"]["/v1/query/stream"]["post"]

        self.assertEqual(complete["operationId"], "answer_query_v1")
        self.assertEqual(streamed["operationId"], "stream_query_v1")
        self.assertEqual(complete["tags"], ["query"])
        self.assertEqual(
            set(complete["responses"]),
            {"200", "422", "429", "502", "503", "504"},
        )
        self.assertEqual(
            set(streamed["responses"]),
            {"200", "422", "429", "502", "503", "504"},
        )
        self.assertIn(
            "example",
            complete["responses"]["200"]["content"]["application/json"],
        )
        self.assertIn(
            "text/event-stream",
            streamed["responses"]["200"]["content"],
        )
        stream_content = streamed["responses"]["200"]["content"][
            "text/event-stream"
        ]
        self.assertEqual(stream_content["schema"]["type"], "string")
        self.assertEqual(
            set(stream_content["examples"]),
            {"successful_stream", "error_stream"},
        )
        self.assertEqual(
            set(streamed["x-sse-event-schemas"]),
            {"delta", "sources", "error", "done"},
        )
        self.assertIn("sanitized error event", streamed["description"])
        self.assertIn("Client disconnection cancels", streamed["description"])

    def test_documents_every_query_error_response(self) -> None:
        """Describe validation and sanitized dependency errors with examples."""

        for path in ("/v1/query", "/v1/query/stream"):
            responses = self.schema["paths"][path]["post"]["responses"]
            for status_code in ("422", "429", "502", "503", "504"):
                with self.subTest(path=path, status_code=status_code):
                    response = responses[status_code]
                    self.assertTrue(response["description"])
                    self.assertIn("application/json", response["content"])
                    self.assertIn(
                        "example",
                        response["content"]["application/json"],
                    )

    def test_documents_payload_fields_and_examples(self) -> None:
        """Keep request, result, and citation schemas self-explanatory."""

        schemas = self.schema["components"]["schemas"]

        self.assertTrue(schemas["QueryRequest"]["examples"])
        self.assertTrue(
            schemas["QueryRequest"]["properties"]["question"]["description"]
        )
        self.assertTrue(schemas["RetrievalResponse"]["examples"])
        self.assertTrue(
            schemas["SourceReference"]["properties"]["object_key"]["description"]
        )
        self.assertTrue(
            schemas["CitationReference"]["properties"]["citation_id"][
                "description"
            ]
        )


if __name__ == "__main__":
    unittest.main()

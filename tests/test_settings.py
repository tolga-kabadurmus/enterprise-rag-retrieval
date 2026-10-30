"""Unit tests for validated query-service configuration."""

import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.common.settings import LLMProvider, LLMSettings, VectorSettings, load_settings


VALID_ENV = {
    "EMBEDDING_SERVICE_URL": "http://embedding:8081",
    "QDRANT_URL": "http://qdrant:6333",
    "QDRANT_COLLECTION_NAME": "chunks",
    "LLM_PROVIDER": "openai_compatible",
    "LLM_BASE_URL": "https://llm.example.test/v1",
    "LLM_MODEL": "test-model",
}


class SettingsTests(unittest.TestCase):
    """Verify required values, bounds, aggregation, and secret handling."""

    def test_load_settings_groups_valid_environment(self) -> None:
        with patch.dict(os.environ, VALID_ENV, clear=True):
            settings = load_settings()

        self.assertEqual(settings.api.port, 8082)
        self.assertEqual(settings.retrieval.top_k, 5)
        self.assertEqual(settings.vector.collection_name, "chunks")
        self.assertEqual(settings.llm.model, "test-model")
        self.assertIn("supplied context", settings.llm.system_prompt)
        self.assertEqual(settings.embedding.url, "http://embedding:8081")

    def test_llm_provider_is_required(self) -> None:
        environment = VALID_ENV | {"LLM_PROVIDER": ""}
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(ValidationError):
                LLMSettings()

    def test_vector_timeout_is_bounded(self) -> None:
        environment = VALID_ENV | {"QDRANT_TIMEOUT_SECONDS": "0"}
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(ValidationError):
                VectorSettings()

    def test_llm_api_key_is_masked(self) -> None:
        environment = VALID_ENV | {"LLM_API_KEY": "private-value"}
        with patch.dict(os.environ, environment, clear=True):
            settings = LLMSettings()

        self.assertNotIn("private-value", repr(settings))

    def test_native_provider_requires_api_key(self) -> None:
        for provider in (LLMProvider.ANTHROPIC, LLMProvider.COHERE):
            environment = VALID_ENV | {
                "LLM_PROVIDER": provider.value,
                "LLM_API_KEY": "",
            }
            with self.subTest(provider=provider):
                with patch.dict(os.environ, environment, clear=True):
                    with self.assertRaises(ValidationError):
                        LLMSettings()

    def test_blank_system_prompt_is_rejected(self) -> None:
        environment = VALID_ENV | {"LLM_SYSTEM_PROMPT": "   "}
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(ValidationError):
                LLMSettings()

    def test_blank_optional_values_use_none(self) -> None:
        environment = VALID_ENV | {
            "QDRANT_API_KEY": "",
            "RAG_SCORE_THRESHOLD": "",
            "LLM_API_KEY": "",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = load_settings()

        self.assertIsNone(settings.vector.api_key)
        self.assertIsNone(settings.retrieval.score_threshold)
        self.assertIsNone(settings.llm.api_key)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for supported SDK-backed LLM repositories."""

import unittest
from unittest.mock import AsyncMock, Mock

from app.common.settings.llm import LLMProvider, LLMSettings
from app.repository.llm import (
    AnthropicLLMRepository,
    CohereLLMRepository,
    LLMMessage,
    LLMRepositoryFactory,
    OpenAILLMRepository,
)


class OpenAILLMRepositoryTests(unittest.IsolatedAsyncioTestCase):
    """Verify complete generation, streaming, factory selection, and cleanup."""

    def setUp(self) -> None:
        self.client = Mock()
        self.client.close = AsyncMock()
        self.client.chat.completions.create = AsyncMock()
        self.settings = LLMSettings(
            provider=LLMProvider.OPENAI_COMPATIBLE,
            base_url="https://llm.example.test/v1",
            model="test-model",
            system_prompt="Configured system prompt",
        )
        self.messages = (LLMMessage(role="user", content="Question"),)

    async def test_generates_complete_answer(self) -> None:
        self.client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="  Grounded answer.  "))]
        )
        repository = OpenAILLMRepository(self.client, self.settings)

        answer = await repository.generate(self.messages)

        self.assertEqual(answer, "Grounded answer.")
        kwargs = self.client.chat.completions.create.await_args.kwargs
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["messages"][0]["role"], "system")
        self.assertEqual(
            kwargs["messages"][0]["content"],
            "Configured system prompt",
        )
        self.assertEqual(kwargs["max_completion_tokens"], 1_024)
        self.assertNotIn("max_tokens", kwargs)
        self.assertNotIn("temperature", kwargs)
        self.assertFalse(kwargs["stream"])

    async def test_streams_non_empty_text_deltas(self) -> None:
        async def events():
            for content in ("Grounded", None, " answer."):
                yield Mock(choices=[Mock(delta=Mock(content=content))])

        self.client.chat.completions.create.return_value = events()
        repository = OpenAILLMRepository(self.client, self.settings)

        deltas = [delta async for delta in repository.stream(self.messages)]

        self.assertEqual(deltas, ["Grounded", " answer."])
        kwargs = self.client.chat.completions.create.await_args.kwargs
        self.assertEqual(kwargs["max_completion_tokens"], 1_024)
        self.assertNotIn("max_tokens", kwargs)
        self.assertNotIn("temperature", kwargs)
        self.assertTrue(kwargs["stream"])

    async def test_factory_creates_registered_openai_repository(self) -> None:
        factory = LLMRepositoryFactory()
        factory.register(LLMProvider.OPENAI_COMPATIBLE, OpenAILLMRepository)

        repository = factory.create(
            LLMProvider.OPENAI_COMPATIBLE,
            self.client,
            self.settings,
        )

        self.assertIsInstance(repository, OpenAILLMRepository)
        await repository.close()
        self.client.close.assert_awaited_once()


class AnthropicLLMRepositoryTests(unittest.IsolatedAsyncioTestCase):
    """Verify Anthropic message conversion, generation, and streaming."""

    def setUp(self) -> None:
        self.client = Mock()
        self.client.close = AsyncMock()
        self.client.messages.create = AsyncMock()
        self.settings = LLMSettings(
            provider=LLMProvider.ANTHROPIC,
            base_url="https://api.anthropic.test",
            model="claude-test",
            api_key="secret",
            system_prompt="Configured system prompt",
        )
        self.messages = (
            LLMMessage(role="system", content="Ground answers."),
            LLMMessage(role="user", content="Question"),
        )

    async def test_generates_answer_with_separate_system_prompt(self) -> None:
        self.client.messages.create.return_value = Mock(
            content=[Mock(type="text", text=" Grounded answer. ")]
        )
        repository = AnthropicLLMRepository(self.client, self.settings)

        answer = await repository.generate(self.messages)

        self.assertEqual(answer, "Grounded answer.")
        kwargs = self.client.messages.create.await_args.kwargs
        self.assertEqual(kwargs["system"], "Configured system prompt")
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "Question"}])

    async def test_streams_text_deltas(self) -> None:
        async def events():
            yield Mock(
                type="content_block_delta",
                delta=Mock(type="text_delta", text="Answer"),
            )

        self.client.messages.create.return_value = events()
        repository = AnthropicLLMRepository(self.client, self.settings)

        deltas = [delta async for delta in repository.stream(self.messages)]

        self.assertEqual(deltas, ["Answer"])


class CohereLLMRepositoryTests(unittest.IsolatedAsyncioTestCase):
    """Verify Cohere V2 generation, streaming, and pool cleanup."""

    def setUp(self) -> None:
        self.client = Mock()
        self.client.chat = AsyncMock()
        self.http_client = Mock()
        self.http_client.aclose = AsyncMock()
        self.settings = LLMSettings(
            provider=LLMProvider.COHERE,
            base_url="https://api.cohere.test",
            model="command-test",
            api_key="secret",
            system_prompt="Configured system prompt",
        )
        self.messages = (LLMMessage(role="user", content="Question"),)

    async def test_generates_complete_answer(self) -> None:
        self.client.chat.return_value = Mock(
            message=Mock(content=[Mock(type="text", text=" Grounded answer. ")])
        )
        repository = CohereLLMRepository(
            self.client,
            self.settings,
            self.http_client,
        )

        answer = await repository.generate(self.messages)

        self.assertEqual(answer, "Grounded answer.")
        self.assertEqual(
            self.client.chat.await_args.kwargs["messages"],
            [
                {"role": "system", "content": "Configured system prompt"},
                {"role": "user", "content": "Question"},
            ],
        )

    async def test_streams_content_deltas_and_closes_pool(self) -> None:
        async def events(**kwargs: object):
            del kwargs
            yield Mock(
                type="content-delta",
                delta=Mock(message=Mock(content=Mock(text="Answer"))),
            )

        self.client.chat_stream = events
        repository = CohereLLMRepository(
            self.client,
            self.settings,
            self.http_client,
        )

        deltas = [delta async for delta in repository.stream(self.messages)]
        await repository.close()

        self.assertEqual(deltas, ["Answer"])
        self.http_client.aclose.assert_awaited_once()

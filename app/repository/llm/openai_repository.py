"""OpenAI SDK implementation of configurable LLM generation."""

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from app.common.settings.llm import LLMSettings
from app.repository.llm.base import LLMMessage, LLMRepositoryBase


class OpenAILLMRepository(LLMRepositoryBase):
    """Use an injected process-wide OpenAI SDK client for chat generation."""

    def __init__(self, client: AsyncOpenAI, settings: LLMSettings) -> None:
        super().__init__(settings.system_prompt)
        self._client = client
        self._settings = settings
        self._closed = False

    async def _generate(self, messages: tuple[LLMMessage, ...]) -> str:
        """Return a validated non-empty Chat Completions response."""

        self._ensure_available(messages)
        response = await self._client.chat.completions.create(
            **self._request_arguments(messages),
            stream=False,
        )
        if not response.choices:
            raise RuntimeError("LLM provider returned no choices")
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("LLM provider returned an empty answer")
        return content.strip()

    async def _stream(
        self,
        messages: tuple[LLMMessage, ...],
    ) -> AsyncIterator[str]:
        """Yield non-empty Chat Completions text deltas."""

        self._ensure_available(messages)
        response_stream = await self._client.chat.completions.create(
            **self._request_arguments(messages),
            stream=True,
        )
        async for event in response_stream:
            if not event.choices:
                continue
            content = event.choices[0].delta.content
            if content:
                yield content

    def _request_arguments(self, messages: tuple[LLMMessage, ...]) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [message.model_dump(mode="json") for message in messages],
            "max_completion_tokens": self._settings.max_output_tokens,
        }
        if self._settings.temperature is not None:
            arguments["temperature"] = self._settings.temperature
        return arguments

    def _ensure_available(self, messages: tuple[LLMMessage, ...]) -> None:
        if self._closed:
            raise RuntimeError("LLM repository is closed")
        if not messages:
            raise ValueError("at least one LLM message is required")

    async def close(self) -> None:
        """Close the injected SDK client and its HTTP connection pool once."""

        if self._closed:
            return
        self._closed = True
        await self._client.close()

"""Anthropic SDK implementation of the LLM repository contract."""

from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from app.common.settings.llm import LLMSettings
from app.repository.llm.base import LLMMessage, LLMRepositoryBase


class AnthropicLLMRepository(LLMRepositoryBase):
    """Generate text with an injected process-wide Anthropic SDK client."""

    def __init__(self, client: AsyncAnthropic, settings: LLMSettings) -> None:
        super().__init__(settings.system_prompt)
        self._client = client
        self._settings = settings
        self._closed = False

    async def _generate(self, messages: tuple[LLMMessage, ...]) -> str:
        """Return all non-empty text blocks from one Anthropic message."""

        self._ensure_available(messages)
        response = await self._client.messages.create(
            **self._request_arguments(messages),
            stream=False,
        )
        answer = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
            and getattr(block, "text", None)
        ).strip()
        if not answer:
            raise RuntimeError("LLM provider returned an empty answer")
        return answer

    async def _stream(
        self,
        messages: tuple[LLMMessage, ...],
    ) -> AsyncIterator[str]:
        """Yield non-empty Anthropic text deltas."""

        self._ensure_available(messages)
        response_stream = await self._client.messages.create(
            **self._request_arguments(messages),
            stream=True,
        )
        async for event in response_stream:
            delta = getattr(event, "delta", None)
            if (
                getattr(event, "type", None) == "content_block_delta"
                and getattr(delta, "type", None) == "text_delta"
                and delta.text
            ):
                yield delta.text

    def _request_arguments(self, messages: tuple[LLMMessage, ...]) -> dict[str, Any]:
        system = "\n\n".join(
            message.content for message in messages if message.role == "system"
        )
        arguments: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [
                message.model_dump(mode="json")
                for message in messages
                if message.role != "system"
            ],
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_output_tokens,
        }
        if system:
            arguments["system"] = system
        return arguments

    def _ensure_available(self, messages: tuple[LLMMessage, ...]) -> None:
        if self._closed:
            raise RuntimeError("LLM repository is closed")
        if not messages:
            raise ValueError("at least one LLM message is required")
        if not any(message.role != "system" for message in messages):
            raise ValueError("Anthropic requires at least one user or assistant message")

    async def close(self) -> None:
        """Close the injected Anthropic client and HTTP pool once."""

        if self._closed:
            return
        self._closed = True
        await self._client.close()

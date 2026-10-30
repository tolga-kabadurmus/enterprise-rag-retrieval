"""Cohere SDK implementation of the LLM repository contract."""

from collections.abc import AsyncIterator
from typing import Any

import httpx
from cohere import AsyncClientV2

from app.common.settings.llm import LLMSettings
from app.repository.llm.base import LLMMessage, LLMRepositoryBase


class CohereLLMRepository(LLMRepositoryBase):
    """Generate text with an injected Cohere V2 client and HTTP pool."""

    def __init__(
        self,
        client: AsyncClientV2,
        settings: LLMSettings,
        http_client: httpx.AsyncClient,
    ) -> None:
        super().__init__(settings.system_prompt)
        self._client = client
        self._settings = settings
        self._http_client = http_client
        self._closed = False

    async def _generate(self, messages: tuple[LLMMessage, ...]) -> str:
        """Return all non-empty Cohere response text blocks."""

        self._ensure_available(messages)
        response = await self._client.chat(**self._request_arguments(messages))
        answer = "".join(
            block.text
            for block in response.message.content
            if getattr(block, "type", "text") == "text"
            and getattr(block, "text", None)
        ).strip()
        if not answer:
            raise RuntimeError("LLM provider returned an empty answer")
        return answer

    async def _stream(
        self,
        messages: tuple[LLMMessage, ...],
    ) -> AsyncIterator[str]:
        """Yield non-empty Cohere V2 content deltas."""

        self._ensure_available(messages)
        async for event in self._client.chat_stream(
            **self._request_arguments(messages)
        ):
            if getattr(event, "type", None) != "content-delta":
                continue
            content = event.delta.message.content
            if content.text:
                yield content.text

    def _request_arguments(self, messages: tuple[LLMMessage, ...]) -> dict[str, Any]:
        return {
            "model": self._settings.model,
            "messages": [message.model_dump(mode="json") for message in messages],
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_output_tokens,
        }

    def _ensure_available(self, messages: tuple[LLMMessage, ...]) -> None:
        if self._closed:
            raise RuntimeError("LLM repository is closed")
        if not messages:
            raise ValueError("at least one LLM message is required")

    async def close(self) -> None:
        """Close the injected Cohere HTTP connection pool once."""

        if self._closed:
            return
        self._closed = True
        await self._http_client.aclose()

import json
import time
from typing import AsyncGenerator

import httpx

from .base import (
    ChatResponse,
    LLMProvider,
    ProviderError,
    build_messages,
    read_stream_error,
    try_read_body,
)

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_TIMEOUT = 180.0
MODELS_TIMEOUT = 30.0


class OpenRouterProvider(LLMProvider):
    """Provider for OpenRouter's unified API."""

    def __init__(self, api_key: str, client: httpx.AsyncClient):
        self.api_key = api_key
        self._client = client

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/fusion-app",
            "X-Title": "Fusion App",
        }

    def _body(self, messages: list[dict], model: str, stream: bool, **kwargs) -> dict:
        body = {
            "model": model,
            "messages": build_messages(messages, kwargs.get("system_prompt")),
            "stream": stream,
        }
        if kwargs.get("max_tokens"):
            body["max_tokens"] = kwargs["max_tokens"]
        return body

    async def chat(
        self,
        messages: list[dict],
        model: str,
        **kwargs,
    ) -> ChatResponse:
        if not self.api_key:
            return ChatResponse(
                content="", model=model, error="OpenRouter API key not set"
            )

        start = time.monotonic()
        try:
            resp = await self._client.post(
                OPENROUTER_CHAT_URL,
                headers=self._headers(),
                json=self._body(messages, model, stream=False, **kwargs),
                timeout=DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.monotonic() - start) * 1000

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage")
            return ChatResponse(
                content=content, model=model, latency_ms=elapsed, usage=usage
            )
        except httpx.HTTPStatusError as e:
            detail = await try_read_body(e.response)
            elapsed = (time.monotonic() - start) * 1000
            return ChatResponse(
                content="",
                model=model,
                latency_ms=elapsed,
                error=f"HTTP {e.response.status_code}: {detail}",
            )
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return ChatResponse(
                content="", model=model, latency_ms=elapsed, error=str(e)
            )

    async def chat_stream(
        self,
        messages: list[dict],
        model: str,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Yield plain-text content deltas. Raises ProviderError on failure."""
        if not self.api_key:
            raise ProviderError("OpenRouter API key not set")

        async with self._client.stream(
            "POST",
            OPENROUTER_CHAT_URL,
            headers=self._headers(),
            json=self._body(messages, model, stream=True, **kwargs),
            timeout=DEFAULT_TIMEOUT,
        ) as resp:
            if resp.status_code >= 400:
                detail = await read_stream_error(resp)
                raise ProviderError(f"HTTP {resp.status_code}: {detail}")
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    yield delta

    async def list_models(self) -> list[dict]:
        if not self.api_key:
            return []
        try:
            resp = await self._client.get(
                OPENROUTER_MODELS_URL, headers=self._headers(), timeout=MODELS_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])
        except Exception:
            return []

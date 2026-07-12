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

DEFAULT_BASE_URL = "http://localhost:11434"
# Backstop only — slot_timeout (panel._chat_with_timeout) is the real cap and
# must stay below this, or httpx aborts first with a confusing error.
DEFAULT_TIMEOUT = 1200.0
MODELS_TIMEOUT = 30.0


class OllamaProvider(LLMProvider):
    """Provider for local Ollama instance."""

    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self.base_url = base_url.rstrip("/")
        self._client = client

    def _chat_url(self) -> str:
        return f"{self.base_url}/api/chat"

    def _tags_url(self) -> str:
        return f"{self.base_url}/api/tags"

    def _body(self, messages: list[dict], model: str, stream: bool, **kwargs) -> dict:
        body = {
            "model": model,
            "messages": build_messages(messages, kwargs.get("system_prompt")),
            "stream": stream,
        }
        if kwargs.get("max_tokens"):
            body["options"] = {"num_predict": kwargs["max_tokens"]}
        return body

    async def chat(
        self,
        messages: list[dict],
        model: str,
        **kwargs,
    ) -> ChatResponse:
        start = time.monotonic()
        try:
            resp = await self._client.post(
                self._chat_url(),
                json=self._body(messages, model, stream=False, **kwargs),
                timeout=DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.monotonic() - start) * 1000

            content = data["message"]["content"]
            return ChatResponse(content=content, model=model, latency_ms=elapsed)
        except httpx.HTTPStatusError as e:
            detail = await try_read_body(e.response)
            elapsed = (time.monotonic() - start) * 1000
            return ChatResponse(
                content="",
                model=model,
                latency_ms=elapsed,
                error=f"HTTP {e.response.status_code}: {detail}",
            )
        except httpx.ConnectError:
            elapsed = (time.monotonic() - start) * 1000
            return ChatResponse(
                content="",
                model=model,
                latency_ms=elapsed,
                error=f"Cannot connect to Ollama at {self.base_url}",
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
        try:
            async with self._client.stream(
                "POST",
                self._chat_url(),
                json=self._body(messages, model, stream=True, **kwargs),
                timeout=DEFAULT_TIMEOUT,
            ) as resp:
                if resp.status_code >= 400:
                    detail = await read_stream_error(resp)
                    raise ProviderError(f"HTTP {resp.status_code}: {detail}")
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("error"):
                        raise ProviderError(str(data["error"]))
                    delta = (data.get("message") or {}).get("content")
                    if delta:
                        yield delta
                    if data.get("done"):
                        break
        except httpx.ConnectError:
            raise ProviderError(f"Cannot connect to Ollama at {self.base_url}")

    async def list_models(self) -> list[dict]:
        try:
            resp = await self._client.get(self._tags_url(), timeout=MODELS_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data.get("models", [])
        except Exception:
            return []

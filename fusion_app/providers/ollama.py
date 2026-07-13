import json
import time
from typing import AsyncGenerator, Optional

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

    @staticmethod
    def _to_wire_message(msg: dict) -> dict:
        """Translate one normalized message to Ollama's wire format.

        Kept in one place so a format drift upstream is a one-function fix.
        Ollama tool results use `tool_name` (no call ids); assistant
        tool_calls carry `arguments` as a plain object.
        """
        if msg.get("role") == "tool":
            return {
                "role": "tool",
                "content": msg.get("content", ""),
                "tool_name": msg.get("name", ""),
            }
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            return {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": [
                    {
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": tc.get("arguments") or {},
                        }
                    }
                    for tc in msg["tool_calls"]
                ],
            }
        return msg

    @staticmethod
    def _parse_tool_calls(message: dict) -> Optional[list]:
        """Normalize Ollama tool_calls: arguments arrive as an object (guard
        for str-encoded JSON anyway); Ollama has no call ids, synthesize them."""
        raw = message.get("tool_calls")
        if not raw:
            return None
        calls = []
        for i, tc in enumerate(raw):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = None
            if not isinstance(args, dict):
                args = args if args is None else None
            calls.append({"id": f"call_{i}", "name": fn.get("name", ""), "arguments": args})
        return calls

    def _body(
        self,
        messages: list[dict],
        model: str,
        stream: bool,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> dict:
        msgs = build_messages(messages, kwargs.get("system_prompt"))
        body = {
            "model": model,
            "messages": [self._to_wire_message(m) for m in msgs],
            "stream": stream,
        }
        if tools:
            body["tools"] = tools
        if kwargs.get("max_tokens"):
            body["options"] = {"num_predict": kwargs["max_tokens"]}
        return body

    async def chat(
        self,
        messages: list[dict],
        model: str,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> ChatResponse:
        start = time.monotonic()
        try:
            resp = await self._client.post(
                self._chat_url(),
                json=self._body(messages, model, stream=False, tools=tools, **kwargs),
                timeout=DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.monotonic() - start) * 1000

            message = data.get("message") or {}
            # Content is empty/absent when the model returns tool_calls.
            content = message.get("content") or ""
            return ChatResponse(
                content=content,
                model=model,
                latency_ms=elapsed,
                tool_calls=self._parse_tool_calls(message),
            )
        except httpx.HTTPStatusError as e:
            detail = await try_read_body(e.response)
            elapsed = (time.monotonic() - start) * 1000
            # Models without tool support 400 with a "does not support tools"
            # message — retry once without tools so the request still succeeds.
            if (
                tools
                and e.response.status_code == 400
                and "does not support tools" in detail.lower()
            ):
                retry = await self.chat(messages, model, tools=None, **kwargs)
                retry.warning = f"model '{model}' does not support tools; ran without them"
                return retry
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

    async def show_capabilities(self, model: str) -> list[str]:
        """Return the model's capability list from /api/show (e.g. ["completion", "tools"])."""
        resp = await self._client.post(
            f"{self.base_url}/api/show", json={"model": model}, timeout=MODELS_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json().get("capabilities", []) or []

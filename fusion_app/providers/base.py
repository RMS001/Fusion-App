import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import httpx


class ProviderError(Exception):
    """Raised when an upstream provider request fails (bad status, bad key, unreachable)."""


@dataclass
class ChatResponse:
    """Unified response from any LLM provider."""
    content: str
    model: str
    latency_ms: float = 0.0
    usage: Optional[dict] = None
    error: Optional[str] = None
    # Normalized tool calls regardless of provider:
    # [{"id": str|None, "name": str, "arguments": dict|None}]
    # arguments is None when the provider returned unparseable JSON.
    tool_calls: Optional[list] = None
    # Non-fatal notice (e.g. "model does not support tools; ran without").
    warning: Optional[str] = None
    # Tool-loop trace attached by PanelManager when this response was
    # produced through _chat_with_tools.
    tool_trace: Optional[list] = None


def build_messages(messages: list[dict], system_prompt: Optional[str] = None) -> list[dict]:
    """Prepend an optional system prompt to a copy of the message list."""
    msgs = list(messages)
    if system_prompt:
        msgs.insert(0, {"role": "system", "content": system_prompt})
    return msgs


async def try_read_body(response: httpx.Response) -> str:
    """Extract a short error detail from an already-read response."""
    try:
        body = response.json()
        return json.dumps(body, indent=2)[:500]
    except Exception:
        return response.text[:500]


async def read_stream_error(response: httpx.Response) -> str:
    """Read a short error detail from a streaming response body."""
    raw = await response.aread()
    return raw[:500].decode(errors="replace")


class LLMProvider(ABC):
    """Abstract base for all LLM providers (OpenRouter, Ollama, etc.)."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        model: str,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send messages to a model and return the full response.

        `tools` is an OpenAI-style specs array. Messages may contain
        normalized assistant tool_calls / role:"tool" entries (see
        ChatResponse.tool_calls); each provider translates them to its
        native wire format. chat_stream() never takes tools — the tool
        loop is non-streaming by design.
        """
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict],
        model: str,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Send messages and yield plain-text content deltas as they arrive.

        Raises ProviderError (or an httpx error) on failure — never yields
        error payloads as content.
        """
        ...

    @abstractmethod
    async def list_models(self) -> list[dict]:
        """Return available models from this provider."""
        ...

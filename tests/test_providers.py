"""
Provider unit tests — lock the chat_stream contract: plain-text deltas out,
exceptions (never error payloads) on failure.
"""
import json

import httpx
import pytest
import respx

from fusion_app.providers import OllamaProvider, OpenRouterProvider, ProviderError
from fusion_app.providers.openrouter import OPENROUTER_CHAT_URL


@pytest.fixture
async def http_client():
    async with httpx.AsyncClient() as c:
        yield c


def openrouter_sse(*chunks: str) -> str:
    """Build an OpenRouter-style SSE body from content deltas."""
    lines = []
    for c in chunks:
        lines.append("data: " + json.dumps({"choices": [{"delta": {"content": c}}]}))
        lines.append("")
    lines.append("data: [DONE]")
    lines.append("")
    return "\n".join(lines)


def ollama_ndjson(*chunks: str) -> str:
    """Build an Ollama-style NDJSON body from content deltas."""
    lines = [
        json.dumps({"message": {"content": c}, "done": False}) for c in chunks
    ]
    lines.append(json.dumps({"message": {"content": ""}, "done": True, "eval_count": 7}))
    return "\n".join(lines)


# ── OpenRouter ──────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_openrouter_stream_yields_plain_text_deltas(http_client):
    with respx.mock:
        respx.post(OPENROUTER_CHAT_URL).mock(
            return_value=httpx.Response(200, text=openrouter_sse("Hel", "lo", " world"))
        )
        p = OpenRouterProvider("sk-test", http_client)
        out = [c async for c in p.chat_stream([{"role": "user", "content": "hi"}], "test/model")]
    assert out == ["Hel", "lo", " world"]


@pytest.mark.anyio
async def test_openrouter_stream_raises_on_http_error(http_client):
    with respx.mock:
        respx.post(OPENROUTER_CHAT_URL).mock(
            return_value=httpx.Response(401, json={"error": {"message": "bad key"}})
        )
        p = OpenRouterProvider("sk-test", http_client)
        with pytest.raises(ProviderError, match="HTTP 401"):
            async for _ in p.chat_stream([{"role": "user", "content": "hi"}], "test/model"):
                pass


@pytest.mark.anyio
async def test_openrouter_stream_raises_without_key(http_client):
    p = OpenRouterProvider("", http_client)
    with pytest.raises(ProviderError, match="API key not set"):
        async for _ in p.chat_stream([{"role": "user", "content": "hi"}], "test/model"):
            pass


@pytest.mark.anyio
async def test_openrouter_chat_forwards_max_tokens_and_usage(http_client):
    with respx.mock:
        route = respx.post(OPENROUTER_CHAT_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"total_tokens": 3},
                },
            )
        )
        p = OpenRouterProvider("sk-test", http_client)
        resp = await p.chat([{"role": "user", "content": "hi"}], "test/model", max_tokens=42)

    body = json.loads(route.calls[0].request.content)
    assert body["max_tokens"] == 42
    assert resp.content == "ok"
    assert resp.usage == {"total_tokens": 3}
    assert resp.error is None


@pytest.mark.anyio
async def test_openrouter_chat_system_prompt_prepended(http_client):
    with respx.mock:
        route = respx.post(OPENROUTER_CHAT_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        )
        p = OpenRouterProvider("sk-test", http_client)
        await p.chat(
            [{"role": "user", "content": "hi"}], "test/model", system_prompt="Be terse."
        )
    body = json.loads(route.calls[0].request.content)
    assert body["messages"][0] == {"role": "system", "content": "Be terse."}


# ── Ollama ──────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_ollama_stream_yields_plain_text_deltas(http_client):
    with respx.mock:
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, text=ollama_ndjson("Hel", "lo"))
        )
        p = OllamaProvider("http://localhost:11434", http_client)
        out = [c async for c in p.chat_stream([{"role": "user", "content": "hi"}], "llama3.2")]
    # The final done-frame stats must NOT leak into the content
    assert out == ["Hel", "lo"]


@pytest.mark.anyio
async def test_ollama_stream_raises_on_http_error(http_client):
    with respx.mock:
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(404, json={"error": "model not found"})
        )
        p = OllamaProvider("http://localhost:11434", http_client)
        with pytest.raises(ProviderError, match="HTTP 404"):
            async for _ in p.chat_stream([{"role": "user", "content": "hi"}], "nope"):
                pass


@pytest.mark.anyio
async def test_ollama_chat_forwards_max_tokens_as_num_predict(http_client):
    with respx.mock:
        route = respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": "ok"}})
        )
        p = OllamaProvider("http://localhost:11434", http_client)
        await p.chat([{"role": "user", "content": "hi"}], "llama3.2", max_tokens=64)
    body = json.loads(route.calls[0].request.content)
    assert body["options"] == {"num_predict": 64}

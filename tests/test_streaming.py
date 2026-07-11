"""
End-to-end tests for the /api/chat/stream SSE contract with mocked upstreams:
token events carry plain text, partial failures surface as error events, and
the synthesizer receives clean collected text (not raw provider JSON).
"""
import asyncio
import json

import httpx
import pytest
import respx
from httpx import AsyncClient

from fusion_app.config import SlotConfig
from fusion_app.providers.openrouter import OPENROUTER_CHAT_URL

from .test_providers import ollama_ndjson, openrouter_sse

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event_type, data) pairs."""
    events = []
    for block in text.strip().split("\n\n"):
        etype, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                etype = line[7:].strip()
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if etype is not None:
            events.append((etype, data))
    return events


async def configure(client: AsyncClient, slots: list[SlotConfig], **extra):
    resp = await client.put(
        "/api/config", json={"slots": [s.model_dump() for s in slots], **extra}
    )
    assert resp.status_code == 200


def five_slots(**enabled_slots: SlotConfig) -> list[SlotConfig]:
    """Build a 5-slot list; enabled_slots keys are 's0'..'s4'."""
    slots = [SlotConfig(provider="openrouter") for _ in range(5)]
    for key, slot in enabled_slots.items():
        slots[int(key[1])] = slot
    return slots


@pytest.mark.anyio
async def test_stream_endpoint_yields_plain_text_tokens(client: AsyncClient):
    slots = five_slots(s0=SlotConfig(provider="openrouter", model="test/model", enabled=True))
    await configure(client, slots, openrouter_key="sk-test")

    with respx.mock:
        respx.post(OPENROUTER_CHAT_URL).mock(
            return_value=httpx.Response(200, text=openrouter_sse("Hello", " world"))
        )
        resp = await client.post("/api/chat/stream", json={"prompt": "hi"})

    assert resp.status_code == 200
    events = parse_sse(resp.text)

    tokens = [d["content"] for t, d in events if t == "token"]
    assert tokens == ["Hello", " world"]

    done = next(d for t, d in events if t == "done")
    assert done["slot"] == 0
    assert done["full_content"] == "Hello world"


@pytest.mark.anyio
async def test_stream_endpoint_partial_failure(client: AsyncClient):
    """One slot fails (HTTP 500), the other succeeds — both surface correctly."""
    slots = five_slots(
        s0=SlotConfig(provider="openrouter", model="test/model", enabled=True),
        s1=SlotConfig(provider="ollama", model="llama3.2", enabled=True),
    )
    await configure(client, slots, openrouter_key="sk-test")

    with respx.mock:
        respx.post(OPENROUTER_CHAT_URL).mock(
            return_value=httpx.Response(500, json={"error": "boom"})
        )
        respx.post(OLLAMA_CHAT_URL).mock(
            return_value=httpx.Response(200, text=ollama_ndjson("ok"))
        )
        resp = await client.post("/api/chat/stream", json={"prompt": "hi"})

    events = parse_sse(resp.text)

    error = next(d for t, d in events if t == "error")
    assert error["slot"] == 0
    assert "HTTP 500" in error["error"]

    done = next(d for t, d in events if t == "done")
    assert done["slot"] == 1
    assert done["full_content"] == "ok"


@pytest.mark.anyio
async def test_stream_endpoint_honors_slot_param(client: AsyncClient):
    """body.slot streams only that slot."""
    slots = five_slots(
        s0=SlotConfig(provider="openrouter", model="test/model", enabled=True),
        s1=SlotConfig(provider="ollama", model="llama3.2", enabled=True),
    )
    await configure(client, slots, openrouter_key="sk-test")

    with respx.mock:
        respx.post(OLLAMA_CHAT_URL).mock(
            return_value=httpx.Response(200, text=ollama_ndjson("only me"))
        )
        resp = await client.post("/api/chat/stream", json={"prompt": "hi", "slot": 1})

    events = parse_sse(resp.text)
    slots_seen = {d["slot"] for t, d in events if t in ("token", "done", "error")}
    assert slots_seen == {1}


@pytest.mark.anyio
async def test_stream_synth_mode_feeds_clean_text_to_synth(client: AsyncClient):
    """Regression for the raw-JSON streaming bug: the synth meta-prompt must
    contain the streamed slots' plain text, not provider JSON chunks."""
    slots = five_slots(
        s0=SlotConfig(provider="ollama", model="llama3.2", enabled=True),
        s2=SlotConfig(provider="openrouter", model="synth/model", enabled=True),
    )
    await configure(
        client, slots, openrouter_key="sk-test", synth_mode=True, synth_slot=2
    )

    with respx.mock:
        # Slot 0 streams from Ollama; the synth call goes to OpenRouter
        respx.post(OLLAMA_CHAT_URL).mock(
            return_value=httpx.Response(200, text=ollama_ndjson("Hello", " world"))
        )
        synth_route = respx.post(OPENROUTER_CHAT_URL).mock(
            return_value=httpx.Response(
                200, json={"choices": [{"message": {"content": "synthesized!"}}]}
            )
        )
        resp = await client.post("/api/chat/stream", json={"prompt": "hi"})

        synth_request = json.loads(synth_route.calls[0].request.content)

    events = parse_sse(resp.text)
    synth = next(d for t, d in events if t == "synth")
    assert synth["content"] == "synthesized!"
    assert synth["error"] is None

    # The draft passed to the synthesizer is the clean concatenated text
    draft_prompt = synth_request["messages"][-1]["content"]
    assert "Hello world" in draft_prompt
    assert '"message"' not in draft_prompt  # no raw Ollama JSON leaked


@pytest.mark.anyio
async def test_stream_no_slots_enabled(client: AsyncClient):
    slots = five_slots()
    await configure(client, slots)
    resp = await client.post("/api/chat/stream", json={"prompt": "hi"})
    events = parse_sse(resp.text)
    error = next(d for t, d in events if t == "error")
    assert error["slot"] == -1
    assert "No slots enabled" in error["error"]


@pytest.mark.anyio
async def test_v1_stream_yields_openai_chunks(client: AsyncClient):
    """/v1 streaming emits OpenAI-style chunks built from plain-text deltas."""
    slots = five_slots(s0=SlotConfig(provider="openrouter", model="test/model", enabled=True))
    await configure(client, slots, openrouter_key="sk-test")

    with respx.mock:
        respx.post(OPENROUTER_CHAT_URL).mock(
            return_value=httpx.Response(200, text=openrouter_sse("Hi", " there"))
        )
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

    assert resp.status_code == 200
    payloads = [
        line[6:] for line in resp.text.split("\n") if line.startswith("data: ")
    ]
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1]]
    deltas = [
        c["choices"][0]["delta"].get("content")
        for c in chunks
        if c.get("object") == "chat.completion.chunk"
    ]
    assert deltas[:-1] == ["Hi", " there"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.anyio
async def test_stream_deltas_emits_heartbeats_during_silence(monkeypatch):
    """A silent upstream produces SSE comment heartbeats, and the stream still
    finishes with the normal content/finish/[DONE] frames once deltas arrive."""
    from fusion_app import api

    monkeypatch.setattr(api, "HEARTBEAT_INTERVAL", 0.05)

    async def slow_gen():
        await asyncio.sleep(0.18)
        yield "late"

    frames = [f async for f in api._stream_deltas(slow_gen(), "cid", 0, "m")]

    heartbeat_idxs = [i for i, f in enumerate(frames) if f == ": heartbeat\n\n"]
    assert len(heartbeat_idxs) >= 2

    content_idx = next(
        i for i, f in enumerate(frames)
        if f.startswith("data: ") and '"late"' in f
    )
    # Heartbeats fill the silence BEFORE the first delta, never after [DONE]
    assert all(i < content_idx for i in heartbeat_idxs)
    assert frames[-1] == "data: [DONE]\n\n"
    assert json.loads(frames[-2][6:])["choices"][0]["finish_reason"] == "stop"


@pytest.mark.anyio
async def test_v1_synth_stream_heartbeats_while_drafts_gather(
    client: AsyncClient, monkeypatch
):
    """In synth mode, /v1 streaming emits heartbeats during the draft-gathering
    silent window without corrupting the OpenAI chunk stream."""
    from fusion_app import api

    monkeypatch.setattr(api, "HEARTBEAT_INTERVAL", 0.05)

    slots = five_slots(
        s0=SlotConfig(provider="ollama", model="llama3.2", enabled=True),
        s2=SlotConfig(provider="openrouter", model="synth/model", enabled=True),
    )
    await configure(
        client, slots, openrouter_key="sk-test", synth_mode=True, synth_slot=2
    )

    async def slow_draft(request):
        await asyncio.sleep(0.2)
        return httpx.Response(200, text=ollama_ndjson("draft"))

    with respx.mock:
        respx.post(OLLAMA_CHAT_URL).mock(side_effect=slow_draft)
        respx.post(OPENROUTER_CHAT_URL).mock(
            return_value=httpx.Response(200, text=openrouter_sse("synth!"))
        )
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

    assert resp.status_code == 200
    assert ": heartbeat" in resp.text

    # Comment lines must not break OpenAI-format parsing of data: lines
    payloads = [
        line[6:] for line in resp.text.split("\n") if line.startswith("data: ")
    ]
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1]]
    deltas = [
        c["choices"][0]["delta"].get("content")
        for c in chunks
        if c.get("object") == "chat.completion.chunk"
    ]
    assert "synth!" in "".join(d for d in deltas if d)

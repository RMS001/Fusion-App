"""
Tool-calling tests: loop mechanics, iteration cap, provider parsing and
fallback, SSRF guard, config back-compat, masking, streaming synth with tools.
All offline (respx-mocked).
"""
import json
from pathlib import Path

import httpx
import pytest
import respx

import fusion_app.panel as panel_module
from fusion_app.config import FusionConfig, SlotConfig, ToolsConfig, load_config, save_config
from fusion_app.panel import PanelManager
from fusion_app.providers import OllamaProvider, OpenRouterProvider
from fusion_app.providers.openrouter import OPENROUTER_CHAT_URL
from fusion_app.tools import dispatch
from fusion_app.tools.base import Tool
from fusion_app.tools.web import WebFetchTool, check_url_allowed

OLLAMA_CHAT = "http://localhost:11434/api/chat"


class FakeTool(Tool):
    name = "fake_lookup"
    description = "test tool"
    parameters = {"type": "object", "properties": {"q": {"type": "string"}}}

    def __init__(self):
        self.calls = []

    async def execute(self, args: dict) -> str:
        self.calls.append(args)
        return "FAKE RESULT"


@pytest.fixture
async def http_client():
    async with httpx.AsyncClient() as c:
        yield c


def _tools_config(**overrides) -> FusionConfig:
    cfg = FusionConfig()
    cfg.slots[0] = SlotConfig(provider="ollama", model="m0", enabled=True)
    cfg.slots[1] = SlotConfig(
        provider="ollama", model="synth-m", enabled=True, tools_enabled=True
    )
    for i in range(2, 5):
        cfg.slots[i].enabled = False
    cfg.synth_slot = 1
    for k, v in overrides.items():
        setattr(cfg.tools, k, v)
    return cfg


def ollama_tool_call_response(name="fake_lookup", arguments=None):
    return httpx.Response(
        200,
        json={
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": name, "arguments": arguments or {"q": "x"}}}
                ],
            },
            "done": True,
        },
    )


def ollama_plain_response(content="final answer"):
    return httpx.Response(
        200, json={"message": {"role": "assistant", "content": content}, "done": True}
    )


# ── Loop mechanics ──────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_tool_loop_executes_tool_and_returns_final(http_client, monkeypatch):
    cfg = _tools_config()
    fake = FakeTool()
    monkeypatch.setattr(panel_module, "build_tools", lambda c, cl: [fake])
    manager = PanelManager(cfg, http_client)
    with respx.mock:
        route = respx.post(OLLAMA_CHAT).mock(
            side_effect=[ollama_tool_call_response(), ollama_plain_response()]
        )
        msgs, resp, trace = await manager._chat_with_tools(
            cfg.slots[1], [{"role": "user", "content": "hi"}], [fake]
        )
    assert route.call_count == 2
    assert fake.calls == [{"q": "x"}]
    assert resp.content == "final answer"
    assert resp.error is None
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "FAKE RESULT"
    assert tool_msgs[0]["name"] == "fake_lookup"
    assert len(trace) == 1
    assert trace[0]["name"] == "fake_lookup"
    assert trace[0]["result_preview"] == "FAKE RESULT"
    assert resp.tool_trace == trace


@pytest.mark.anyio
async def test_tool_loop_iteration_cap_forces_finalization(http_client, monkeypatch):
    cfg = _tools_config(max_iterations=2)
    fake = FakeTool()
    monkeypatch.setattr(panel_module, "build_tools", lambda c, cl: [fake])
    manager = PanelManager(cfg, http_client)
    with respx.mock:
        route = respx.post(OLLAMA_CHAT).mock(
            side_effect=[
                ollama_tool_call_response(),
                ollama_tool_call_response(),
                ollama_plain_response("forced"),
            ]
        )
        msgs, resp, trace = await manager._chat_with_tools(
            cfg.slots[1], [{"role": "user", "content": "hi"}], [fake]
        )
    # exactly max_iterations tool rounds + one forced no-tools finalization
    assert route.call_count == 3
    assert len(trace) == 2
    assert resp.content == "forced"
    bodies = [json.loads(c.request.content) for c in route.calls]
    assert "tools" in bodies[0] and "tools" in bodies[1]
    assert "tools" not in bodies[2]
    assert any(
        m.get("role") == "user" and "Tool budget exhausted" in m.get("content", "")
        for m in bodies[2]["messages"]
    )


@pytest.mark.anyio
async def test_unknown_tool_returns_error_and_completes(http_client, monkeypatch):
    cfg = _tools_config()
    fake = FakeTool()
    monkeypatch.setattr(panel_module, "build_tools", lambda c, cl: [fake])
    manager = PanelManager(cfg, http_client)
    with respx.mock:
        respx.post(OLLAMA_CHAT).mock(
            side_effect=[
                ollama_tool_call_response(name="nonexistent_tool"),
                ollama_plain_response(),
            ]
        )
        msgs, resp, trace = await manager._chat_with_tools(
            cfg.slots[1], [{"role": "user", "content": "hi"}], [fake]
        )
    assert resp.content == "final answer"
    assert trace[0]["result_preview"].startswith("ERROR: unknown tool")


@pytest.mark.anyio
async def test_dispatch_timeout_returns_error_string():
    import asyncio

    class SlowTool(Tool):
        name = "slow"
        description = "slow"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, args: dict) -> str:
            await asyncio.sleep(5)
            return "never"

    result = await dispatch([SlowTool()], "slow", {}, timeout=0.05)
    assert result.startswith("ERROR:")
    assert "timed out" in result


# ── Provider layer ──────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_ollama_no_tool_support_fallback(http_client):
    p = OllamaProvider("http://localhost:11434", http_client)
    spec = FakeTool().spec()
    with respx.mock:
        route = respx.post(OLLAMA_CHAT).mock(
            side_effect=[
                httpx.Response(
                    400, json={"error": "registry.ollama.ai/library/m does not support tools"}
                ),
                ollama_plain_response("plain anyway"),
            ]
        )
        resp = await p.chat([{"role": "user", "content": "hi"}], "m", tools=[spec])
    assert route.call_count == 2
    assert resp.error is None
    assert resp.content == "plain anyway"
    assert resp.warning is not None and "does not support tools" in resp.warning
    assert "tools" not in json.loads(route.calls[1].request.content)


@pytest.mark.anyio
async def test_ollama_400_without_tools_message_still_errors(http_client):
    p = OllamaProvider("http://localhost:11434", http_client)
    with respx.mock:
        respx.post(OLLAMA_CHAT).mock(
            return_value=httpx.Response(400, json={"error": "something else"})
        )
        resp = await p.chat(
            [{"role": "user", "content": "hi"}], "m", tools=[FakeTool().spec()]
        )
    assert resp.error is not None and "HTTP 400" in resp.error


@pytest.mark.anyio
async def test_openrouter_parses_string_arguments_and_tolerates_malformed(http_client):
    p = OpenRouterProvider("sk-test", http_client)
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "web_fetch", "arguments": '{"url": "https://x.y"}'},
            },
            {
                "id": "call_bad",
                "type": "function",
                "function": {"name": "web_fetch", "arguments": "{not json"},
            },
        ],
    }
    with respx.mock:
        respx.post(OPENROUTER_CHAT_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": message}]})
        )
        resp = await p.chat([{"role": "user", "content": "hi"}], "m", tools=[FakeTool().spec()])
    assert resp.error is None
    assert resp.content == ""  # null content normalized, no crash
    assert resp.tool_calls[0] == {
        "id": "call_abc",
        "name": "web_fetch",
        "arguments": {"url": "https://x.y"},
    }
    assert resp.tool_calls[1]["arguments"] is None  # malformed → per-call ERROR later


@pytest.mark.anyio
async def test_malformed_arguments_become_error_result(http_client, monkeypatch):
    cfg = _tools_config()
    fake = FakeTool()
    monkeypatch.setattr(panel_module, "build_tools", lambda c, cl: [fake])
    manager = PanelManager(cfg, http_client)
    bad_args_response = httpx.Response(
        200,
        json={
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "fake_lookup", "arguments": "{oops"}}],
            },
            "done": True,
        },
    )
    with respx.mock:
        respx.post(OLLAMA_CHAT).mock(
            side_effect=[bad_args_response, ollama_plain_response()]
        )
        _, resp, trace = await manager._chat_with_tools(
            cfg.slots[1], [{"role": "user", "content": "hi"}], [fake]
        )
    assert fake.calls == []  # never dispatched
    assert trace[0]["result_preview"].startswith("ERROR: malformed JSON")
    assert resp.content == "final answer"


def test_openrouter_wire_message_round_trip():
    normalized = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_1", "name": "t", "arguments": {"a": 1}}],
    }
    wire = OpenRouterProvider._to_wire_message(normalized)
    assert wire["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    tool_msg = OpenRouterProvider._to_wire_message(
        {"role": "tool", "tool_call_id": "call_1", "name": "t", "content": "r"}
    )
    assert tool_msg == {"role": "tool", "tool_call_id": "call_1", "content": "r"}


def test_ollama_wire_message_round_trip():
    wire = OllamaProvider._to_wire_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_0", "name": "t", "arguments": {"a": 1}}],
        }
    )
    assert wire["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    tool_msg = OllamaProvider._to_wire_message(
        {"role": "tool", "tool_call_id": "call_0", "name": "t", "content": "r"}
    )
    assert tool_msg == {"role": "tool", "content": "r", "tool_name": "t"}


# ── SSRF guard ──────────────────────────────────────────────────────────────


@pytest.mark.anyio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",
        "http://192.168.1.5/x",
        "http://10.0.0.1/x",
        "http://169.254.169.254/latest/meta-data/",
        "http://100.117.97.90/x",  # CGNAT / Tailscale range
        "ftp://example.com/x",
    ],
)
async def test_ssrf_guard_rejects_private_and_bad_schemes(url):
    err = await check_url_allowed(url, allow_private=False)
    assert err is not None and err.startswith("ERROR")


@pytest.mark.anyio
async def test_ssrf_guard_allows_private_when_configured():
    assert await check_url_allowed("http://192.168.1.5/x", allow_private=True) is None


@pytest.mark.anyio
async def test_web_fetch_rejects_redirect_to_private(http_client):
    tool = WebFetchTool(ToolsConfig(), http_client)
    with respx.mock:
        respx.get("http://93.184.216.34/start").mock(
            return_value=httpx.Response(302, headers={"location": "http://192.168.1.5/admin"})
        )
        result = await tool.execute({"url": "http://93.184.216.34/start"})
    assert result.startswith("ERROR")
    assert "private" in result


@pytest.mark.anyio
async def test_web_fetch_head_mode_reports_status(http_client):
    tool = WebFetchTool(ToolsConfig(), http_client)
    with respx.mock:
        respx.head("http://93.184.216.34/lib.js").mock(
            return_value=httpx.Response(
                200, headers={"content-type": "text/javascript", "content-length": "123"}
            )
        )
        result = await tool.execute({"url": "http://93.184.216.34/lib.js", "mode": "head"})
    assert result.startswith("HTTP 200")
    assert "text/javascript" in result


# ── Config back-compat & masking ────────────────────────────────────────────


def test_old_config_without_tools_loads_and_round_trips(tmp_path: Path):
    path = tmp_path / "config.json"
    old = {
        "openrouter_key": "sk-x",
        "synth_slot": 1,
        "slots": [
            {"provider": "ollama", "model": "m", "enabled": True} for _ in range(5)
        ],
    }
    path.write_text(json.dumps(old))
    cfg = load_config(path)
    assert cfg.tools.web_search_backend == "duckduckgo"
    assert cfg.tools.max_iterations == 5
    assert cfg.slots[0].tools_enabled is False
    save_config(cfg, path)
    again = load_config(path)
    assert again.tools.model_dump() == cfg.tools.model_dump()


@pytest.mark.anyio
async def test_tools_keys_masked_in_get_config(client):
    resp = await client.put(
        "/api/config",
        json={"tools": {"context7_api_key": "ctx7sk-supersecret123456", "brave_api_key": ""}},
    )
    assert resp.status_code == 200
    data = (await client.get("/api/config")).json()
    tools = data["tools"]
    assert "supersecret" not in json.dumps(data)
    assert tools["context7_api_key_set"] is True
    assert tools["brave_api_key_set"] is False
    # partial update didn't clobber: saving unrelated field keeps the key
    await client.put("/api/config", json={"tools": {"web_enabled": False}})
    data2 = (await client.get("/api/config")).json()
    assert data2["tools"]["context7_api_key_set"] is True
    assert data2["tools"]["web_enabled"] is False


# ── Streaming synth with tools ──────────────────────────────────────────────


@pytest.mark.anyio
async def test_streaming_synth_with_tools_emits_events_then_content(
    http_client, monkeypatch
):
    cfg = _tools_config()
    fake = FakeTool()
    monkeypatch.setattr(panel_module, "build_tools", lambda c, cl: [fake])
    manager = PanelManager(cfg, http_client)
    with respx.mock:
        respx.post(OLLAMA_CHAT).mock(
            side_effect=[ollama_tool_call_response(), ollama_plain_response("synth out")]
        )
        items = [
            item
            async for item in manager.synthesize_stream_from_collected(
                [{"role": "user", "content": "hi"}],
                {"Slot 0 (m0)": "draft text"},
                yield_events=True,
            )
        ]
    dict_events = [i for i in items if isinstance(i, dict)]
    str_chunks = [i for i in items if isinstance(i, str)]
    assert [e["type"] for e in dict_events] == ["tool_call", "synth_meta"]
    assert dict_events[0]["name"] == "fake_lookup"
    assert dict_events[1]["trace"][0]["result_preview"] == "FAKE RESULT"
    assert str_chunks == ["synth out"]
    # tools clause injected at runtime into the synth system message
    assert fake.calls == [{"q": "x"}]


@pytest.mark.anyio
async def test_streaming_synth_without_events_yields_only_text(
    http_client, monkeypatch
):
    """/v1 path: default yield_events=False must never emit dicts."""
    cfg = _tools_config()
    fake = FakeTool()
    monkeypatch.setattr(panel_module, "build_tools", lambda c, cl: [fake])
    manager = PanelManager(cfg, http_client)
    with respx.mock:
        respx.post(OLLAMA_CHAT).mock(
            side_effect=[ollama_tool_call_response(), ollama_plain_response("synth out")]
        )
        items = [
            item
            async for item in manager.synthesize_stream_from_collected(
                [{"role": "user", "content": "hi"}], {"Slot 0 (m0)": "draft"}
            )
        ]
    assert items == ["synth out"]


@pytest.mark.anyio
async def test_synth_system_prompt_gains_tools_addendum(http_client, monkeypatch):
    cfg = _tools_config()
    fake = FakeTool()
    monkeypatch.setattr(panel_module, "build_tools", lambda c, cl: [fake])
    manager = PanelManager(cfg, http_client)
    with respx.mock:
        route = respx.post(OLLAMA_CHAT).mock(return_value=ollama_plain_response())
        await manager.synthesize_from_collected(
            [{"role": "user", "content": "hi"}], {"Slot 0 (m0)": "draft"}
        )
    body = json.loads(route.calls[0].request.content)
    system = next(m for m in body["messages"] if m["role"] == "system")
    assert "fact-checking tools" in system["content"]
    # tools off → no addendum (byte-identical behavior)
    cfg.slots[1].tools_enabled = False
    with respx.mock:
        route2 = respx.post(OLLAMA_CHAT).mock(return_value=ollama_plain_response())
        await manager.synthesize_from_collected(
            [{"role": "user", "content": "hi"}], {"Slot 0 (m0)": "draft"}
        )
    body2 = json.loads(route2.calls[0].request.content)
    system2 = next(m for m in body2["messages"] if m["role"] == "system")
    assert "fact-checking tools" not in system2["content"]


# ── Tools on every path (streaming) ─────────────────────────────────────────


@pytest.mark.anyio
async def test_chat_all_stream_tooled_slot_emits_tool_events_then_content(
    http_client, monkeypatch
):
    cfg = _tools_config()
    cfg.synth_slot = -1
    cfg.slots[0].tools_enabled = True  # tooled draft slot
    cfg.slots[1].enabled = False
    fake = FakeTool()
    monkeypatch.setattr(panel_module, "build_tools", lambda c, cl: [fake])
    manager = PanelManager(cfg, http_client)
    with respx.mock:
        respx.post(OLLAMA_CHAT).mock(
            side_effect=[ollama_tool_call_response(), ollama_plain_response("draft out")]
        )
        events = [
            e
            async for e in manager.chat_all_stream([{"role": "user", "content": "hi"}])
        ]
    assert [e["type"] for e in events] == ["tool_call", "token", "done"]
    assert events[0]["slot"] == 0 and events[0]["name"] == "fake_lookup"
    assert events[1] == {"type": "token", "slot": 0, "content": "draft out"}
    assert events[2]["full_content"] == "draft out"
    assert events[2]["tool_trace"][0]["result_preview"] == "FAKE RESULT"
    assert events[2]["warning"] is None


@pytest.mark.anyio
async def test_chat_all_stream_untooled_slot_streams_unchanged(
    http_client, monkeypatch
):
    cfg = _tools_config()
    cfg.synth_slot = -1
    cfg.slots[0].tools_enabled = False
    cfg.slots[1].enabled = False
    monkeypatch.setattr(panel_module, "build_tools", lambda c, cl: [FakeTool()])
    manager = PanelManager(cfg, http_client)
    stream_body = "\n".join(
        json.dumps(x)
        for x in [
            {"message": {"content": "Hel"}},
            {"message": {"content": "lo"}, "done": True},
        ]
    )
    with respx.mock:
        respx.post(OLLAMA_CHAT).mock(return_value=httpx.Response(200, text=stream_body))
        events = [
            e
            async for e in manager.chat_all_stream([{"role": "user", "content": "hi"}])
        ]
    assert [e["type"] for e in events] == ["token", "token", "done"]
    assert "tool_trace" not in events[2]


@pytest.mark.anyio
async def test_chat_all_stream_tooled_slot_error_carries_trace(
    http_client, monkeypatch
):
    cfg = _tools_config()
    cfg.synth_slot = -1
    cfg.slots[0].tools_enabled = True
    cfg.slots[1].enabled = False
    fake = FakeTool()
    monkeypatch.setattr(panel_module, "build_tools", lambda c, cl: [fake])
    manager = PanelManager(cfg, http_client)
    with respx.mock:
        respx.post(OLLAMA_CHAT).mock(
            side_effect=[
                ollama_tool_call_response(),
                httpx.Response(500, json={"error": "boom"}),
            ]
        )
        events = [
            e
            async for e in manager.chat_all_stream([{"role": "user", "content": "hi"}])
        ]
    assert [e["type"] for e in events] == ["tool_call", "error"]
    assert "HTTP 500" in events[1]["error"]
    assert len(events[1]["tool_trace"]) == 1


@pytest.mark.anyio
async def test_final_answer_reminder_transient_on_every_call(http_client, monkeypatch):
    from fusion_app.panel import TOOLS_FINAL_ANSWER_REMINDER

    cfg = _tools_config()
    fake = FakeTool()
    monkeypatch.setattr(panel_module, "build_tools", lambda c, cl: [fake])
    manager = PanelManager(cfg, http_client)
    with respx.mock:
        route = respx.post(OLLAMA_CHAT).mock(
            side_effect=[ollama_tool_call_response(), ollama_plain_response()]
        )
        msgs, _, _ = await manager._chat_with_tools(
            cfg.slots[1], [{"role": "user", "content": "hi"}], [fake]
        )
    for call in route.calls:
        body = json.loads(call.request.content)
        assert body["messages"][-1]["content"] == TOOLS_FINAL_ANSWER_REMINDER
        # not duplicated earlier in the transcript
        assert (
            sum(
                1
                for m in body["messages"]
                if m.get("content") == TOOLS_FINAL_ANSWER_REMINDER
            )
            == 1
        )
    # never persisted into the returned transcript
    assert all(m.get("content") != TOOLS_FINAL_ANSWER_REMINDER for m in msgs)


@pytest.mark.anyio
async def test_addendum_includes_loading_pattern_rule(http_client, monkeypatch):
    cfg = _tools_config()
    fake = FakeTool()
    monkeypatch.setattr(panel_module, "build_tools", lambda c, cl: [fake])
    manager = PanelManager(cfg, http_client)
    with respx.mock:
        route = respx.post(OLLAMA_CHAT).mock(return_value=ollama_plain_response())
        await manager.synthesize_from_collected(
            [{"role": "user", "content": "hi"}], {"Slot 0 (m0)": "draft"}
        )
    body = json.loads(route.calls[0].request.content)
    system = next(m for m in body["messages"] if m["role"] == "system")
    assert "DOCUMENTED" in system["content"]


# ── /v1 streaming with a tooled slot (e2e) ──────────────────────────────────


@pytest.mark.anyio
async def test_v1_stream_tooled_slot_single_text_delta(client, monkeypatch):
    from fusion_app.config import SlotConfig as SC

    slots = [SC(provider="openrouter") for _ in range(5)]
    slots[0] = SC(provider="ollama", model="m0", enabled=True, tools_enabled=True)
    resp = await client.put(
        "/api/config",
        json={"slots": [s.model_dump() for s in slots], "synth_mode": False},
    )
    assert resp.status_code == 200

    fake = FakeTool()
    monkeypatch.setattr(panel_module, "build_tools", lambda c, cl: [fake])
    with respx.mock:
        respx.post(OLLAMA_CHAT).mock(
            side_effect=[ollama_tool_call_response(), ollama_plain_response("answer")]
        )
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "fusion-panel",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert resp.status_code == 200
    payloads = [
        line[6:] for line in resp.text.split("\n") if line.startswith("data: ")
    ]
    assert payloads[-1].strip() == "[DONE]"
    deltas = [
        (json.loads(p)["choices"][0]["delta"].get("content") or "")
        for p in payloads
        if p.strip() != "[DONE]"
    ]
    assert "".join(deltas) == "answer"
    assert fake.calls == [{"q": "x"}]
    # text-only contract: no tool_calls anywhere in the stream
    assert "tool_call" not in resp.text

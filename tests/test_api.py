"""
Tests for Fusion App API.

Run:  python -m pytest tests/ -v
"""
import json
from pathlib import Path

import httpx
import pytest
import respx
from httpx import AsyncClient

from fusion_app.config import SlotConfig, load_config


# ── Health ──────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── Config ──────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_get_config_defaults(client: AsyncClient):
    resp = await client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "openrouter_key" in data
    assert "ollama_base_url" in data
    assert len(data["slots"]) == 5
    assert data["openrouter_key_set"] is False


@pytest.mark.anyio
async def test_update_config(client: AsyncClient):
    slots = [
        SlotConfig(provider="openrouter", model="gpt-4o", enabled=True),
        SlotConfig(provider="ollama", model="llama3.2", enabled=False),
        SlotConfig(provider="openrouter", model="", enabled=False),
        SlotConfig(provider="openrouter", model="", enabled=False),
        SlotConfig(provider="openrouter", model="", enabled=False),
    ]
    resp = await client.put(
        "/api/config",
        json={
            "openrouter_key": "sk-or-v1-test123",
            "ollama_base_url": "http://192.168.1.100:11434",
            "slots": [s.model_dump() for s in slots],
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "saved"}

    # Verify it persisted
    resp2 = await client.get("/api/config")
    data = resp2.json()
    assert data["ollama_base_url"] == "http://192.168.1.100:11434"
    assert data["openrouter_key_set"] is True
    assert data["slots"][1]["enabled"] is False


@pytest.mark.anyio
async def test_update_config_wrong_slot_count(client: AsyncClient):
    resp = await client.put(
        "/api/config",
        json={
            "slots": [{"provider": "openrouter", "model": "gpt-4o", "enabled": True}],
        },
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_update_config_rejects_non_http_url(client: AsyncClient):
    """Slot URL overrides and the global Ollama URL must be http(s) URLs."""
    slots = [SlotConfig(provider="openrouter") for _ in range(5)]
    slot_dicts = [s.model_dump() for s in slots]
    slot_dicts[0]["base_url_override"] = "ftp://192.168.1.10:11434"
    resp = await client.put("/api/config", json={"slots": slot_dicts})
    assert resp.status_code == 422

    resp = await client.put("/api/config", json={"ollama_base_url": "not-a-url"})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_update_config_failed_validation_leaves_config_unchanged(client: AsyncClient):
    """A rejected update must not partially apply (validate-then-apply)."""
    resp = await client.put(
        "/api/config",
        json={"openrouter_key": "sk-should-not-stick", "ollama_base_url": "not-a-url"},
    )
    assert resp.status_code == 422

    resp2 = await client.get("/api/config")
    assert resp2.json()["openrouter_key_set"] is False


# ── Auth on /api/* ──────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_api_requires_auth_when_key_set(client: AsyncClient):
    """Once a private key is set, /api/* endpoints require the Bearer token."""
    resp = await client.put("/api/config", json={"private_api_key": "k1"})
    assert resp.status_code == 200

    # No header → 401
    resp = await client.get("/api/config")
    assert resp.status_code == 401
    # Wrong key → 401
    resp = await client.get("/api/config", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401
    # Correct key → 200
    resp = await client.get("/api/config", headers={"Authorization": "Bearer k1"})
    assert resp.status_code == 200
    # Chat endpoints are guarded too
    resp = await client.post("/api/chat", json={"prompt": "hi"})
    assert resp.status_code == 401


# ── Synth ────────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_synth_no_slot_configured(client: AsyncClient):
    """Synth endpoint returns error when no synth slot is set."""
    resp = await client.post("/api/synth", json={"prompt": "Hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["synth_slot"] == -1
    assert data["synthesis"]["error"] is not None
    assert "No synth slot configured" in data["synthesis"]["error"]


@pytest.mark.anyio
async def test_synth_with_configured_slot(client: AsyncClient):
    """Synth endpoint collects responses and runs synthesis."""
    slots = [
        SlotConfig(provider="openrouter", model="gpt-4o", enabled=True),
        SlotConfig(provider="ollama", model="llama3.2", enabled=False),
        SlotConfig(provider="openrouter", model="synth-model", enabled=True),
        SlotConfig(provider="openrouter", model="", enabled=False),
        SlotConfig(provider="openrouter", model="", enabled=False),
    ]
    await client.put(
        "/api/config",
        json={"synth_slot": 2, "slots": [s.model_dump() for s in slots]},
    )

    resp = await client.post("/api/synth", json={"prompt": "Say hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["synth_slot"] == 2
    assert data["synth_model"] == "synth-model"
    # Slot 0 should have an error (no API key), slot 2 is the synth
    slot_key = next(k for k in data["responses"] if "Slot 0" in k)
    assert data["responses"][slot_key]["error"] is not None
    # Synth itself should also have an error (no key)
    assert data["synthesis"]["error"] is not None


# ── Models endpoints ────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_list_openrouter_models(client: AsyncClient):
    """Models endpoint returns valid JSON with a models key regardless of key state."""
    resp = await client.get("/api/models/openrouter")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert isinstance(data["models"], list)


@pytest.mark.anyio
async def test_list_ollama_models(client: AsyncClient):
    with respx.mock:
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(200, json={"models": [{"name": "llama3.2"}]})
        )
        resp = await client.get("/api/models/ollama")
    assert resp.status_code == 200
    data = resp.json()
    assert data["models"] == [{"name": "llama3.2"}]


# ── Chat ────────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_chat_no_slots_enabled(client: AsyncClient):
    # Disable all slots first
    slots = [
        SlotConfig(provider="openrouter", model="gpt-4o", enabled=False),
        SlotConfig(provider="ollama", model="llama3.2", enabled=False),
        SlotConfig(provider="openrouter", model="", enabled=False),
        SlotConfig(provider="openrouter", model="", enabled=False),
        SlotConfig(provider="openrouter", model="", enabled=False),
    ]
    await client.put(
        "/api/config",
        json={"slots": [s.model_dump() for s in slots]},
    )

    resp = await client.post("/api/chat", json={"prompt": "Hello"})
    assert resp.status_code == 200
    data = resp.json()
    # All slots should be None
    assert all(v is None for v in data.values())


@pytest.mark.anyio
async def test_chat_single_slot(client: AsyncClient):
    # Enable just slot 0 with a known "model" that will fail gracefully
    slots = [
        SlotConfig(provider="openrouter", model="nonexistent-model-test", enabled=True),
        SlotConfig(provider="ollama", model="llama3.2", enabled=False),
        SlotConfig(provider="openrouter", model="", enabled=False),
        SlotConfig(provider="openrouter", model="", enabled=False),
        SlotConfig(provider="openrouter", model="", enabled=False),
    ]
    await client.put(
        "/api/config",
        json={"slots": [s.model_dump() for s in slots]},
    )

    # Without API key, should fail gracefully with error
    resp = await client.post("/api/chat", json={"prompt": "Hi", "slot": 0})
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    # Should have error since no key
    assert data["response"]["error"] is not None
    assert data["response"]["content"] == ""


@pytest.mark.anyio
async def test_chat_slot_out_of_range_is_422(client: AsyncClient):
    resp = await client.post("/api/chat", json={"prompt": "Hi", "slot": 9})
    assert resp.status_code == 422
    resp = await client.post("/api/chat", json={"prompt": "Hi", "slot": -2})
    assert resp.status_code == 422


# ── New config fields ─────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_config_returns_new_fields(client: AsyncClient):
    """GET /api/config returns private_api_key_set, synth_mode fields."""
    resp = await client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "private_api_key" in data
    assert "private_api_key_set" in data
    assert data["private_api_key_set"] is False
    assert "synth_mode" in data
    assert data["synth_mode"] is False
    assert "synth_system_prompt" in data
    assert "slot_timeout" in data


@pytest.mark.anyio
async def test_update_config_with_private_api_key_and_synth_mode(client: AsyncClient):
    """PUT /api/config persists private_api_key and synth_mode."""
    resp = await client.put(
        "/api/config",
        json={
            "private_api_key": "sk-test-secret-123",
            "synth_mode": True,
            "synth_slot": 0,
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "saved"}

    # Verify persistence (subsequent /api/* calls now require the key)
    resp2 = await client.get(
        "/api/config", headers={"Authorization": "Bearer sk-test-secret-123"}
    )
    data = resp2.json()
    assert data["private_api_key_set"] is True
    assert data["synth_mode"] is True
    assert data["synth_slot"] == 0
    # Key itself should be masked
    assert data["private_api_key"] != "sk-test-secret-123"


# ── Slot URL override ──────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_config_with_base_url_override(client: AsyncClient):
    """PUT /api/config persists base_url_override on individual slots."""
    slots = [
        SlotConfig(provider="ollama", model="qwen-coder-80b", enabled=True, base_url_override="http://192.168.1.10:11434"),
        SlotConfig(provider="ollama", model="llama-3.3-70b", enabled=True, base_url_override="http://192.168.1.11:11434"),
        SlotConfig(provider="openrouter", model="", enabled=False),
        SlotConfig(provider="openrouter", model="", enabled=False),
        SlotConfig(provider="openrouter", model="deepseek/deepseek-v4-pro", enabled=True),
    ]
    resp = await client.put(
        "/api/config",
        json={"slots": [s.model_dump() for s in slots]},
    )
    assert resp.status_code == 200

    # Verify
    resp2 = await client.get("/api/config")
    data = resp2.json()
    assert data["slots"][0]["base_url_override"] == "http://192.168.1.10:11434"
    assert data["slots"][1]["base_url_override"] == "http://192.168.1.11:11434"
    assert data["slots"][2]["base_url_override"] is None


# ── OpenAI-compatible endpoints (/v1/*) ────────────────────────────────────────


@pytest.mark.anyio
async def test_v1_models_returns_model(client: AsyncClient):
    """GET /v1/models returns the fusion-panel model."""
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    model_ids = [m["id"] for m in data["data"]]
    assert "fusion-panel" in model_ids


@pytest.mark.anyio
async def test_v1_chat_completions_success(client: AsyncClient):
    """POST /v1/chat/completions returns a chat completion with real usage."""
    slots = [
        SlotConfig(provider="openrouter", model="test/model", enabled=True),
        SlotConfig(provider="ollama", model="", enabled=False),
        SlotConfig(provider="openrouter", model="", enabled=False),
        SlotConfig(provider="openrouter", model="", enabled=False),
        SlotConfig(provider="openrouter", model="", enabled=False),
    ]
    await client.put(
        "/api/config",
        json={"openrouter_key": "sk-test", "slots": [s.model_dump() for s in slots]},
    )
    with respx.mock:
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Hello!"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
            )
        )
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "fusion-panel",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "Hello!"}
    assert data["usage"]["total_tokens"] == 5


@pytest.mark.anyio
async def test_v1_chat_completions_provider_error_is_502(client: AsyncClient):
    """Provider failure surfaces as an OpenAI-style error object, not a 200."""
    # Default config: slot 0 is OpenRouter with no key set
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "fusion-panel",
            "messages": [{"role": "user", "content": "Say hello"}],
        },
    )
    assert resp.status_code == 502
    data = resp.json()
    assert "OpenRouter API key not set" in data["error"]["message"]


@pytest.mark.anyio
async def test_v1_chat_completions_requires_auth_when_key_set(client: AsyncClient):
    """When private_api_key is set, /v1/* requires Bearer token."""
    # Set a private key
    await client.put(
        "/api/config",
        json={"private_api_key": "my-secret-key"},
    )

    # Without auth header → 401
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert resp.status_code == 401
    assert "Authorization" in resp.json()["detail"]

    # Wrong key → 401
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
        },
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 401

    # Correct key → reaches the panel (502: no upstream key configured)
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hi"}],
        },
        headers={"Authorization": "Bearer my-secret-key"},
    )
    assert resp.status_code == 502


@pytest.mark.anyio
async def test_v1_synth_mode_error_includes_synth_metadata(client: AsyncClient):
    """When synth mode is on and the synth fails, /v1 returns 502 with fusion_synth metadata."""
    # Enable synth mode
    slots = [
        SlotConfig(provider="openrouter", model="gpt-4o", enabled=True),
        SlotConfig(provider="ollama", model="llama3.2", enabled=False),
        SlotConfig(provider="openrouter", model="synth-model", enabled=True),
        SlotConfig(provider="openrouter", model="", enabled=False),
        SlotConfig(provider="openrouter", model="", enabled=False),
    ]
    await client.put(
        "/api/config",
        json={
            "synth_mode": True,
            "synth_slot": 2,
            "slots": [s.model_dump() for s in slots],
        },
    )

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Say hello"}],
        },
    )
    # No API keys configured → the synth call fails upstream
    assert resp.status_code == 502
    data = resp.json()
    assert "error" in data
    assert data["fusion_synth"]["synth_slot"] == 2
    assert data["fusion_synth"]["synth_model"] == "synth-model"
    assert "responses" in data["fusion_synth"]


# ── Config backward compatibility ──────────────────────────────────────────


@pytest.mark.anyio
async def test_config_backward_compat_judge_to_synth(tmp_path: Path, monkeypatch):
    """Old config with judge_* fields should load correctly as synth_* fields."""
    from fusion_app import config as cfg_module

    test_path = tmp_path / "compat_config.json"
    monkeypatch.setattr(cfg_module, "DEFAULT_CONFIG_PATH", test_path)

    # Write old-style config with judge_ fields
    old_config = {
        "openrouter_key": "",
        "ollama_base_url": "http://localhost:11434",
        "private_api_key": "",
        "judge_mode": True,
        "judge_slot": 3,
        "slots": [
            {"provider": "openrouter", "model": "gpt-4o", "enabled": True},
            {"provider": "ollama", "model": "llama3.2", "enabled": True},
            {"provider": "openrouter", "model": "", "enabled": False},
            {"provider": "openrouter", "model": "", "enabled": False},
            {"provider": "openrouter", "model": "", "enabled": False},
        ],
    }
    test_path.write_text(json.dumps(old_config))

    cfg = load_config(test_path)
    assert cfg.synth_mode is True
    assert cfg.synth_slot == 3


@pytest.mark.anyio
async def test_corrupt_config_is_backed_up(tmp_path: Path, monkeypatch):
    """A corrupt config file is backed up instead of silently discarded."""
    from fusion_app import config as cfg_module

    test_path = tmp_path / "corrupt_config.json"
    monkeypatch.setattr(cfg_module, "DEFAULT_CONFIG_PATH", test_path)
    test_path.write_text("{not valid json")

    cfg = load_config(test_path)
    assert cfg.openrouter_key == ""  # defaults
    backup = test_path.with_name(test_path.name + ".bak")
    assert backup.exists()
    assert backup.read_text() == "{not valid json"


def test_saved_config_is_private(tmp_path: Path):
    """Config file containing keys must be 0600."""
    from fusion_app.config import FusionConfig, save_config

    path = tmp_path / "config.json"
    save_config(FusionConfig(openrouter_key="sk-secret"), path)
    assert (path.stat().st_mode & 0o777) == 0o600

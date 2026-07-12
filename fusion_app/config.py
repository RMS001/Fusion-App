import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ── Config file location ──────────────────────────────────────────────────
DEFAULT_CONFIG_DIR = Path.home() / ".fusion_app"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.json"

# ── Default synth prompt ──────────────────────────────────────────────────
DEFAULT_SYNTH_PROMPT = """\
You are an expert Synthesizer AI. Below are responses from multiple specialized \
AI sub-agents to the same prompt.

Your task is to synthesize the ultimate, optimal response by:
1. Identifying the strongest reasoning, most accurate facts, and best code from each draft
2. Fixing any bugs, hallucinations, or errors present in individual drafts
3. Merging the best elements into a single, clean, authoritative answer
4. Do NOT simply review or rank the responses — produce the FINAL output directly

Write your synthesized answer as if you are the only respondent. Do not reference \
the individual drafts or mention that multiple models were consulted."""


def _validate_http_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"must be an http(s) URL with a host, got {value!r}")
    return value


class SlotConfig(BaseModel):
    """Configuration for a single model slot (0-4)."""

    provider: str = Field(default="openrouter", description="'openrouter' or 'ollama'")
    model: str = Field(default="", description="Model identifier (e.g. 'gpt-4o', 'llama3.2')")
    enabled: bool = Field(default=False, description="Whether this slot is active")
    base_url_override: Optional[str] = Field(
        default=None,
        description="Override URL for this slot (e.g. a specific Ollama node IP). "
        "Falls back to global ollama_base_url if not set.",
    )
    timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description="Per-slot timeout in seconds for non-streaming calls. "
        "Falls back to global slot_timeout if not set.",
    )

    @field_validator("base_url_override")
    @classmethod
    def _check_base_url_override(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        return _validate_http_url(v)


class FusionConfig(BaseModel):
    """Full panel configuration persisted to disk."""

    openrouter_key: str = Field(default="", description="OpenRouter API key")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    private_api_key: str = Field(
        default="",
        description="API key for authenticating calls to /api/* and /v1/* endpoints",
    )
    synth_mode: bool = Field(
        default=False,
        description="When true, every /api/chat and /v1/* call auto-runs the synth flow",
    )
    synth_slot: int = Field(
        default=-1,
        description="Slot index to use as the synthesizer model (-1 = no synth)",
        ge=-1,
        le=4,
    )
    synth_system_prompt: str = Field(
        default=DEFAULT_SYNTH_PROMPT,
        description="System prompt used by the synthesizer model",
    )
    slot_timeout: float = Field(
        default=300.0,
        gt=0,
        description="Max seconds to wait for a single slot's non-streaming response",
    )

    slots: list[SlotConfig] = Field(
        default_factory=lambda: [
            SlotConfig(provider="openrouter", model="openai/gpt-4o", enabled=True),
            SlotConfig(provider="ollama", model="llama3.2", enabled=True),
            SlotConfig(provider="openrouter", model="", enabled=False),
            SlotConfig(provider="openrouter", model="", enabled=False),
            SlotConfig(provider="openrouter", model="", enabled=False),
        ]
    )

    @field_validator("ollama_base_url")
    @classmethod
    def _check_ollama_base_url(cls, v: str) -> str:
        return _validate_http_url(v)


# ── Load / Save ───────────────────────────────────────────────────────────

def load_config(path: Optional[Path] = None) -> FusionConfig:
    """Load config from JSON file, returning defaults if missing.

    Handles backward-compatible migration from old judge_* field names
    to the new synth_* field names. A corrupt config file is backed up
    to <name>.bak instead of being silently discarded.
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    cfg: Optional[FusionConfig] = None
    if path.exists():
        try:
            raw = json.loads(path.read_text())

            # ── Backward compat: migrate old judge_* → synth_* ──
            _migrate_key(raw, "judge_mode", "synth_mode")
            _migrate_key(raw, "judge_slot", "synth_slot")

            cfg = FusionConfig(**raw)
        except Exception as e:
            backup = path.with_name(path.name + ".bak")
            logger.warning(
                "Could not load config from %s (%s) — backing up to %s and using defaults",
                path, e, backup,
            )
            try:
                shutil.copy2(path, backup)
            except OSError:
                pass
    if cfg is None:
        cfg = FusionConfig()
    if not cfg.openrouter_key:
        cfg.openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    return cfg


def _migrate_key(data: dict, old_key: str, new_key: str) -> None:
    """Move an old config key to its new name if present."""
    if old_key in data and new_key not in data:
        data[new_key] = data.pop(old_key)
    elif old_key in data:
        data.pop(old_key)


def save_config(config: FusionConfig, path: Optional[Path] = None) -> None:
    """Persist config to JSON file, readable only by the current user."""
    if path is None:
        path = DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keys are stored in plaintext — keep the file private (0600)
    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)
    path.write_text(config.model_dump_json(indent=2))


def mask_key(key: str) -> str:
    """Mask an API key for display."""
    if len(key) <= 12:
        return "********"
    return key[:4] + "****" + key[-4:]

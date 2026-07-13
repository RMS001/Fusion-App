import asyncio
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from uuid import uuid4

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import FusionConfig, SlotConfig, load_config, mask_key, save_config
from .panel import PanelManager
from .providers import ProviderError

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

# ── App factory ────────────────────────────────────────────────────────────

_config: Optional[FusionConfig] = None
_manager: Optional[PanelManager] = None
_http_client: Optional[httpx.AsyncClient] = None


def _get_http_client() -> httpx.AsyncClient:
    """Shared HTTP client (connection pooling); closed on app shutdown."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(180.0))
    return _http_client


def _get_config() -> FusionConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _get_manager() -> PanelManager:
    global _manager
    if _manager is None:
        _manager = PanelManager(_get_config(), _get_http_client())
    return _manager


def _reload() -> None:
    """Reload config from disk and rebuild the panel manager."""
    global _config, _manager
    _config = load_config()
    _manager = PanelManager(_config, _get_http_client())


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    yield
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


# ── Auth dependency (used by /api/* and /v1/* endpoints) ────────

def _require_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> None:
    """
    If private_api_key is configured, all /api/* and /v1/* requests MUST
    include it as a Bearer token. If no key is configured, allow all.
    """
    cfg = _get_config()
    if not cfg.private_api_key:
        return  # no key configured → open access
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not secrets.compare_digest(credentials.credentials, cfg.private_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── /v1 streaming helpers ────────────────────────────────────────

def _v1_chunk(
    completion_id: str,
    created: int,
    model: str,
    delta: Optional[str] = None,
    finish: Optional[str] = None,
) -> str:
    frame = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta} if delta is not None else {},
                "finish_reason": finish,
            }
        ],
    }
    return f"data: {json.dumps(frame)}\n\n"


HEARTBEAT_INTERVAL = 10.0


async def _stream_deltas(agen, completion_id: str, created: int, model: str):
    """Wrap a plain-text delta generator into OpenAI-style SSE chunks.

    Whenever HEARTBEAT_INTERVAL passes with no delta (drafts being gathered,
    synth model loading or thinking), emit an SSE comment line to keep bytes
    flowing so idle timeouts in clients and proxies don't kill the stream.
    Comment lines are ignored by SSE parsers, so the OpenAI chunk format is
    unaffected. The pending __anext__ is shielded — a heartbeat timeout must
    not cancel the underlying generator.
    """
    ait = agen.__aiter__()
    next_delta: Optional[asyncio.Task] = None
    try:
        while True:
            if next_delta is None:
                next_delta = asyncio.ensure_future(ait.__anext__())
            try:
                delta = await asyncio.wait_for(
                    asyncio.shield(next_delta), timeout=HEARTBEAT_INTERVAL
                )
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
                continue
            except StopAsyncIteration:
                break
            next_delta = None
            if delta:
                yield _v1_chunk(completion_id, created, model, delta=delta)
    except Exception as e:
        err = {"error": {"message": str(e), "type": "upstream_error"}}
        yield f"data: {json.dumps(err)}\n\n"
    finally:
        if next_delta is not None:
            next_delta.cancel()
    yield _v1_chunk(completion_id, created, model, finish="stop")
    yield "data: [DONE]\n\n"


def _v1_error(message: str, code: str, extra: Optional[dict] = None) -> JSONResponse:
    content = {"error": {"message": message, "type": "upstream_error", "code": code}}
    if extra:
        content.update(extra)
    return JSONResponse(status_code=502, content=content)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Fusion App",
        version="1.0.0",
        description="Multi-LLM panel running OpenRouter + Ollama models.",
        lifespan=_lifespan,
    )

    # ── Static files (UI) ──────────────────────────────────────────
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="ui")

    # ── Health ─────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ── Config endpoints ───────────────────────────────────────────

    @app.get("/api/config", dependencies=[Depends(_require_api_key)])
    async def get_config():
        cfg = _get_config()
        return {
            "openrouter_key": mask_key(cfg.openrouter_key),
            "openrouter_key_set": bool(cfg.openrouter_key),
            "ollama_base_url": cfg.ollama_base_url,
            "private_api_key": mask_key(cfg.private_api_key) if cfg.private_api_key else "",
            "private_api_key_set": bool(cfg.private_api_key),
            "synth_mode": cfg.synth_mode,
            "synth_slot": cfg.synth_slot,
            "synth_system_prompt": cfg.synth_system_prompt,
            "slot_timeout": cfg.slot_timeout,
            "slots": [s.model_dump() for s in cfg.slots],
            "tools": {
                **cfg.tools.model_dump(),
                "context7_api_key": mask_key(cfg.tools.context7_api_key)
                if cfg.tools.context7_api_key
                else "",
                "context7_api_key_set": bool(cfg.tools.context7_api_key),
                "brave_api_key": mask_key(cfg.tools.brave_api_key)
                if cfg.tools.brave_api_key
                else "",
                "brave_api_key_set": bool(cfg.tools.brave_api_key),
                "tavily_api_key": mask_key(cfg.tools.tavily_api_key)
                if cfg.tools.tavily_api_key
                else "",
                "tavily_api_key_set": bool(cfg.tools.tavily_api_key),
            },
        }

    class UpdateConfigBody(BaseModel):
        openrouter_key: Optional[str] = None
        ollama_base_url: Optional[str] = None
        private_api_key: Optional[str] = None
        synth_mode: Optional[bool] = None
        synth_slot: Optional[int] = Field(default=None, ge=-1, le=4)
        synth_system_prompt: Optional[str] = None
        slot_timeout: Optional[float] = Field(default=None, gt=0)
        slots: Optional[list[SlotConfig]] = Field(default=None, min_length=5, max_length=5)
        # Partial update: only the keys present are changed, so the UI can
        # omit secret fields it didn't touch instead of echoing masked values.
        tools: Optional[dict] = None

    @app.put("/api/config", dependencies=[Depends(_require_api_key)])
    async def update_config(body: UpdateConfigBody):
        cfg = _get_config()

        # Validate the merged result BEFORE touching shared state, so a
        # rejected update can't leave a half-applied config in memory.
        updates = {
            k: v
            for k, v in body.model_dump(exclude_unset=True).items()
            if v is not None
        }
        if "tools" in updates:
            updates["tools"] = {**cfg.tools.model_dump(), **updates["tools"]}
        try:
            new_cfg = FusionConfig(**{**cfg.model_dump(), **updates})
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))

        save_config(new_cfg)
        _reload()

        return {"status": "saved"}

    # ── Model discovery ────────────────────────────────────────────

    @app.get("/api/models/openrouter", dependencies=[Depends(_require_api_key)])
    async def list_openrouter_models():
        cfg = _get_config()
        if not cfg.openrouter_key:
            return {"models": [], "error": "OpenRouter API key not set"}
        mgr = _get_manager()
        # Create a temporary slot to get the provider
        tmp_slot = SlotConfig(provider="openrouter")
        provider = mgr.get_provider(tmp_slot)
        try:
            models = await provider.list_models()
            return {"models": models}
        except Exception as e:
            return {"models": [], "error": str(e)}

    @app.get("/api/models/ollama", dependencies=[Depends(_require_api_key)])
    async def list_ollama_models(base_url: Optional[str] = None):
        mgr = _get_manager()
        try:
            tmp_slot = SlotConfig(provider="ollama", base_url_override=base_url or None)
            provider = mgr.get_provider(tmp_slot)
            models = await provider.list_models()
            return {"models": models}
        except Exception as e:
            return {"models": [], "error": str(e)}

    @app.get(
        "/api/models/ollama/capabilities", dependencies=[Depends(_require_api_key)]
    )
    async def ollama_model_capabilities(model: str, base_url: Optional[str] = None):
        """Capability badge helper for the UI. Advisory only — the provider's
        400-fallback is the real safety net for non-tool-capable models."""
        mgr = _get_manager()
        try:
            tmp_slot = SlotConfig(provider="ollama", base_url_override=base_url or None)
            provider = mgr.get_provider(tmp_slot)
            caps = await provider.show_capabilities(model)
            return {"capabilities": caps, "tools": "tools" in caps}
        except Exception as e:
            return {"capabilities": [], "tools": None, "error": str(e)}

    # ── Chat endpoints ─────────────────────────────────────────────

    class ChatBody(BaseModel):
        prompt: str = Field(..., description="User message to send")
        system_prompt: Optional[str] = None
        slot: Optional[int] = Field(
            default=None,
            ge=0,
            le=4,
            description="If set, only send to this slot (0-4). If omitted, send to all enabled.",
        )
        synth: bool = Field(
            default=False,
            description="Streaming only: run synthesis after the drafts even if "
            "synth_mode is off (used by the UI's Synthesize button).",
        )

    class ChatResponseItem(BaseModel):
        content: str
        model: str
        latency_ms: float
        error: Optional[str] = None
        usage: Optional[dict] = None
        warning: Optional[str] = None
        tool_trace: Optional[list] = None

    def _serialize(resp) -> dict:
        return ChatResponseItem(
            content=resp.content,
            model=resp.model,
            latency_ms=resp.latency_ms,
            error=resp.error,
            usage=resp.usage,
            warning=resp.warning,
            tool_trace=resp.tool_trace,
        ).model_dump()

    @app.post("/api/chat", dependencies=[Depends(_require_api_key)])
    async def chat(body: ChatBody):
        manager = _get_manager()
        cfg = _get_config()
        messages = [{"role": "user", "content": body.prompt}]
        kwargs = {}
        if body.system_prompt:
            kwargs["system_prompt"] = body.system_prompt

        # Specific slot → always send to that slot
        if body.slot is not None:
            resp = await manager.chat_slot(body.slot, messages, **kwargs)
            return {"slot": body.slot, "response": _serialize(resp)}

        # Synth mode ON → auto-run synth
        if cfg.synth_mode and cfg.synth_slot >= 0:
            synth_result = await manager.synthesize(messages, **kwargs)
            out = {
                label: _serialize(resp)
                for label, resp in synth_result["responses"].items()
            }
            out["synthesis"] = _serialize(synth_result["synthesis"])
            return out

        # Normal mode → send to all enabled slots
        results = await manager.chat_all(messages, **kwargs)
        return {
            key: _serialize(resp) if resp is not None else None
            for key, resp in results.items()
        }

    # ── Synth endpoint ────────────────────────────────────────────

    @app.post("/api/synth", dependencies=[Depends(_require_api_key)])
    async def synth(body: ChatBody):
        """
        Send prompt to all enabled (non-synth) slots, then have the
        configured synth model merge them into a single optimal response.
        """
        manager = _get_manager()
        messages = [{"role": "user", "content": body.prompt}]
        kwargs = {}
        if body.system_prompt:
            kwargs["system_prompt"] = body.system_prompt

        result = await manager.synthesize(messages, **kwargs)
        s = result["synthesis"]

        return {
            "synth_slot": _get_config().synth_slot,
            "synth_model": s.model,
            "responses": {
                label: _serialize(resp) for label, resp in result["responses"].items()
            },
            "synthesis": _serialize(s),
        }

    # ── Streaming SSE endpoint ─────────────────────────────────────

    @app.post("/api/chat/stream", dependencies=[Depends(_require_api_key)])
    async def chat_stream(body: ChatBody):
        manager = _get_manager()
        cfg = _get_config()
        messages = [{"role": "user", "content": body.prompt}]
        kwargs = {}
        if body.system_prompt:
            kwargs["system_prompt"] = body.system_prompt

        # Single-slot streams skip synth; otherwise exclude the synth slot
        # from streaming when synth mode is on.
        only_slot = body.slot
        exclude_synth = (
            cfg.synth_slot
            if (
                only_slot is None
                and (cfg.synth_mode or body.synth)
                and cfg.synth_slot >= 0
            )
            else None
        )

        async def event_generator():
            yield f"event: meta\ndata: {json.dumps({'status': 'started'})}\n\n"

            # Collect full responses while streaming so we can pass them to synth
            collected_responses: dict[str, str] = {}

            async for event in manager.chat_all_stream(
                messages, exclude_slot=exclude_synth, only_slot=only_slot, **kwargs
            ):
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

                # Capture completed responses for synth (avoids double-generation)
                if event["type"] == "done" and event.get("full_content"):
                    slot_idx = event["slot"]
                    slot = cfg.slots[slot_idx]
                    collected_responses[f"Slot {slot_idx} ({slot.model})"] = event["full_content"]

            # If synth was requested, stream it using already-collected responses
            if exclude_synth is not None and collected_responses:
                synth_model = cfg.slots[cfg.synth_slot].model or "synth"
                start_event = {
                    "type": "synth_start",
                    "synth_slot": cfg.synth_slot,
                    "synth_model": synth_model,
                }
                yield f"event: synth_start\ndata: {json.dumps(start_event)}\n\n"

                synth_started = time.monotonic()
                full_content = ""
                synth_error = None
                tool_trace: list = []
                synth_warning = None
                try:
                    async for delta in manager.synthesize_stream_from_collected(
                        messages, collected_responses, yield_events=True, **kwargs
                    ):
                        if isinstance(delta, dict):
                            # Tool-loop liveness/meta events from the synth slot.
                            # "synth": True distinguishes these from draft-slot
                            # tool_call events (which carry a "slot" index).
                            if delta["type"] == "tool_call":
                                payload = {**delta, "synth": True}
                                yield f"event: synth_tool\ndata: {json.dumps(payload)}\n\n"
                            elif delta["type"] == "synth_meta":
                                tool_trace = delta.get("trace") or []
                                synth_warning = delta.get("warning")
                            continue
                        full_content += delta
                        token_event = {"type": "synth_token", "content": delta}
                        yield f"event: synth_token\ndata: {json.dumps(token_event)}\n\n"
                except Exception as e:
                    synth_error = str(e)

                synth_event = {
                    "type": "synth",
                    "synth_slot": cfg.synth_slot,
                    "synth_model": synth_model,
                    "content": full_content if not synth_error else f"[Synth error: {synth_error}]",
                    "error": synth_error,
                    "warning": synth_warning,
                    "tool_trace": tool_trace,
                    "latency_ms": (time.monotonic() - synth_started) * 1000,
                }
                yield f"event: synth\ndata: {json.dumps(synth_event)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    # ── OpenAI-compatible endpoints (/v1/*) ───────────────────────

    @app.get("/v1/models", dependencies=[Depends(_require_api_key)])
    async def v1_list_models():
        """OpenAI-compatible model listing."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "fusion-panel",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "fusion-app",
                }
            ],
        }

    class V1Message(BaseModel):
        model_config = ConfigDict(extra="allow")

        role: str
        content: Optional[str] = None

    class V1ChatBody(BaseModel):
        model: str = "fusion-panel"
        messages: list[V1Message] = Field(..., min_length=1, description="OpenAI message format")
        stream: bool = False
        max_tokens: Optional[int] = Field(default=None, gt=0)

    @app.post("/v1/chat/completions", dependencies=[Depends(_require_api_key)])
    async def v1_chat_completions(body: V1ChatBody):
        """OpenAI-compatible chat completions endpoint. Routes through Fusion panel."""
        manager = _get_manager()
        cfg = _get_config()

        # Forward the FULL message array — preserve multi-turn context
        messages = [m.model_dump(exclude_none=True) for m in body.messages]

        kwargs = {}
        if body.max_tokens:
            kwargs["max_tokens"] = body.max_tokens

        if body.stream:
            return _v1_chat_stream(manager, cfg, messages, kwargs)

        return await _v1_chat_once(manager, cfg, messages, kwargs)

    async def _v1_chat_once(
        manager: PanelManager, cfg: FusionConfig, messages: list[dict], kwargs: dict
    ):
        """Non-streaming response for /v1/chat/completions."""
        empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if cfg.synth_mode and cfg.synth_slot >= 0:
            result = await manager.synthesize(messages, **kwargs)
            s = result["synthesis"]
            responses_meta = {
                label: {"content": r.content, "model": r.model, "error": r.error}
                for label, r in result["responses"].items()
            }
            fusion_meta = {
                "synth_model": s.model,
                "synth_slot": cfg.synth_slot,
                "responses": responses_meta,
                "tool_trace": result.get("tool_trace") or [],
                "warning": s.warning,
            }
            if s.error:
                return _v1_error(s.error, "synth_failed", {"fusion_synth": fusion_meta})
            return {
                "id": "fusion-synth-" + uuid4().hex[:12],
                "object": "chat.completion",
                "created": int(time.time()),
                "model": s.model or "fusion-panel",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": s.content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": s.usage or empty_usage,
                "fusion_synth": fusion_meta,
            }
        else:
            resp = await manager.chat_for_v1(messages, **kwargs)
            if resp.error:
                return _v1_error(resp.error, "provider_error")
            return {
                "id": "fusion-chat-" + uuid4().hex[:12],
                "object": "chat.completion",
                "created": int(time.time()),
                "model": resp.model or "fusion-panel",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": resp.content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": resp.usage or empty_usage,
            }

    def _v1_chat_stream(
        manager: PanelManager, cfg: FusionConfig, messages: list[dict], kwargs: dict
    ) -> StreamingResponse:
        """Streaming SSE response for /v1/chat/completions."""
        created = int(time.time())

        if cfg.synth_mode and cfg.synth_slot >= 0:
            # Stream the synth model's response token-by-token
            # (drafts are gathered internally, only the synth output is streamed)
            model = cfg.slots[cfg.synth_slot].model or "fusion-panel"
            agen = manager.synthesize_stream(messages, **kwargs)
            completion_id = "fusion-synth-stream-" + uuid4().hex[:12]
        else:
            # Stream from the first enabled slot
            slot = next(
                (
                    cfg.slots[i]
                    for i in range(5)
                    if not (cfg.synth_mode and i == cfg.synth_slot)
                    and cfg.slots[i].enabled
                    and cfg.slots[i].model
                ),
                None,
            )
            if slot is None:
                async def _empty():
                    yield "data: [DONE]\n\n"
                return StreamingResponse(
                    _empty(), media_type="text/event-stream", headers=SSE_HEADERS
                )

            model = slot.model
            if manager._slot_tools(slot):
                # Tooled slot: run the (non-streaming) tool loop and emit the
                # final answer as one delta. /v1 streaming stays text-only;
                # _stream_deltas heartbeats cover the silent loop window.
                async def _tooled_single(slot=slot):
                    resp = await manager._chat_slot_routed(slot, messages, **kwargs)
                    if resp.error:
                        raise ProviderError(resp.error)
                    if resp.content:
                        yield resp.content

                agen = _tooled_single()
            else:
                agen = manager.get_provider(slot).chat_stream(
                    messages, slot.model, **kwargs
                )
            completion_id = "fusion-stream-" + uuid4().hex[:12]

        return StreamingResponse(
            _stream_deltas(agen, completion_id, created, model),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    # ── Root redirect → UI ─────────────────────────────────────────

    @app.get("/")
    async def root():
        return RedirectResponse(url="/ui/", status_code=307)

    return app

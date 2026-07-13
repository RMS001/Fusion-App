import asyncio
import logging
import time
from typing import AsyncGenerator, Optional, Union

import httpx

from .config import FusionConfig, SlotConfig
from .providers import ChatResponse, OllamaProvider, OpenRouterProvider
from .tools import Tool, build_tools, dispatch, specs

logger = logging.getLogger(__name__)

STREAM_QUEUE_MAXSIZE = 256

TRACE_PREVIEW_CHARS = 300

# Injected at runtime whenever a slot runs with tools, instead of baked into
# DEFAULT_SYNTH_PROMPT: synth_system_prompt is persisted verbatim into
# config.json, so a changed default would never reach existing installs.
TOOLS_SYSTEM_ADDENDUM = """\
You have fact-checking tools (documentation lookup, web search, URL fetch). \
Before finalizing, you MUST verify with tools any claim that depends on \
current external state, including: dependency/CDN URLs and versions, library \
loading patterns, API signatures, and anything the drafts disagree on. \
Do not trust a draft's URL or API usage because it looks plausible; check it. \
An HTTP 200 on a URL proves it exists, not that it works: for browser and \
library loading (CDN scripts, ES modules, import maps), verify the DOCUMENTED \
loading pattern via documentation lookup before shipping it. \
If tools fail, say nothing about them — fall back to your most conservative \
knowledge and prefer patterns you are certain are long-term stable."""

# Appended transiently to every tool-loop model call (never persisted into the
# transcript): after many tool rounds the system prompt's output contract is
# far away and loses to recency — this keeps it adjacent to the final turn.
TOOLS_FINAL_ANSWER_REMINDER = (
    "Keep calling tools if you still need verification. When you have enough, "
    "immediately write the final, polished answer as the sole respondent — do "
    "not mention drafts, other engineers, tools, verification, or your process."
)


class PanelManager:
    """Orchestrates 5 model slots, fanning out prompts in parallel."""

    def __init__(self, config: FusionConfig, client: httpx.AsyncClient):
        self.config = config
        self._client = client
        self._providers: dict[str, Union[OpenRouterProvider, OllamaProvider]] = {}

    # ── Provider access (lazy init, keyed by URL for multi-node) ────

    def get_provider(self, slot: SlotConfig) -> Union[OpenRouterProvider, OllamaProvider]:
        """Get or create a provider for a slot, keyed by actual endpoint URL."""
        if slot.provider == "openrouter":
            if "openrouter" not in self._providers:
                self._providers["openrouter"] = OpenRouterProvider(
                    self.config.openrouter_key, self._client
                )
            return self._providers["openrouter"]

        elif slot.provider == "ollama":
            url = slot.base_url_override or self.config.ollama_base_url
            provider_key = f"ollama_{url}"
            if provider_key not in self._providers:
                self._providers[provider_key] = OllamaProvider(url, self._client)
            return self._providers[provider_key]

        raise ValueError(f"Unknown provider: {slot.provider}")

    async def _chat_with_timeout(
        self, slot: SlotConfig, messages: list[dict], **kwargs
    ) -> ChatResponse:
        """Run a slot's chat with the configured per-slot deadline."""
        provider = self.get_provider(slot)
        timeout = slot.timeout or self.config.slot_timeout
        try:
            return await asyncio.wait_for(
                provider.chat(messages, slot.model, **kwargs), timeout=timeout
            )
        except asyncio.TimeoutError:
            return ChatResponse(
                content="",
                model=slot.model,
                latency_ms=timeout * 1000,
                error=f"Timed out after {timeout:g}s",
            )

    # ── Agentic tool loop ──────────────────────────────────────────

    def _slot_tools(self, slot: SlotConfig) -> list[Tool]:
        """Tools available to this slot ([] when the slot hasn't opted in)."""
        if not slot.tools_enabled:
            return []
        return build_tools(self.config, self._client)

    async def _chat_with_tools_events(
        self, slot: SlotConfig, messages: list[dict], tools: list[Tool], **kwargs
    ):
        """Non-streaming tool loop as an event generator.

        Yields {"type": "tool_call", name, arguments, result_preview, ms}
        as each tool executes (so streaming consumers can show liveness),
        then exactly one {"type": "final", response, trace, messages}.

        Timeout semantics: each model call gets the per-slot slot_timeout,
        each tool execution gets tools.tool_timeout; the loop itself is
        capped by tools.max_iterations, not a wall clock.
        """
        tool_specs = specs(tools)
        msgs = list(messages)
        trace: list[dict] = []
        warning = None
        resp = None
        reminder = {"role": "user", "content": TOOLS_FINAL_ANSWER_REMINDER}
        for _ in range(self.config.tools.max_iterations):
            # Reminder rides on the call, never on the persisted transcript.
            resp = await self._chat_with_timeout(
                slot, msgs + [reminder], tools=tool_specs, **kwargs
            )
            if resp.warning:
                warning = resp.warning
            if resp.error or not resp.tool_calls:
                break
            msgs.append(
                {
                    "role": "assistant",
                    "content": resp.content or "",
                    "tool_calls": resp.tool_calls,
                }
            )
            for call in resp.tool_calls:
                name = call.get("name", "")
                args = call.get("arguments")
                start = time.monotonic()
                if isinstance(args, dict):
                    result = await dispatch(
                        tools, name, args, self.config.tools.tool_timeout
                    )
                else:
                    result = f"ERROR: malformed JSON in arguments for tool '{name}'"
                elapsed = (time.monotonic() - start) * 1000
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": name,
                        "content": result,
                    }
                )
                entry = {
                    "name": name,
                    "arguments": args if isinstance(args, dict) else None,
                    "result_preview": result[:TRACE_PREVIEW_CHARS],
                    "ms": round(elapsed),
                }
                trace.append(entry)
                yield {"type": "tool_call", **entry}
        else:
            # Iteration cap hit with the model still asking for tools.
            msgs.append(
                {
                    "role": "user",
                    "content": "Tool budget exhausted. Answer the original "
                    "request now with what you have. "
                    + TOOLS_FINAL_ANSWER_REMINDER,
                }
            )
            resp = await self._chat_with_timeout(slot, msgs, **kwargs)

        if warning and not resp.warning:
            resp.warning = warning
        resp.tool_trace = trace
        yield {"type": "final", "response": resp, "trace": trace, "messages": msgs}

    async def _chat_with_tools(
        self, slot: SlotConfig, messages: list[dict], tools: list[Tool], **kwargs
    ) -> tuple[list[dict], ChatResponse, list[dict]]:
        """Run the tool loop to completion. Returns (messages, response, trace)."""
        async for event in self._chat_with_tools_events(slot, messages, tools, **kwargs):
            if event["type"] == "final":
                return event["messages"], event["response"], event["trace"]
        raise RuntimeError("tool loop ended without a final event")  # unreachable

    async def _chat_slot_routed(
        self, slot: SlotConfig, messages: list[dict], **kwargs
    ) -> ChatResponse:
        """One slot call, through the tool loop when the slot has opted in."""
        tools = self._slot_tools(slot)
        if not tools:
            return await self._chat_with_timeout(slot, messages, **kwargs)
        _, resp, _ = await self._chat_with_tools(slot, messages, tools, **kwargs)
        return resp

    # ── Single-shot chat ───────────────────────────────────────────

    async def chat_all(self, messages: list[dict], **kwargs) -> dict[str, Optional[ChatResponse]]:
        """Send to all enabled slots in parallel. Returns slot_0 … slot_4 keys."""

        async def _do(i: int):
            slot = self.config.slots[i]
            if not slot.enabled:
                return i, None
            return i, await self._chat_slot_routed(slot, messages, **kwargs)

        results_list = await asyncio.gather(
            *(_do(i) for i in range(5)), return_exceptions=True
        )

        out: dict[str, Optional[ChatResponse]] = {f"slot_{i}": None for i in range(5)}
        for item in results_list:
            if isinstance(item, Exception):
                continue
            i, resp = item
            out[f"slot_{i}"] = resp

        return out

    async def chat_slot(
        self, slot_index: int, messages: list[dict], **kwargs
    ) -> ChatResponse:
        """Send to a single slot by index (0-4)."""
        if not 0 <= slot_index <= 4:
            raise ValueError(f"Slot index must be 0-4, got {slot_index}")

        slot = self.config.slots[slot_index]
        if not slot.enabled:
            return ChatResponse(
                content="",
                model=slot.model,
                error=f"Slot {slot_index} is disabled",
            )

        return await self._chat_slot_routed(slot, messages, **kwargs)

    # ── Streaming chat (multiplexed over an asyncio.Queue) ─────────

    async def chat_all_stream(
        self,
        messages: list[dict],
        exclude_slot: Optional[int] = None,
        only_slot: Optional[int] = None,
        **kwargs,
    ):
        """
        Async generator yielding streaming events:

        ``{"type": "token", "slot": 0, "content": "Hel"}``
        ``{"type": "done",  "slot": 0, "model": "...", "latency_ms": 1234, "full_content": "..."}``
        ``{"type": "error", "slot": 0, "error": "..."}``

        Args:
            exclude_slot: If set, skip this slot index (used for synth mode).
            only_slot: If set, stream only this slot index.

        Upstream slot tasks are cancelled if the consumer stops iterating
        (e.g. the HTTP client disconnects).
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=STREAM_QUEUE_MAXSIZE)

        async def _stream_slot(i: int):
            slot = self.config.slots[i]
            start = time.monotonic()

            # tools_enabled applies on EVERY path. The loop is non-streaming,
            # so a tooled slot emits live tool_call events for liveness and
            # then its full text in one chunk (best answer over speed). Note
            # its model calls run under slot_timeout per call, unlike pure
            # streaming which has no wall clock.
            tools = self._slot_tools(slot)
            if tools:
                try:
                    async for event in self._chat_with_tools_events(
                        slot, messages, tools, **kwargs
                    ):
                        if event["type"] == "tool_call":
                            entry = {k: v for k, v in event.items() if k != "type"}
                            await queue.put({"type": "tool_call", "slot": i, **entry})
                            continue
                        resp = event["response"]
                        elapsed = (time.monotonic() - start) * 1000
                        if resp.error:
                            await queue.put(
                                {
                                    "type": "error",
                                    "slot": i,
                                    "error": resp.error,
                                    "tool_trace": event["trace"],
                                }
                            )
                        else:
                            await queue.put(
                                {"type": "token", "slot": i, "content": resp.content}
                            )
                            await queue.put(
                                {
                                    "type": "done",
                                    "slot": i,
                                    "model": slot.model,
                                    "latency_ms": elapsed,
                                    "full_content": resp.content,
                                    "tool_trace": event["trace"],
                                    "warning": resp.warning,
                                }
                            )
                except Exception as e:
                    logger.warning("Slot %d tool loop failed: %s", i, e)
                    await queue.put({"type": "error", "slot": i, "error": str(e)})
                return

            provider = self.get_provider(slot)
            full_text = ""

            try:
                async for chunk in provider.chat_stream(messages, slot.model, **kwargs):
                    full_text += chunk
                    await queue.put({"type": "token", "slot": i, "content": chunk})

                elapsed = (time.monotonic() - start) * 1000
                await queue.put(
                    {
                        "type": "done",
                        "slot": i,
                        "model": slot.model,
                        "latency_ms": elapsed,
                        "full_content": full_text,
                    }
                )
            except Exception as e:
                logger.warning("Slot %d stream failed: %s", i, e)
                await queue.put({"type": "error", "slot": i, "error": str(e)})

        indices = [
            i
            for i in range(5)
            if self.config.slots[i].enabled
            and (exclude_slot is None or i != exclude_slot)
            and (only_slot is None or i == only_slot)
        ]
        if not indices:
            yield {"type": "error", "slot": -1, "error": "No slots enabled"}
            return

        tasks = [asyncio.create_task(_stream_slot(i)) for i in indices]

        async def _wait():
            await asyncio.gather(*tasks, return_exceptions=True)
            await queue.put({"type": "all_done"})

        waiter = asyncio.create_task(_wait())
        try:
            while True:
                event = await queue.get()
                if event["type"] == "all_done":
                    break
                yield event
        finally:
            for t in tasks:
                t.cancel()
            waiter.cancel()

    # ── OpenAI-compatible helper ────────────────────────────────

    async def chat_for_v1(
        self, messages: list[dict], **kwargs
    ) -> ChatResponse:
        """
        Pick the first enabled (non-synth, if synth_mode) slot and return its response.
        Used by /v1/chat/completions when synth_mode=False.
        """
        for i in range(5):
            if self.config.synth_mode and i == self.config.synth_slot:
                continue
            slot = self.config.slots[i]
            if slot.enabled and slot.model:
                return await self._chat_slot_routed(slot, messages, **kwargs)
        return ChatResponse(
            content="",
            model="",
            error="No enabled slots with models configured",
        )

    # ── Synthesizer (merge all responses via a designated synth model) ─

    async def _gather_drafts(
        self, messages: list[dict], **kwargs
    ) -> dict[str, ChatResponse]:
        """Get responses from all enabled non-synth slots in parallel."""

        async def _get_draft(i: int):
            slot = self.config.slots[i]
            resp = await self._chat_slot_routed(slot, messages, **kwargs)
            return f"Slot {i} ({slot.model})", resp

        tasks = [
            _get_draft(i)
            for i in range(5)
            if i != self.config.synth_slot and self.config.slots[i].enabled
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses: dict[str, ChatResponse] = {}
        for item in results:
            if isinstance(item, Exception):
                continue
            label, resp = item
            responses[label] = resp
        return responses

    def _synth_config_error(self) -> Optional[ChatResponse]:
        """Return an error ChatResponse if the synth slot is unusable, else None."""
        if self.config.synth_slot < 0 or self.config.synth_slot > 4:
            return ChatResponse(
                content="",
                model="",
                error="No synth slot configured. Set synth_slot to 0-4 in settings.",
            )
        synth_slot = self.config.slots[self.config.synth_slot]
        if not synth_slot.enabled:
            return ChatResponse(
                content="",
                model=synth_slot.model,
                error=f"Synth slot {self.config.synth_slot} is disabled",
            )
        return None

    async def synthesize(
        self, messages: list[dict], **kwargs
    ) -> dict:
        """
        Send prompt to all enabled slots in parallel, then have the synth
        slot merge the best parts into a single optimal answer.

        Returns:
            { "responses": { "Slot N (model)": ChatResponse, ... },
              "synthesis": ChatResponse }
        """
        config_error = self._synth_config_error()
        if config_error:
            return {"responses": {}, "synthesis": config_error}

        synth_slot = self.config.slots[self.config.synth_slot]

        responses = await self._gather_drafts(messages, **kwargs)
        if not responses:
            return {
                "responses": {},
                "synthesis": ChatResponse(
                    content="",
                    model=synth_slot.model,
                    error="No non-synth slots are enabled",
                ),
            }

        tools = self._slot_tools(synth_slot)
        synth_messages = self._build_synth_messages(
            messages, responses, tools_active=bool(tools)
        )
        if tools:
            _, synthesis, trace = await self._chat_with_tools(
                synth_slot, synth_messages, tools, **kwargs
            )
        else:
            synthesis = await self._chat_with_timeout(synth_slot, synth_messages, **kwargs)
            trace = []

        return {
            "responses": responses,
            "synthesis": synthesis,
            "tool_trace": trace,
        }

    async def synthesize_stream(
        self, messages: list[dict], yield_events: bool = False, **kwargs
    ) -> AsyncGenerator:
        """
        Gather draft responses from all enabled slots, build the synth
        meta-prompt, then stream the synth model's response token-by-token.

        Yields plain-text content deltas. With yield_events=True, dict
        events ({"type": "tool_call"} / {"type": "synth_meta"}) are
        interleaved when the synth slot runs with tools.
        """
        config_error = self._synth_config_error()
        if config_error:
            yield f"[Error: {config_error.error}]"
            return

        synth_slot = self.config.slots[self.config.synth_slot]

        # Drafts are gathered non-streaming, since the user only sees the
        # synth output.
        responses = await self._gather_drafts(messages, **kwargs)
        if not responses:
            yield "[Error: No non-synth slots are enabled]"
            return

        async for item in self._stream_synth_turn(
            synth_slot, messages, responses, yield_events, **kwargs
        ):
            yield item

    async def _stream_synth_turn(
        self,
        synth_slot: SlotConfig,
        messages: list[dict],
        responses: dict[str, ChatResponse],
        yield_events: bool,
        **kwargs,
    ):
        """Shared tail of the streaming synth paths.

        Without tools: token deltas straight from chat_stream (no wall-clock
        timeout — tokens are liveness). With tools: the loop is non-streaming,
        so yield dict events for liveness ({"type": "tool_call"} per call,
        one {"type": "synth_meta"} with trace+warning), then the final
        buffered answer as one str chunk. Deliberately NOT regenerating the
        final turn as a stream — on large local synth models that would
        double the synth latency.
        """
        tools = self._slot_tools(synth_slot)
        synth_messages = self._build_synth_messages(
            messages, responses, tools_active=bool(tools)
        )
        if tools:
            async for event in self._chat_with_tools_events(
                synth_slot, synth_messages, tools, **kwargs
            ):
                if event["type"] == "tool_call":
                    if yield_events:
                        yield event
                    continue
                resp = event["response"]
                if yield_events:
                    yield {
                        "type": "synth_meta",
                        "trace": event["trace"],
                        "warning": resp.warning,
                    }
                if resp.error:
                    yield f"[Error: {resp.error}]"
                elif resp.content:
                    yield resp.content
            return

        synth_provider = self.get_provider(synth_slot)
        async for chunk in synth_provider.chat_stream(
            synth_messages, synth_slot.model, **kwargs
        ):
            yield chunk

    def _build_synth_messages(
        self,
        original_messages: list[dict],
        responses: dict[str, ChatResponse],
        tools_active: bool = False,
    ) -> list[dict]:
        """Build the message array for the synthesizer model."""
        user_prompt = original_messages[-1]["content"] if original_messages else "(no prompt)"
        parts = [
            f"# Original Prompt\n{user_prompt}\n",
            "# Draft Responses\n",
        ]
        for label, resp in responses.items():
            msg = resp.error if resp.error else resp.content
            parts.append(f"## {label}\n{msg}\n")

        # Strategy-neutral: how to weigh the drafts is entirely the
        # synth_system_prompt's job — a merge/solve-first directive here would
        # override it from last position.
        parts.append("\nProduce your final response to the original prompt now.")

        system = self.config.synth_system_prompt
        if tools_active:
            system = f"{system}\n\n{TOOLS_SYSTEM_ADDENDUM}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(parts)},
        ]

    async def synthesize_from_collected(
        self, messages: list[dict], collected_responses: dict[str, str], **kwargs
    ) -> dict:
        """
        Run synthesis using pre-collected response texts (avoids double-generation
        when called after streaming).

        Args:
            collected_responses: dict of "Slot N (model)" -> response text
        """
        config_error = self._synth_config_error()
        if config_error:
            return {"synthesis": config_error}

        synth_slot = self.config.slots[self.config.synth_slot]

        responses = {
            label: ChatResponse(content=text, model=label)
            for label, text in collected_responses.items()
        }

        tools = self._slot_tools(synth_slot)
        synth_messages = self._build_synth_messages(
            messages, responses, tools_active=bool(tools)
        )
        if tools:
            _, synthesis, trace = await self._chat_with_tools(
                synth_slot, synth_messages, tools, **kwargs
            )
        else:
            synthesis = await self._chat_with_timeout(synth_slot, synth_messages, **kwargs)
            trace = []

        return {"synthesis": synthesis, "tool_trace": trace}

    async def synthesize_stream_from_collected(
        self,
        messages: list[dict],
        collected_responses: dict[str, str],
        yield_events: bool = False,
        **kwargs,
    ) -> AsyncGenerator:
        """
        Stream the synth model's response token-by-token using pre-collected
        draft texts (the streaming twin of synthesize_from_collected).

        Yields plain-text content deltas (plus dict events when
        yield_events=True and the synth slot runs with tools). No wall-clock
        timeout: the tokens themselves are the liveness signal, matching the
        other streaming paths.
        """
        config_error = self._synth_config_error()
        if config_error:
            yield f"[Error: {config_error.error}]"
            return

        synth_slot = self.config.slots[self.config.synth_slot]

        responses = {
            label: ChatResponse(content=text, model=label)
            for label, text in collected_responses.items()
        }

        async for item in self._stream_synth_turn(
            synth_slot, messages, responses, yield_events, **kwargs
        ):
            yield item

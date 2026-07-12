import asyncio
import logging
import time
from typing import AsyncGenerator, Optional, Union

import httpx

from .config import FusionConfig, SlotConfig
from .providers import ChatResponse, OllamaProvider, OpenRouterProvider

logger = logging.getLogger(__name__)

STREAM_QUEUE_MAXSIZE = 256


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

    # ── Single-shot chat ───────────────────────────────────────────

    async def chat_all(self, messages: list[dict], **kwargs) -> dict[str, Optional[ChatResponse]]:
        """Send to all enabled slots in parallel. Returns slot_0 … slot_4 keys."""

        async def _do(i: int):
            slot = self.config.slots[i]
            if not slot.enabled:
                return i, None
            return i, await self._chat_with_timeout(slot, messages, **kwargs)

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

        return await self._chat_with_timeout(slot, messages, **kwargs)

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
            provider = self.get_provider(slot)
            full_text = ""
            start = time.monotonic()

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
                return await self._chat_with_timeout(slot, messages, **kwargs)
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
            resp = await self._chat_with_timeout(slot, messages, **kwargs)
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

        synth_messages = self._build_synth_messages(messages, responses)
        synthesis = await self._chat_with_timeout(synth_slot, synth_messages, **kwargs)

        return {
            "responses": responses,
            "synthesis": synthesis,
        }

    async def synthesize_stream(
        self, messages: list[dict], **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Gather draft responses from all enabled slots, build the synth
        meta-prompt, then stream the synth model's response token-by-token.

        Yields plain-text content deltas.
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

        synth_messages = self._build_synth_messages(messages, responses)
        synth_provider = self.get_provider(synth_slot)
        async for chunk in synth_provider.chat_stream(synth_messages, synth_slot.model, **kwargs):
            yield chunk

    def _build_synth_messages(
        self, original_messages: list[dict], responses: dict[str, ChatResponse]
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

        return [
            {"role": "system", "content": self.config.synth_system_prompt},
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

        synth_messages = self._build_synth_messages(messages, responses)
        synthesis = await self._chat_with_timeout(synth_slot, synth_messages, **kwargs)

        return {"synthesis": synthesis}

    async def synthesize_stream_from_collected(
        self, messages: list[dict], collected_responses: dict[str, str], **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream the synth model's response token-by-token using pre-collected
        draft texts (the streaming twin of synthesize_from_collected).

        Yields plain-text content deltas. No wall-clock timeout: the tokens
        themselves are the liveness signal, matching the other streaming paths.
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

        synth_messages = self._build_synth_messages(messages, responses)
        synth_provider = self.get_provider(synth_slot)
        async for chunk in synth_provider.chat_stream(
            synth_messages, synth_slot.model, **kwargs
        ):
            yield chunk

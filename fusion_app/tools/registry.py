import asyncio
import logging

import httpx

from ..config import FusionConfig
from .base import Tool
from .context7 import GetLibraryDocsTool, ResolveLibraryTool
from .web import WebFetchTool, WebSearchTool

logger = logging.getLogger(__name__)


def build_tools(config: FusionConfig, client: httpx.AsyncClient) -> list[Tool]:
    """Instantiate the enabled, available tools for one request."""
    t = config.tools
    tools: list[Tool] = []
    if t.context7_enabled:
        tools.append(ResolveLibraryTool(t, client))
        tools.append(GetLibraryDocsTool(t, client))
    if t.web_enabled:
        tools.append(WebSearchTool(t, client))
        tools.append(WebFetchTool(t, client))
    return [tool for tool in tools if tool.available]


def specs(tools: list[Tool]) -> list[dict]:
    """The OpenAI-style array passed to providers."""
    return [t.spec() for t in tools]


async def dispatch(tools: list[Tool], name: str, args: dict, timeout: float) -> str:
    """Run one tool call. Never raises — unknown tools, timeouts, and
    internal failures all come back as 'ERROR: ...' strings the model
    can adapt to."""
    for tool in tools:
        if tool.name == name:
            try:
                return await asyncio.wait_for(tool.execute(args), timeout=timeout)
            except asyncio.TimeoutError:
                return f"ERROR: tool '{name}' timed out after {timeout:g}s"
            except Exception as e:
                logger.warning("Tool %s failed: %s", name, e)
                return f"ERROR: tool '{name}' failed: {e}"
    return f"ERROR: unknown tool '{name}'"

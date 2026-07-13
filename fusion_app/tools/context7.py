"""Context7 documentation-lookup tools.

API (verified 2026-07-12 against context7.com/docs/api-reference):
  GET https://context7.com/api/v2/libs/search?libraryName=<name>&query=<q>
  GET https://context7.com/api/v2/context?libraryId=</org/proj>&query=<q>&type=txt
Anonymous access is allowed; an optional `ctx7sk-...` key raises rate limits
(Authorization: Bearer header).
"""

import httpx

from ..config import ToolsConfig
from .base import Tool

CONTEXT7_SEARCH_URL = "https://context7.com/api/v2/libs/search"
CONTEXT7_CONTEXT_URL = "https://context7.com/api/v2/context"
REQUEST_TIMEOUT = 25.0
MAX_SEARCH_RESULTS = 5


class _Context7Tool(Tool):
    def __init__(self, cfg: ToolsConfig, client: httpx.AsyncClient):
        self._cfg = cfg
        self._client = client

    def _headers(self) -> dict:
        if self._cfg.context7_api_key:
            return {"Authorization": f"Bearer {self._cfg.context7_api_key}"}
        return {}

    async def _get(self, url: str, params: dict) -> httpx.Response:
        return await self._client.get(
            url, params=params, headers=self._headers(), timeout=REQUEST_TIMEOUT
        )


class ResolveLibraryTool(_Context7Tool):
    name = "resolve_library"
    description = (
        "Look up a library/framework in the Context7 documentation index and "
        "return candidate library IDs. Call this first, then pass the best ID "
        "to get_library_docs. Example: library_name='three.js' returns IDs "
        "like '/mrdoob/three.js'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "library_name": {
                "type": "string",
                "description": "Library or framework name, e.g. 'react', 'three.js'",
            },
            "query": {
                "type": "string",
                "description": "What you want to know — used to rank results",
            },
        },
        "required": ["library_name"],
    }

    async def execute(self, args: dict) -> str:
        library_name = str(args.get("library_name") or "").strip()
        if not library_name:
            return "ERROR: resolve_library requires 'library_name'"
        query = str(args.get("query") or library_name)
        try:
            resp = await self._get(
                CONTEXT7_SEARCH_URL,
                {"libraryName": library_name, "query": query, "fast": "true"},
            )
            if resp.status_code != 200:
                return f"ERROR: context7 {resp.status_code}: {resp.text[:200]}"
            results = (resp.json().get("results") or [])[:MAX_SEARCH_RESULTS]
        except Exception as e:
            return f"ERROR: context7 request failed: {e}"
        if not results:
            return f"No libraries found for {library_name!r}"
        lines = []
        for r in results:
            lines.append(
                f"{r.get('id')} — {r.get('title')} "
                f"(trust {r.get('trustScore')}, {r.get('totalSnippets')} snippets): "
                f"{(r.get('description') or '')[:150]}"
            )
        return "\n".join(lines)


class GetLibraryDocsTool(_Context7Tool):
    name = "get_library_docs"
    description = (
        "Fetch current, authoritative documentation snippets for a library. "
        "Use the library_id returned by resolve_library (e.g. '/mrdoob/three.js'). "
        "Use this to verify current APIs, import/loading patterns, and versions "
        "instead of trusting memory or drafts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "library_id": {
                "type": "string",
                "description": "Context7 library ID from resolve_library, e.g. '/facebook/react'",
            },
            "query": {
                "type": "string",
                "description": "The specific question or topic, e.g. 'load via CDN importmap'",
            },
        },
        "required": ["library_id", "query"],
    }

    async def execute(self, args: dict) -> str:
        library_id = str(args.get("library_id") or "").strip()
        query = str(args.get("query") or "").strip()
        if not library_id or not query:
            return "ERROR: get_library_docs requires 'library_id' and 'query'"
        try:
            resp = await self._get(
                CONTEXT7_CONTEXT_URL,
                {"libraryId": library_id, "query": query, "type": "txt"},
            )
            if resp.status_code != 200:
                return f"ERROR: context7 {resp.status_code}: {resp.text[:200]}"
            text = resp.text
        except Exception as e:
            return f"ERROR: context7 request failed: {e}"
        # Local models have small contexts — a doc dump crowds out reasoning.
        limit = self._cfg.web_fetch_max_chars
        if len(text) > limit:
            text = text[:limit] + "\n[...truncated]"
        return text or "No documentation returned"

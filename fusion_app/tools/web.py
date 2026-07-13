"""web_search + web_fetch tools.

SSRF posture: unless config allows private networks, every URL — including
every redirect hop — must resolve only to global addresses. This server may
sit next to Ollama nodes and other LAN services; the model must not be able
to probe them.
"""

import asyncio
import html as html_lib
import ipaddress
import re
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx

from ..config import ToolsConfig
from .base import Tool

try:
    from ddgs import DDGS  # current package name (renamed from duckduckgo_search)
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

try:
    import trafilatura
except ImportError:
    trafilatura = None

MAX_REDIRECTS = 5
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
REQUEST_TIMEOUT = 20.0
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
REDIRECT_STATUSES = (301, 302, 303, 307, 308)


async def check_url_allowed(url: str, allow_private: bool) -> Optional[str]:
    """Return an error string if the URL must not be fetched, else None.

    Rejects non-http(s) schemes and hosts that resolve to any non-global
    address (loopback, RFC1918, link-local incl. 169.254.169.254 metadata,
    CGNAT/Tailscale 100.64/10, reserved).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"ERROR: only http/https URLs are allowed, got {parsed.scheme!r}"
    host = parsed.hostname
    if not host:
        return "ERROR: URL has no host"
    if allow_private:
        return None
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except OSError:
        return f"ERROR: cannot resolve host {host!r}"
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if not ip.is_global:
            return f"ERROR: host {host!r} resolves to a private/reserved address; refusing to fetch"
    return None


def _strip_tags(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", html_lib.unescape(text)).strip()


def _extract_text(html: str) -> str:
    if trafilatura is not None:
        try:
            text = trafilatura.extract(html)
            if text:
                return text
        except Exception:
            pass
    return _strip_tags(html)


class WebFetchTool(Tool):
    name = "web_fetch"
    description = (
        "Fetch a URL. Use mode='head' to check whether a URL (e.g. a CDN "
        "script path) actually resolves before recommending it — it returns "
        "the status, final URL, and content-type only. mode='auto' (default) "
        "downloads the page and returns its readable text."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Absolute http(s) URL to fetch"},
            "mode": {
                "type": "string",
                "enum": ["auto", "head"],
                "description": "'head' = existence/metadata check only; 'auto' = fetch and extract text",
            },
        },
        "required": ["url"],
    }

    def __init__(self, cfg: ToolsConfig, client: httpx.AsyncClient):
        self._cfg = cfg
        self._client = client

    async def execute(self, args: dict) -> str:
        url = str(args.get("url") or "").strip()
        if not url:
            return "ERROR: web_fetch requires 'url'"
        mode = args.get("mode") or "auto"
        try:
            return await self._fetch(url, mode)
        except httpx.HTTPError as e:
            return f"ERROR: fetch failed: {e}"

    async def _fetch(self, url: str, mode: str) -> str:
        # Manual redirect loop: every hop is re-validated, so a public URL
        # cannot bounce the request onto the LAN.
        for _ in range(MAX_REDIRECTS + 1):
            err = await check_url_allowed(url, self._cfg.allow_private_networks)
            if err:
                return err

            if mode == "head":
                resp = await self._client.head(
                    url, follow_redirects=False, timeout=REQUEST_TIMEOUT
                )
                if resp.status_code in (405, 501):
                    resp = await self._request_get(url, read_body=False)
            else:
                resp = await self._request_get(url, read_body=True)

            location = resp.headers.get("location")
            if resp.status_code in REDIRECT_STATUSES and location:
                url = urljoin(url, location)
                continue

            return self._render(resp, url, mode)
        return f"ERROR: more than {MAX_REDIRECTS} redirects"

    async def _request_get(self, url: str, read_body: bool) -> httpx.Response:
        async with self._client.stream(
            "GET", url, follow_redirects=False, timeout=REQUEST_TIMEOUT
        ) as resp:
            if read_body and resp.status_code not in REDIRECT_STATUSES:
                body = b""
                async for chunk in resp.aiter_bytes():
                    body += chunk
                    if len(body) >= MAX_RESPONSE_BYTES:
                        break
                resp._content = body
            else:
                resp._content = b""
        return resp

    def _render(self, resp: httpx.Response, final_url: str, mode: str) -> str:
        status_line = f"HTTP {resp.status_code} {final_url}"
        if mode == "head":
            return (
                f"{status_line}\n"
                f"content-type: {resp.headers.get('content-type', '?')}\n"
                f"content-length: {resp.headers.get('content-length', '?')}"
            )
        content_type = resp.headers.get("content-type", "")
        body = resp.content.decode(resp.charset_encoding or "utf-8", errors="replace")
        if "html" in content_type:
            text = _extract_text(body)
        else:
            text = body.strip()
        limit = self._cfg.web_fetch_max_chars
        if len(text) > limit:
            text = text[:limit] + "\n[...truncated]"
        return f"{status_line}\n\n{text}" if text else status_line


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web. Returns a list of title / url / snippet results. "
        "Follow up with web_fetch on promising URLs to verify claims."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {
                "type": "integer",
                "description": "Number of results (default 5)",
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: ToolsConfig, client: httpx.AsyncClient):
        self._cfg = cfg
        self._client = client
        backend = cfg.web_search_backend
        self.available = not (
            (backend == "duckduckgo" and DDGS is None)
            or (backend == "searxng" and not cfg.searxng_base_url)
            or (backend == "brave" and not cfg.brave_api_key)
            or (backend == "tavily" and not cfg.tavily_api_key)
        )

    async def execute(self, args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return "ERROR: web_search requires 'query'"
        n = args.get("max_results") or 5
        n = max(1, min(int(n), 10))
        backend = self._cfg.web_search_backend
        try:
            results = await getattr(self, f"_search_{backend}")(query, n)
        except Exception as e:
            return f"ERROR: {backend} search failed: {e}"
        if not results:
            return "No results"
        return "\n".join(
            f"{title}\n{url}\n{snippet[:300]}\n" for title, url, snippet in results
        )

    async def _search_duckduckgo(self, query: str, n: int) -> list[tuple]:
        def _run():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=n))

        raw = await asyncio.to_thread(_run)
        return [(r.get("title", ""), r.get("href", ""), r.get("body", "")) for r in raw]

    async def _search_searxng(self, query: str, n: int) -> list[tuple]:
        resp = await self._client.get(
            f"{self._cfg.searxng_base_url.rstrip('/')}/search",
            params={"q": query, "format": "json"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        raw = (resp.json().get("results") or [])[:n]
        return [(r.get("title", ""), r.get("url", ""), r.get("content", "")) for r in raw]

    async def _search_brave(self, query: str, n: int) -> list[tuple]:
        resp = await self._client.get(
            BRAVE_SEARCH_URL,
            params={"q": query, "count": n},
            headers={"X-Subscription-Token": self._cfg.brave_api_key},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        raw = ((resp.json().get("web") or {}).get("results") or [])[:n]
        return [
            (r.get("title", ""), r.get("url", ""), r.get("description", "")) for r in raw
        ]

    async def _search_tavily(self, query: str, n: int) -> list[tuple]:
        resp = await self._client.post(
            TAVILY_SEARCH_URL,
            json={"api_key": self._cfg.tavily_api_key, "query": query, "max_results": n},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        raw = (resp.json().get("results") or [])[:n]
        return [(r.get("title", ""), r.get("url", ""), r.get("content", "")) for r in raw]

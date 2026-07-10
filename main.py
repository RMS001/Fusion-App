#!/usr/bin/env python3
"""
Fusion App — Multi-LLM Panel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Run 5 model slots side-by-side (OpenRouter cloud + Ollama local).

Usage:
    python main.py              # Start server on http://localhost:8000
    python main.py --port 8080  # Custom port
    python main.py --host 0.0.0.0  # Listen on all interfaces
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `fusion_app` is importable
HERE = Path(__file__).parent.resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import uvicorn
from fusion_app.api import create_app

app = create_app()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fusion App — Multi-LLM Panel")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Allow binding to a non-loopback address without a private API key",
    )
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.insecure:
        from fusion_app.config import load_config

        if not load_config().private_api_key:
            parser.error(
                f"refusing to bind to {args.host}: no private API key is configured, so the "
                "config and chat endpoints would be exposed unauthenticated to the network. "
                "Set a Private API Key in Settings first, or pass --insecure to override."
            )

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

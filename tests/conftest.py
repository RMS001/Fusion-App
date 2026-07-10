from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from fusion_app.api import create_app
from fusion_app.config import FusionConfig, save_config


@pytest.fixture(autouse=True)
def _patch_config_path(tmp_path: Path, monkeypatch):
    """Redirect config to a temp file so tests don't touch real config."""
    from fusion_app import config as cfg_module

    test_path = tmp_path / "config.json"
    monkeypatch.setattr(cfg_module, "DEFAULT_CONFIG_PATH", test_path)
    # Keep tests hermetic — never pick up a real key from the environment
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # Start each test with a fresh empty config
    save_config(FusionConfig())

    from fusion_app import api
    # Reset global state so next _get_config() loads from our fresh file
    api._config = None
    api._manager = None
    yield
    api._config = None
    api._manager = None


@pytest.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)  # type: ignore
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

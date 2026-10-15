"""Regression tests for fixes introduced by the code audit."""

import pytest
from fastapi.testclient import TestClient

from aiforge.data.json_utils import parse_llm_json, strip_code_fences


@pytest.mark.parametrize("raw", ['[1, 2]', '```json\n[1, 2]\n```', '```\n[1, 2]```', '  [1, 2]  '])
def test_parse_llm_json_handles_fences(raw):
    assert parse_llm_json(raw) == [1, 2]


def test_strip_code_fences_none_safe():
    assert strip_code_fences(None) == ""


def test_ui_rejects_foreign_host():
    from aiforge.ui.app import app
    with TestClient(app, base_url="http://evil.example.com") as client:
        assert client.post("/api/server/stop").status_code == 403
        assert client.get("/api/experiments").status_code == 403


def test_ui_rejects_foreign_origin_on_mutation():
    from aiforge.ui.app import app
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        res = client.post("/api/forge/start", json={"name": "x", "model": "m", "topic": "t"},
                          headers={"Origin": "http://evil.example.com"})
        assert res.status_code == 403


def test_ui_start_server_unknown_experiment_is_404(tmp_path, monkeypatch):
    monkeypatch.setenv("AIFORGE_REGISTRY_PATH", str(tmp_path / "reg"))
    import aiforge.ui.app
    from importlib import reload
    reload(aiforge.ui.app)
    with TestClient(aiforge.ui.app.app) as client:
        assert client.post("/api/server/start", json={"experiment_id": "missing"}).status_code == 404


def test_ui_download_rejects_unsupported_model():
    from aiforge.ui.app import app
    with TestClient(app) as client:
        assert client.post("/api/models/download", json={"model_id": "attacker/huge"}).status_code == 400

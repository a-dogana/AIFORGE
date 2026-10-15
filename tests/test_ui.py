import pytest
import os
from fastapi.testclient import TestClient

def test_ui_dashboard_load(tmp_path):
    from aiforge.ui.app import app
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        assert "AIFORGE Dashboard" in res.text

def test_ui_api_status(tmp_path):
    from aiforge.ui.app import app
    with TestClient(app) as client:
        res = client.get("/api/status")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert "llm_server" in data

def test_ui_api_experiments_empty(tmp_path, monkeypatch):
    registry_file = str(tmp_path / "reg_dir")
    monkeypatch.setenv("AIFORGE_REGISTRY_PATH", registry_file)
    # Reload app to pick up new env var
    import aiforge.ui.app
    from importlib import reload
    reload(aiforge.ui.app)
    
    with TestClient(aiforge.ui.app.app) as client:
        res = client.get("/api/experiments")
        assert res.status_code == 200
        assert res.json() == []

def test_ui_api_experiments_populated(tmp_path, monkeypatch):
    registry_file = str(tmp_path / "reg_dir")
    monkeypatch.setenv("AIFORGE_REGISTRY_PATH", registry_file)
    from aiforge.experiments.registry import ExperimentRegistry
    from aiforge.core.config import ExperimentConfig
    r = ExperimentRegistry(registry_file)
    cfg = ExperimentConfig(experiment_name="Test Exp", domain="llm", task="generation", architecture={"type": "llm"}, data={"path": "dummy", "type": "jsonl"})
    r.save_experiment(cfg, {}, {"acc": 0.99}, "cpu", experiment_id="exp_1")
    
    import aiforge.ui.app
    from importlib import reload
    reload(aiforge.ui.app)
    
    with TestClient(aiforge.ui.app.app) as client:
        res = client.get("/api/experiments")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 1
        assert data[0]["id"] == "exp_1"
        
        # Test detail
        res_detail = client.get("/api/experiments/exp_1")
        assert res_detail.status_code == 200
        assert res_detail.json()["config"]["experiment_name"] == "Test Exp"
        
        # Test missing
        res_missing = client.get("/api/experiments/exp_2")
        assert res_missing.status_code == 404

def test_ui_api_generate_offline(monkeypatch):
    import aiforge.ui.app
    with TestClient(aiforge.ui.app.app) as client:
        # Assuming server is offline in test env
        res = client.post("/api/generate", json={"prompt": "test", "max_new_tokens": 10})
        assert res.status_code == 503
        assert "non raggiungibile" in res.json()["detail"].lower()
        
        # Test validation
        res_val = client.post("/api/generate", json={"prompt": "test", "max_new_tokens": "invalid"})
        assert res_val.status_code == 422

def test_ui_api_generate_online(monkeypatch):
    import aiforge.ui.app
    import httpx
    
    # Mock httpx.AsyncClient.post
    class MockResponse:
        status_code = 200
        def json(self):
            return {"text": "def mock():\n    pass", "inference_ms": 15.0}
            
    async def mock_post(*args, **kwargs):
        return MockResponse()
        
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    
    with TestClient(aiforge.ui.app.app) as client:
        res = client.post("/api/generate", json={"prompt": "def mock():", "max_new_tokens": 10})
        assert res.status_code == 200
        assert "def mock():" in res.json()["text"]


def test_ui_api_export_gguf(tmp_path, monkeypatch):
    import aiforge.ui.app
    from unittest.mock import MagicMock

    mock_exporter = MagicMock()
    mock_exporter.export.return_value = {
        "status": "success",
        "experiment_id": "test_exp",
        "quantization": "Q4_K_M",
        "target_gguf_path": "/fake/path/model.gguf"
    }

    monkeypatch.setattr("aiforge.deployment.gguf_export.GGUFExporter.export", mock_exporter.export)

    with TestClient(aiforge.ui.app.app) as client:
        res = client.post("/api/export/gguf", json={"experiment_id": "test_exp", "quantization": "Q4_K_M"})
        assert res.status_code == 200
        assert res.json()["status"] == "success"
        assert res.json()["quantization"] == "Q4_K_M"

def test_ui_api_merge_preflight_post(monkeypatch):
    import aiforge.ui.app
    from aiforge.core.merge_lab import MergeLab
    from unittest.mock import MagicMock
    
    mock_validate = MagicMock(return_value={"status": "COMPATIBLE", "estimated_ram_mb": 150, "warnings": [], "incompatibilities": []})
    monkeypatch.setattr(MergeLab, "validate_pre_merge", mock_validate)
    
    with TestClient(aiforge.ui.app.app) as client:
        res = client.post("/api/merge/preflight", json={
            "strategy": "slerp",
            "model_a_id": "BASE:openai-community/gpt2",
            "model_b_id": "BASE:openai-community/gpt2",
            "alpha": 0.5
        })
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "COMPATIBLE"
        assert data["estimated_ram_mb"] == 150

def test_ui_api_connectors_test(monkeypatch):
    import aiforge.ui.app
    import httpx
    
    class MockResp:
        status_code = 200
        
    async def mock_get(*args, **kwargs):
        return MockResp()
        
    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    
    with TestClient(aiforge.ui.app.app) as client:
        res = client.post("/api/connectors/test", json={
            "base_url": "http://127.0.0.1:11434/v1"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "Connesso" in data["message"]




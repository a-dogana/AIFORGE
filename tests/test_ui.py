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
        assert "offline" in res.json()["detail"].lower()
        
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


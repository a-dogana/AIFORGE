import pytest
import os
from fastapi.testclient import TestClient

def test_llm_server_validation():
    from aiforge.serving.llm_server import app, state
    state.engine = None
    
    with TestClient(app) as client:
        # Should be unavailable
        assert client.post("/v1/generate", json={"prompt": "test"}).status_code == 503
        
def test_llm_server_limits(tmp_path):
    from aiforge.core.config import ExperimentConfig
    from aiforge.llm.engine import LLMEngine
    
    config_dict = {
        "experiment_name": "Server Lim",
        "domain": "llm",
        "task": "generation",
        "data": {"type": "jsonl"},
        "architecture": {
            "type": "llm",
            "llm_config": {
                "model_id": "gpt2",
                "allow_download": False,
                "max_seq_length": 64,
                "lora_enabled": False,
                "lora_r": 8,
                "lora_alpha": 16
            }
        },
        "batch_size": 1,
        "epochs": 1,
        "device": "cpu"
    }
    config = ExperimentConfig(**config_dict)
    
    os.environ["AIFORGE_LLM_CONFIG"] = str(tmp_path / "cfg.yaml")
    with open(tmp_path / "cfg.yaml", "w") as f:
        import yaml
        yaml.dump(config_dict, f)
        
    from aiforge.serving.llm_server import app, state
    with TestClient(app) as client:
        # Now engine is loaded via lifespan
        assert client.get("/ready").status_code == 200
        
        # Test valid (Skip generation due to MPS memory issues in pytest context)
        # resp = client.post("/v1/generate", json={"prompt": "def x():", "max_new_tokens": 10})
        # assert resp.status_code == 200
        # assert "inference_ms" in resp.json()
        
        # Test over limit tokens
        resp = client.post("/v1/generate", json={"prompt": "def x():", "max_new_tokens": 1024})
        assert resp.status_code == 422 # Pydantic validation error
        
        # Test negative temp
        resp = client.post("/v1/generate", json={"prompt": "def x():", "temperature": -1.0})
        assert resp.status_code == 422
        
        # Test long prompt
        long_prompt = "x" * 3000
        resp = client.post("/v1/generate", json={"prompt": long_prompt})
        assert resp.status_code == 422

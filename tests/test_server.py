import pytest
import os
import shutil
from fastapi.testclient import TestClient

@pytest.fixture
def temp_registry(tmp_path):
    from aiforge.experiments.registry import ExperimentRegistry
    reg_dir = tmp_path / "registry"
    os.makedirs(reg_dir)
    reg = ExperimentRegistry(str(reg_dir))
    yield reg
    shutil.rmtree(reg_dir)

def test_server_without_package():
    # If no package is set, app should start but not be ready
    if "AIFORGE_DEPLOY_PACKAGE" in os.environ:
        del os.environ["AIFORGE_DEPLOY_PACKAGE"]
        
    from aiforge.serving.server import app, state
    # Reset state explicitly for tests
    state.engine = None
    
    with TestClient(app) as client:
        # Health should work
        resp = client.get("/health")
        assert resp.status_code == 200
        
        # Ready should fail
        resp = client.get("/ready")
        assert resp.status_code == 503
        
        # Model should fail
        resp = client.get("/v1/model")
        assert resp.status_code == 503
        
        # Predict should fail
        resp = client.post("/v1/predict", json={"data": [[0.0]]})
        assert resp.status_code == 503

def test_server_with_package(temp_registry, tmp_path):
    from aiforge.core.config import ExperimentConfig
    from aiforge.models import ArchitectureEngine
    from aiforge.deployment.deployer import DeploymentEngine
    
    # 1. Setup a dummy experiment and export it
    config_dict = {
        "experiment_name": "Server Test",
        "data": {"type": "digits_builtin"},
        "architecture": {"type": "mlp", "mlp_config": {"input_size": 2, "hidden_sizes": [2], "output_size": 2}},
        "batch_size": 1,
        "epochs": 1,
        "device": "cpu"
    }
    config = ExperimentConfig(**config_dict)
    dummy_model = ArchitectureEngine.get_model(config)
    exp_id = temp_registry.save_experiment(config, {"best_model_state": dummy_model.state_dict()}, {}, "cpu")
    
    target_dir = str(tmp_path / "deploy_pkg")
    deployer = DeploymentEngine(registry_dir=temp_registry.registry_dir)
    deployer.export(exp_id, target_dir, formats=["pytorch"])
    
    # 2. Inject it into the server
    os.environ["AIFORGE_DEPLOY_PACKAGE"] = target_dir
    os.environ["AIFORGE_DEPLOY_BACKEND"] = "pytorch"
    
    from aiforge.serving.server import app, state
    
    # Use context manager to trigger lifespan startup
    with TestClient(app) as client:
        # Ready should pass
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"
        
        # Metadata
        resp = client.get("/v1/model")
        assert resp.status_code == 200
        assert resp.json()["architecture"] == "mlp"
        
        # Predict
        resp = client.post("/v1/predict", json={"data": [[1.0, 2.0]]})
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        
        # Predict batch too large
        large_batch = [[0.0, 0.0]] * 1025
        resp = client.post("/v1/predict", json={"data": large_batch})
        assert resp.status_code == 400
        
        # Metrics
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.json()["requests_total"] >= 1

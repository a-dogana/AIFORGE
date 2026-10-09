import os
import shutil
import pytest
import torch
import numpy as np
from aiforge.core.config import ExperimentConfig
from aiforge.deployment.deployer import DeploymentEngine
from aiforge.deployment.inference import InferenceEngine

@pytest.fixture
def temp_registry(tmp_path):
    from aiforge.experiments.registry import ExperimentRegistry
    reg_dir = tmp_path / "registry"
    os.makedirs(reg_dir)
    reg = ExperimentRegistry(str(reg_dir))
    yield reg
    shutil.rmtree(reg_dir)

def test_deployment_export_and_inference(temp_registry, tmp_path):
    # Setup dummy experiment
    config_dict = {
        "experiment_name": "Deploy Test",
        "data": {"type": "digits_builtin"},
        "architecture": {"type": "mlp", "mlp_config": {"input_size": 10, "hidden_sizes": [8], "output_size": 2}},
        "batch_size": 4,
        "epochs": 1,
        "device": "cpu"
    }
    config = ExperimentConfig(**config_dict)
    
    from aiforge.models import ArchitectureEngine
    dummy_model = ArchitectureEngine.get_model(config)
    exp_id = temp_registry.save_experiment(config, {"best_model_state": dummy_model.state_dict()}, {}, "cpu", preprocessing_artifacts={"scaler": {"mean": [0]*10, "scale": [1]*10}})
    
    # Export
    target_dir = tmp_path / "deploy_pkg"
    deployer = DeploymentEngine(registry_dir=temp_registry.registry_dir)
    formats = ["pytorch", "torchscript"]
    
    try:
        import onnx
        import onnxruntime
        formats.append("onnx")
    except ImportError:
        pass
        
    manifest = deployer.export(exp_id, str(target_dir), formats=formats)
    
    assert "pytorch" in manifest["formats"]
    assert "torchscript" in manifest["formats"]
    assert os.path.exists(target_dir / "model.pt")
    assert os.path.exists(target_dir / "model.ptc")
    assert os.path.exists(target_dir / "manifest.json")
    assert os.path.exists(target_dir / "architecture.json")
    
    # Inference Parity
    dummy_input = np.random.randn(2, 10).astype(np.float32)
    
    engine_pt = InferenceEngine(str(target_dir), backend="pytorch")
    out_pt = engine_pt.predict(dummy_input)
    assert out_pt.shape == (2, 2)
    
    engine_ts = InferenceEngine(str(target_dir), backend="torchscript")
    out_ts = engine_ts.predict(dummy_input)
    
    np.testing.assert_allclose(out_pt, out_ts, rtol=1e-5, atol=1e-5)
    
    if "onnx" in formats:
        engine_onnx = InferenceEngine(str(target_dir), backend="onnx")
        out_onnx = engine_onnx.predict(dummy_input)
        np.testing.assert_allclose(out_pt, out_onnx, rtol=1e-4, atol=1e-4)

def test_missing_backend(tmp_path):
    manifest = {
        "formats": ["pytorch"]
    }
    os.makedirs(tmp_path / "pkg")
    import json
    with open(tmp_path / "pkg" / "manifest.json", "w") as f:
        json.dump(manifest, f)
        
    with pytest.raises(ValueError, match="is not supported by this deployment package"):
        InferenceEngine(str(tmp_path / "pkg"), backend="onnx")

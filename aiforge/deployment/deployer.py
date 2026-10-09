import os
import json
import torch
from typing import List, Dict, Any
from aiforge.experiments.registry import ExperimentRegistry
from aiforge.core.config import ExperimentConfig
from aiforge.models import ArchitectureEngine
from aiforge.data.dataset import DataEngine

class DeploymentEngine:
    """
    Packages a trained AIFORGE experiment into a portable deployment format.
    """
    def __init__(self, registry_dir: str = "registry"):
        self.registry = ExperimentRegistry(registry_dir)

    def export(self, experiment_id: str, target_dir: str, formats: List[str] = ["pytorch"]):
        """
        Exports the experiment into a self-contained package.
        """
        print(f"Loading experiment {experiment_id}...")
        record, model_state = self.registry.load_experiment(experiment_id)
        if model_state is None:
            raise ValueError(f"No model state found for experiment {experiment_id}.")
            
        config_dict = record["config"]
        config = ExperimentConfig(**config_dict)
        
        os.makedirs(target_dir, exist_ok=True)
        
        # Build model and load weights
        model = ArchitectureEngine.get_model(config)
        model.load_state_dict(model_state)
        model.eval()
        
        # Dummy input for tracing
        dummy_input = self._get_dummy_input(config)
        
        manifest = {
            "experiment_id": experiment_id,
            "experiment_name": record.get("experiment_name", "Unknown"),
            "schema_version": "1.0",
            "task": config.task,
            "domain": config.domain,
            "architecture": config.architecture.type,
            "formats": [],
            "preprocessing_artifacts": record.get("preprocessing_artifacts", {})
        }
        
        # Native PyTorch Export
        if "pytorch" in formats:
            pt_path = os.path.join(target_dir, "model.pt")
            torch.save(model.state_dict(), pt_path)
            manifest["formats"].append("pytorch")
            # We also save a minimal architecture JSON for native loading
            arch_path = os.path.join(target_dir, "architecture.json")
            with open(arch_path, "w") as f:
                json.dump(config_dict["architecture"], f, indent=2)
            print(f"Exported PyTorch weights to {pt_path}")
            
        # ONNX Export
        if "onnx" in formats:
            try:
                import onnx
                onnx_path = os.path.join(target_dir, "model.onnx")
                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
                )
                manifest["formats"].append("onnx")
                print(f"Exported ONNX graph to {onnx_path}")
            except ImportError:
                print("WARNING: ONNX is not installed. Skipping ONNX export.")
            except Exception as e:
                print(f"WARNING: ONNX export failed: {e}")
                
        # TorchScript Export
        if "torchscript" in formats:
            try:
                ts_path = os.path.join(target_dir, "model.ptc")
                scripted_model = torch.jit.script(model)
                scripted_model.save(ts_path)
                manifest["formats"].append("torchscript")
                print(f"Exported TorchScript model to {ts_path}")
            except Exception as e:
                print(f"WARNING: TorchScript export failed: {e}")

        # Save manifest
        manifest_path = os.path.join(target_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Deployment package created at {target_dir}")
        return manifest
        
    def _get_dummy_input(self, config: ExperimentConfig) -> torch.Tensor:
        """
        Creates a dummy tensor for tracing based on the config.
        """
        if config.architecture.type == "mlp":
            in_size = config.architecture.mlp_config.input_size
            return torch.randn(1, in_size)
        elif config.architecture.type == "cnn":
            in_channels = config.architecture.cnn_config.in_channels
            # Assume a default 32x32 size for trace if not explicitly known
            # In AIFORGE v0.4, image size is defined in image_config
            img_size = [32, 32]
            if config.data.type == "images":
                img_size = config.data.image_config.image_size
            return torch.randn(1, in_channels, img_size[0], img_size[1])
        else:
            raise ValueError(f"Unknown architecture type {config.architecture.type}")

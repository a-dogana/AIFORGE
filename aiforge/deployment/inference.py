import os
import json
import torch
import numpy as np
from typing import Dict, Any, Union, List

class InferenceEngine:
    """
    Loads a portable deployment package and provides an inference API.
    Does not depend on the training configuration or original dataset.
    """
    def __init__(self, package_dir: str, backend: str = "pytorch"):
        self.package_dir = package_dir
        self.backend = backend
        
        manifest_path = os.path.join(package_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found in {package_dir}")
            
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
            
        if backend not in self.manifest.get("formats", []):
            raise ValueError(f"Backend '{backend}' is not supported by this deployment package. Supported: {self.manifest.get('formats')}")
            
        self.artifacts = self.manifest.get("preprocessing_artifacts", {})
        self.architecture_type = self.manifest.get("architecture")
        
        self.model = None
        self.ort_session = None
        
        self._load_model()
        
    def _load_model(self):
        if self.backend == "pytorch":
            from aiforge.models import ArchitectureEngine
            from aiforge.core.config import ExperimentConfig
            
            # Load partial config just for architecture
            arch_path = os.path.join(self.package_dir, "architecture.json")
            with open(arch_path, "r") as f:
                arch_dict = json.load(f)
            
            # Reconstruct architecture config
            # We mock the other fields since they are not needed for inference
            mock_config = ExperimentConfig(
                data={"type": "digits_builtin"}, # Mock
                architecture=arch_dict,
                batch_size=1,
                epochs=1
            )
            
            self.model = ArchitectureEngine.get_model(mock_config)
            
            pt_path = os.path.join(self.package_dir, "model.pt")
            # Safe weights_only=True loading
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
        elif self.backend == "onnx":
            try:
                import onnxruntime as ort
                onnx_path = os.path.join(self.package_dir, "model.onnx")
                self.ort_session = ort.InferenceSession(onnx_path)
            except ImportError:
                raise RuntimeError("onnxruntime is not installed. Cannot use ONNX backend.")
                
        elif self.backend == "torchscript":
            ts_path = os.path.join(self.package_dir, "model.ptc")
            self.model = torch.jit.load(ts_path, map_location="cpu")
            self.model.eval()
            
    def _preprocess_tabular(self, data: Union[List[List[float]], np.ndarray]) -> torch.Tensor:
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
            
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
            
        # Apply standard scaler if present
        if "scaler" in self.artifacts:
            mean = np.array(self.artifacts["scaler"]["mean"])
            scale = np.array(self.artifacts["scaler"]["scale"])
            data = (data - mean) / scale
            
        return torch.tensor(data, dtype=torch.float32)

    def _preprocess_images(self, data: torch.Tensor) -> torch.Tensor:
        # Assumes data is already in correct C, H, W format
        if len(data.shape) == 3:
            data = data.unsqueeze(0)
        return data

    def predict(self, inputs: Union[List[List[float]], np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Runs inference on the preprocessed inputs.
        """
        # Preprocessing
        if self.manifest.get("domain") == "vision" and self.architecture_type == "cnn":
            if not isinstance(inputs, torch.Tensor):
                raise ValueError("CNN inference expects a PyTorch Tensor of images (B, C, H, W)")
            tensor_input = self._preprocess_images(inputs)
        else:
            # Tabular / MLP
            tensor_input = self._preprocess_tabular(inputs)
            
        # Inference
        if self.backend in ["pytorch", "torchscript"]:
            with torch.no_grad():
                logits = self.model(tensor_input)
                return logits.numpy()
                
        elif self.backend == "onnx":
            ort_inputs = {self.ort_session.get_inputs()[0].name: tensor_input.numpy()}
            ort_outs = self.ort_session.run(None, ort_inputs)
            return ort_outs[0]
            
        return np.array([])

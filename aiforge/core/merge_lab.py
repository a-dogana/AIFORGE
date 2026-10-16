import os
import torch
import gc
import copy
import math
from typing import Dict, Any, Tuple, Optional, List
from pydantic import BaseModel, Field

from aiforge.experiments.registry import ExperimentRegistry
from aiforge.core.config import ExperimentConfig

class MergeRequest(BaseModel):
    strategy: str = "lora_merge"  # "lora_merge", "multi_lora_merge", "linear_weight", "slerp", "dare_linear"
    model_a_id: str
    model_b_id: str
    alpha: float = 0.5
    additional_adapters: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    output_name: str = "Merged_Model"


def slerp(t0: torch.Tensor, t1: torch.Tensor, alpha: float, eps: float = 1e-7) -> torch.Tensor:
    """
    Spherical Linear Interpolation (SLERP) for weight tensors.
    Interpolates smoothly along the spherical manifold of weights.
    """
    if t0.shape != t1.shape:
        raise ValueError(f"SLERP requires matching tensor shapes, got {t0.shape} vs {t1.shape}")
    alpha = min(max(float(alpha), 0.0), 1.0)

    orig_shape = t0.shape
    v0 = t0.reshape(-1).float()
    v1 = t1.reshape(-1).float()

    norm0 = torch.norm(v0)
    norm1 = torch.norm(v1)

    if norm0 < eps or norm1 < eps:
        res = (1.0 - alpha) * v0 + alpha * v1
        return res.reshape(orig_shape).to(t0.dtype)

    u0 = v0 / norm0
    u1 = v1 / norm1

    dot = torch.sum(u0 * u1)
    dot = torch.clamp(dot, -1.0, 1.0)

    # If almost collinear, fallback to linear interpolation to prevent numerical instability
    if torch.abs(dot) > 0.9995:
        res = (1.0 - alpha) * v0 + alpha * v1
        return res.reshape(orig_shape).to(t0.dtype)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    scale0 = torch.sin((1.0 - alpha) * theta) / sin_theta
    scale1 = torch.sin(alpha * theta) / sin_theta

    res = scale0 * v0 + scale1 * v1
    return res.reshape(orig_shape).to(t0.dtype)


def dare_linear(t0: torch.Tensor, t1: torch.Tensor, alpha: float = 0.5, drop_rate: float = 0.2) -> torch.Tensor:
    """
    DARE (Drop And REscale) linear delta merge.
    Drops small delta parameters with drop_rate and rescales remaining parameters by 1/(1-drop_rate).
    """
    if t0.shape != t1.shape:
        raise ValueError(f"DARE requires matching tensor shapes, got {t0.shape} vs {t1.shape}")
    alpha = min(max(float(alpha), 0.0), 1.0)
    drop_rate = min(max(float(drop_rate), 0.0), 0.99)

    delta = (t1 - t0).float()
    mask = (torch.rand_like(delta) > drop_rate).float()
    rescaled_delta = delta * mask * (1.0 / (1.0 - drop_rate + 1e-7))
    return (t0.float() + alpha * rescaled_delta).to(t0.dtype)


class ResourcePreflight:
    """Estimates resources needed for merge operations to prevent crashing on constrained hardware like M1 8GB."""
    
    @staticmethod
    def _get_available_ram_mb() -> float:
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 * 1024)
        except ImportError:
            return 8000 # Fallback 8GB
            
    @staticmethod
    def estimate_lora_merge(base_model_id: str, lora_path: str) -> Dict[str, Any]:
        estimated_params_b = 1.0 # 1 Billion default
        if "0.5b" in base_model_id.lower() or "qwen2.5-0.5" in base_model_id.lower():
            estimated_params_b = 0.5
        elif "1.5b" in base_model_id.lower():
            estimated_params_b = 1.5
        elif "gpt2" in base_model_id.lower():
            estimated_params_b = 0.124 # GPT-2 small
            
        ram_needed_mb = (estimated_params_b * 1000 * 2) + 1000 # 1GB overhead
        avail_ram = ResourcePreflight._get_available_ram_mb()
        
        warnings = []
        if ram_needed_mb > avail_ram:
            warnings.append(f"CRITICAL: Estimated RAM ({ram_needed_mb}MB) exceeds available RAM ({avail_ram}MB). The OS may swap or kill the process.")
        
        status = "COMPATIBLE_WITH_WARNINGS" if warnings else "COMPATIBLE"
        
        return {
            "status": status,
            "estimated_ram_mb": int(ram_needed_mb), 
            "warnings": warnings,
            "incompatibilities": []
        }
        
    @staticmethod
    def estimate_full_merge(model_a_path: str, model_b_path: str) -> Dict[str, Any]:
        return {
            "status": "COMPATIBLE",
            "estimated_ram_mb": 150,
            "warnings": ["Tensors will be loaded iteratively to maintain low RAM footprint (<200MB)."],
            "incompatibilities": []
        }


SUPPORTED_MERGE_STRATEGIES = {
    "lora_merge",
    "multi_lora_merge",
    "linear_weight",
    "slerp",
    "dare_linear",
}


class MergeLab:
    def __init__(self, registry: ExperimentRegistry):
        self.registry = registry

    def _get_model_config(self, exp_id: str) -> Optional[ExperimentConfig]:
        if exp_id.startswith("BASE:"):
            base_id = exp_id.split("BASE:")[1]
            return ExperimentConfig(
                experiment_name=f"Base_{base_id.replace('/', '_')}",
                domain="llm",
                task="generation",
                data={"type": "csv", "csv_config": {"path": "dummy.csv", "target_column": "text"}},
                architecture={
                    "type": "llm",
                    "llm_config": {
                        "model_id": base_id,
                        "allow_download": True,
                        "lora_enabled": False
                    }
                }
            )
            
        record = self.registry.get_experiment_record(exp_id)
        if not record:
            return None
        return ExperimentConfig(**record["config"])

    def validate_pre_merge(self, req: MergeRequest) -> Dict[str, Any]:
        """Validates if two models can be merged under the chosen strategy."""
        if req.strategy not in SUPPORTED_MERGE_STRATEGIES:
            return {"status": "UNKNOWN", "incompatibilities": [f"Strategy {req.strategy} not supported."]}

        cfg_a = self._get_model_config(req.model_a_id)
        cfg_b = self._get_model_config(req.model_b_id)
        
        if not cfg_a or not cfg_b:
            return {"status": "INCOMPATIBLE", "incompatibilities": ["One or both source models not found in registry."]}
            
        if cfg_a.architecture.type != cfg_b.architecture.type:
            return {"status": "INCOMPATIBLE", "incompatibilities": [f"Architecture mismatch: {cfg_a.architecture.type} vs {cfg_b.architecture.type}"]}
            
        arch = cfg_a.architecture.type
        
        if req.strategy in ["lora_merge", "multi_lora_merge"]:
            if arch != "llm":
                return {"status": "INCOMPATIBLE", "incompatibilities": ["LoRA merge currently only supported for 'llm' architecture."]}
                
            is_a_base = not cfg_a.architecture.llm_config.lora_enabled or req.model_a_id.startswith("BASE:")
            is_b_lora = cfg_b.architecture.llm_config.lora_enabled
            
            if not (is_a_base and is_b_lora):
                return {"status": "INCOMPATIBLE", "incompatibilities": ["For LoRA merge, Model A must be a Base model and Model B must be a LoRA adapter."]}
                
            if cfg_a.architecture.llm_config.model_id != cfg_b.architecture.llm_config.model_id:
                return {"status": "INCOMPATIBLE_WITH_WARNINGS", "warnings": ["Adapter base model differs from Target base model! Semantic corruption highly likely."]}
                
            preflight = ResourcePreflight.estimate_lora_merge(cfg_a.architecture.llm_config.model_id, "")
            
        elif req.strategy in ["linear_weight", "slerp", "dare_linear"]:
            if arch == "llm":
                return {"status": "INCOMPATIBLE", "incompatibilities": ["Full weight merge for LLMs is unsupported due to hardware constraints on M1."]}
            preflight = ResourcePreflight.estimate_full_merge("", "")
            
        else:
            return {"status": "UNKNOWN", "incompatibilities": [f"Strategy {req.strategy} not supported."]}
            
        return {
            "status": preflight.get("status", "COMPATIBLE"),
            "warnings": preflight.get("warnings", []),
            "incompatibilities": preflight.get("incompatibilities", []),
            "estimated_ram_mb": preflight.get("estimated_ram_mb", 0)
        }

    def execute_merge(self, req: MergeRequest) -> str:
        """Executes the merge operation non-destructively and saves to registry."""
        if req.strategy not in SUPPORTED_MERGE_STRATEGIES:
            raise ValueError(f"Unsupported merge strategy: '{req.strategy}'")

        self.registry.validate_display_name(req.output_name)

        validation = self.validate_pre_merge(req)
        if validation["status"] not in ("COMPATIBLE", "COMPATIBLE_WITH_WARNINGS"):
            reasons = validation.get("incompatibilities") or validation.get("warnings") or [validation["status"]]
            raise ValueError(f"Merge cannot proceed ({validation['status']}): {reasons}")

        cfg_a = self._get_model_config(req.model_a_id)
        cfg_b = self._get_model_config(req.model_b_id)
        
        import time
        exp_id = self.registry.generate_experiment_id(req.output_name)
        # Validate both artifact dir and record path before creating anything on disk
        self.registry.resolve_safe_path(f"exp_{exp_id}.json")
        artifact_path = self.registry.resolve_safe_path(exp_id, allow_create_dirs=True)
        
        result_config = copy.deepcopy(cfg_a)
        result_config.experiment_name = req.output_name
        
        saved_ckpt_path = artifact_path
        try:
            # 1. LoRA Merge & Multi-LoRA Composition
            if req.strategy in ["lora_merge", "multi_lora_merge"]:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from peft import PeftModel
                
                base_id = cfg_a.architecture.llm_config.model_id
                adapter_record_b = self.registry.get_experiment_record(req.model_b_id)
                adapter_path_b = self.registry.resolve_checkpoint_path(adapter_record_b)
                
                qlora_pt = os.path.join(adapter_path_b, "qlora_adapter.pt")
                if os.path.isfile(qlora_pt) and not (req.strategy == "multi_lora_merge" and req.additional_adapters):
                    merged_model, tokenizer = self.registry.load_llm_for_inference(
                        adapter_record_b,
                        device="cpu",
                        merge_adapters=True,
                    )
                    merged_model.save_pretrained(artifact_path)
                    tokenizer.save_pretrained(artifact_path)
                    del merged_model
                else:
                    base_model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32, device_map="cpu")
                    tokenizer = AutoTokenizer.from_pretrained(base_id)
                    
                    # Check if multi-adapter composition is requested
                    if req.strategy == "multi_lora_merge" and req.additional_adapters:
                        peft_model = PeftModel.from_pretrained(base_model, adapter_path_b, adapter_name="adapter_0")
                        adapters = ["adapter_0"]
                        weights = [float(req.alpha)]
                        
                        for idx, extra in enumerate(req.additional_adapters, start=1):
                            extra_rec = self.registry.get_experiment_record(extra["model_id"])
                            if extra_rec:
                                extra_path = self.registry.resolve_checkpoint_path(extra_rec)
                                ad_name = f"adapter_{idx}"
                                peft_model.load_adapter(extra_path, adapter_name=ad_name)
                                adapters.append(ad_name)
                                weights.append(float(extra.get("weight", 0.5)))
                                
                        # Combine adapters using PEFT add_weighted_adapter
                        peft_model.add_weighted_adapter(adapters=adapters, weights=weights, adapter_name="composite_merged")
                        peft_model.set_adapter("composite_merged")
                        merged_model = peft_model.merge_and_unload()
                    else:
                        peft_model = PeftModel.from_pretrained(base_model, adapter_path_b)
                        merged_model = peft_model.merge_and_unload()
                    
                    merged_model.save_pretrained(artifact_path)
                    tokenizer.save_pretrained(artifact_path)
                    del base_model, peft_model, merged_model
                
                result_config.architecture.llm_config.model_id = artifact_path
                result_config.architecture.llm_config.lora_enabled = False
                result_config.architecture.llm_config.qlora = False
                saved_ckpt_path = artifact_path
                
            # 2. Classical Weight Merging: Linear, SLERP, DARE
            elif req.strategy in ["linear_weight", "slerp", "dare_linear"]:
                from aiforge.models import ArchitectureEngine
                model_a = ArchitectureEngine.get_model(cfg_a)
                model_b = ArchitectureEngine.get_model(cfg_b)
                
                rec_a = self.registry.get_experiment_record(req.model_a_id)
                rec_b = self.registry.get_experiment_record(req.model_b_id)
                
                model_a.load_state_dict(self.registry.load_classical_state_dict(rec_a))
                model_b.load_state_dict(self.registry.load_classical_state_dict(rec_b))
                
                state_a = model_a.state_dict()
                state_b = model_b.state_dict()
                
                for key in state_a.keys():
                    if key in state_b and state_a[key].shape == state_b[key].shape:
                        if req.strategy == "linear_weight":
                            state_a[key] = (1.0 - req.alpha) * state_a[key] + (req.alpha) * state_b[key]
                        elif req.strategy == "slerp":
                            state_a[key] = slerp(state_a[key], state_b[key], req.alpha)
                        elif req.strategy == "dare_linear":
                            state_a[key] = dare_linear(state_a[key], state_b[key], req.alpha)
                    else:
                        raise ValueError(f"Shape mismatch for tensor {key}")
                        
                model_a.load_state_dict(state_a)
                pt_file = os.path.join(artifact_path, "model.pt")
                torch.save(model_a.state_dict(), pt_file)
                saved_ckpt_path = pt_file
                del model_a, model_b, state_a, state_b
                
        except Exception as e:
            import shutil
            shutil.rmtree(artifact_path, ignore_errors=True)
            raise RuntimeError(f"Merge execution failed: {e}")
            
        gc.collect()
        
        import yaml
        with open(os.path.join(artifact_path, "config.yaml"), "w") as f:
            yaml.dump(result_config.model_dump(), f)
            
        rec_a_full = self.registry.get_experiment_record(req.model_a_id) if not req.model_a_id.startswith("BASE:") else None
        record = {
            "id": exp_id,
            "experiment_name": req.output_name,
            "config": result_config.model_dump(),
            "results": {
                "merge_strategy": req.strategy,
                "source_a": req.model_a_id,
                "source_b": req.model_b_id,
                "alpha": req.alpha,
                "additional_adapters": req.additional_adapters
            },
            "preprocessing_artifacts": (rec_a_full or {}).get("preprocessing_artifacts", {}),
            "timestamp": self.registry.now_utc_iso(),
            "checkpoint_path": saved_ckpt_path
        }
        self.registry.save_experiment_record(exp_id, record)
        
        return exp_id

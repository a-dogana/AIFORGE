import os
import re
import json
import uuid
import datetime
import torch
from typing import Dict, Any, Tuple, Optional, List, Union
from aiforge.core.config import ExperimentConfig

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class ExperimentRegistry:
    def __init__(self, registry_dir: str = "registry"):
        self.registry_dir = os.path.abspath(registry_dir)
        self.registry_path = self.registry_dir
        if os.path.lexists(self.registry_dir) and os.path.islink(self.registry_dir):
            raise ValueError(f"Security: Registry directory cannot be a symlink: '{registry_dir}'")
        os.makedirs(self.registry_dir, exist_ok=True)
        self.checkpoints_dir = self.resolve_safe_path("checkpoints", allow_create_dirs=True)

    def _validate_id(self, experiment_id: str) -> str:
        if not experiment_id or not isinstance(experiment_id, str):
            raise ValueError("Experiment ID must be a non-empty string.")
        if (
            ".." in experiment_id
            or "/" in experiment_id
            or "\\" in experiment_id
            or ":" in experiment_id
            or "\x00" in experiment_id
            or os.path.isabs(experiment_id)
            or _WIN_DRIVE_RE.match(experiment_id)
            or not _SAFE_ID_RE.match(experiment_id)
        ):
            raise ValueError(f"Security: Invalid or unsafe experiment ID: '{experiment_id}'")
        return experiment_id

    def validate_display_name(self, name: str) -> str:
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError("Experiment display name must be a non-empty string.")
        if (
            ".." in name
            or "/" in name
            or "\\" in name
            or "\x00" in name
            or os.path.isabs(name)
            or _WIN_DRIVE_RE.match(name.strip())
        ):
            raise ValueError(f"Security: Unsafe path characters in experiment name: '{name}'")
        return name

    def generate_experiment_id(self, display_name: Optional[str] = None) -> str:
        """Generate a safe, opaque experiment ID decoupled from the display name."""
        if display_name is not None:
            self.validate_display_name(display_name)
        return uuid.uuid4().hex[:8]

    def resolve_safe_path(self, *parts: str, allow_create_dirs: bool = False) -> str:
        """Resolve a path inside ``self.registry_dir`` and reject traversal, absolute paths, or symlinks."""
        reg_abs = os.path.abspath(self.registry_dir)
        if os.path.lexists(reg_abs) and os.path.islink(reg_abs):
            raise ValueError(f"Security: Registry root is a symlink: '{reg_abs}'")
        reg_real = os.path.realpath(reg_abs)

        cur = reg_abs
        for part in parts:
            self._validate_id(part)
            cur = os.path.join(cur, part)
            if os.path.lexists(cur):
                if os.path.islink(cur):
                    raise ValueError(f"Security: Symlink detected inside registry: '{cur}'")
                cur_real = os.path.realpath(cur)
                if os.path.commonpath([reg_real, cur_real]) != reg_real:
                    raise ValueError(f"Security: Path escapes registry directory: '{cur}'")

        parent_real = os.path.realpath(os.path.dirname(cur))
        if os.path.commonpath([reg_real, parent_real]) != reg_real:
            raise ValueError(f"Security: Parent path escapes registry directory: '{cur}'")

        if allow_create_dirs:
            os.makedirs(cur, exist_ok=True)
            if os.path.islink(cur) or os.path.commonpath([reg_real, os.path.realpath(cur)]) != reg_real:
                raise ValueError(f"Security: Created directory escapes registry: '{cur}'")

        return cur

    def save_experiment(
        self,
        config: ExperimentConfig,
        training_results: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        device_used: str,
        preprocessing_artifacts: Dict[str, Any] = None,
        experiment_id: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> str:
        """
        Saves the experiment configuration, results, best model state, last training state, and preprocessing artifacts.
        Contract:
        - MLP/CNN: checkpoint_path is a .pt file path.
        - LLM: checkpoint_path is a directory path containing model or adapter artifacts.
        """
        if preprocessing_artifacts is None:
            preprocessing_artifacts = {}
        training_results = dict(training_results or {})
        display_name = getattr(config, "experiment_name", "N/A")
        if display_name and display_name != "N/A":
            self.validate_display_name(display_name)
        if not experiment_id:
            experiment_id = self.generate_experiment_id(display_name if display_name != "N/A" else None)
        else:
            experiment_id = self._validate_id(experiment_id)

        # Pre-validate target record path before writing any checkpoint files
        file_path = self.resolve_safe_path(f"exp_{experiment_id}.json")
        timestamp = datetime.datetime.now().isoformat()
        
        best_model_state = training_results.pop("best_model_state", None)
        last_training_state = training_results.pop("last_training_state", None)
        # Handle v0.2 compatibility where it was called 'model_state'
        if not best_model_state and "model_state" in training_results:
             best_model_state = training_results.pop("model_state")
             
        popped_ckpt = training_results.pop("checkpoint_path", None)
        resolved_ckpt = checkpoint_path or popped_ckpt
        training_state_path = None
        
        arch_type = getattr(getattr(config, "architecture", None), "type", None)
        if best_model_state is not None:
            resolved_ckpt = self.resolve_safe_path("checkpoints", f"model_best_{experiment_id}.pt")
            try:
                torch.save(best_model_state, resolved_ckpt)
            except Exception as e:
                print(f"Error saving best model checkpoint: {e}")
        elif resolved_ckpt and arch_type in ("mlp", "cnn") and os.path.isdir(resolved_ckpt):
            candidate_pt = os.path.join(resolved_ckpt, "model.pt")
            if os.path.isfile(candidate_pt):
                resolved_ckpt = candidate_pt
        elif resolved_ckpt and arch_type == "llm" and os.path.isfile(resolved_ckpt):
            resolved_ckpt = os.path.dirname(resolved_ckpt)
                
        if last_training_state is not None:
            training_state_path = self.resolve_safe_path("checkpoints", f"training_state_{experiment_id}.pt")
            try:
                # Cannot use weights_only=True for full training state (contains optimizer, history dict, etc.)
                torch.save(last_training_state, training_state_path)
            except Exception as e:
                print(f"Error saving training state checkpoint: {e}")
        
        record = {
            "id": experiment_id,
            "experiment_name": display_name,
            "timestamp": timestamp,
            "device_used": device_used,
            "checkpoint_path": resolved_ckpt,
            "training_state_path": training_state_path,
            "schema_version": "0.5.0",
            "config": config.model_dump() if hasattr(config, "model_dump") else config.dict(),
            "preprocessing_artifacts": preprocessing_artifacts,
            "distillation_info": training_results.pop("distillation_info", {}),
            "results": {
                "training": training_results,
                "evaluation": evaluation_results
            }
        }
        
        with open(file_path, "w") as f:
            json.dump(record, f, indent=4)
            
        print(f"Experiment saved to registry: {file_path}")
        return experiment_id

    def resolve_checkpoint_path(self, experiment_id_or_record: Union[str, Dict[str, Any]]) -> str:
        """
        Resolves and validates ``checkpoint_path`` according to the common contract:
        - MLP/CNN: returns path to a ``.pt`` file (transparently resolving legacy directories containing ``model.pt``).
        - LLM: returns path to a directory containing model or adapter files.
        """
        if isinstance(experiment_id_or_record, str):
            record = self.get_experiment_record(experiment_id_or_record)
            exp_id = experiment_id_or_record
        elif isinstance(experiment_id_or_record, dict):
            record = experiment_id_or_record
            exp_id = record.get("id", "")
        else:
            raise ValueError("Expected experiment_id string or experiment record dict.")

        arch_type = (
            record.get("config", {}).get("architecture", {}).get("type")
            or ("mlp" if record.get("schema_version") == "0.2.0" else "mlp")
        )
        raw_path = record.get("checkpoint_path")

        if arch_type == "llm":
            if raw_path:
                if os.path.isdir(raw_path):
                    return raw_path
                if os.path.isfile(raw_path):
                    return os.path.dirname(raw_path)
            if exp_id:
                candidate_dir = os.path.join(self.registry_dir, exp_id)
                if os.path.isdir(candidate_dir):
                    return candidate_dir
            raise FileNotFoundError(f"LLM checkpoint directory not found at expected path: {raw_path}")

        # Classical (MLP / CNN)
        candidates: List[str] = []
        if raw_path:
            if os.path.isdir(raw_path):
                candidates.append(os.path.join(raw_path, "model.pt"))
            else:
                candidates.append(raw_path)
        if exp_id:
            try:
                safe_id = self._validate_id(exp_id)
                candidates.append(os.path.join(self.checkpoints_dir, f"model_best_{safe_id}.pt"))
                candidates.append(os.path.join(self.checkpoints_dir, f"model_{safe_id}.pt"))
                candidates.append(os.path.join(self.registry_dir, safe_id, "model.pt"))
            except ValueError:
                pass

        for cand in candidates:
            if cand and os.path.isfile(cand):
                return cand

        raise FileNotFoundError(f"Best model checkpoint file not found at expected path: {raw_path}")

    def load_classical_state_dict(self, experiment_id_or_record: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Loads the PyTorch state_dict for a classical (MLP/CNN) experiment, supporting both .pt files and legacy dirs."""
        pt_path = self.resolve_checkpoint_path(experiment_id_or_record)
        try:
            return torch.load(pt_path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(pt_path, map_location="cpu")

    def load_classical_model(self, experiment_id_or_record: Union[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], torch.nn.Module]:
        """Instantiates a classical model from its record config and loads its saved state_dict."""
        from aiforge.models import ArchitectureEngine

        record = (
            self.get_experiment_record(experiment_id_or_record)
            if isinstance(experiment_id_or_record, str)
            else experiment_id_or_record
        )
        cfg = ExperimentConfig(**record["config"])
        model = ArchitectureEngine.get_model(cfg)
        state_dict = self.load_classical_state_dict(record)
        model.load_state_dict(state_dict)
        return record, model

    def load_llm_for_inference(
        self,
        experiment_id_or_record: Union[str, Dict[str, Any]],
        device: str = "cpu",
        merge_adapters: bool = False,
    ) -> Tuple[Any, Any]:
        """
        Centralized loader for LLM experiments or ``BASE:<model_id>`` identifiers.
        Supports:
        - Full HuggingFace model directories (``config.json``)
        - PEFT LoRA adapter directories (``adapter_config.json``)
        - Native AIFORGE QLoRA adapter directories (``qlora_adapter.pt``)
        Returns ``(model, tokenizer)``.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if isinstance(experiment_id_or_record, str) and experiment_id_or_record.startswith("BASE:"):
            base_id = experiment_id_or_record[len("BASE:"):]
            tokenizer = AutoTokenizer.from_pretrained(base_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32).to(device)
            return model, tokenizer

        record = (
            self.get_experiment_record(experiment_id_or_record)
            if isinstance(experiment_id_or_record, str)
            else experiment_id_or_record
        )
        ckpt_dir = self.resolve_checkpoint_path(record)
        llm_cfg = record.get("config", {}).get("architecture", {}).get("llm_config") or {}
        base_id = llm_cfg.get("model_id")

        qlora_pt = os.path.join(ckpt_dir, "qlora_adapter.pt")
        adapter_cfg_json = os.path.join(ckpt_dir, "adapter_config.json")
        hf_cfg_json = os.path.join(ckpt_dir, "config.json")

        tok_source = ckpt_dir if os.path.exists(os.path.join(ckpt_dir, "tokenizer_config.json")) else (base_id or ckpt_dir)
        tokenizer = AutoTokenizer.from_pretrained(tok_source)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if os.path.isfile(qlora_pt):
            from aiforge.llm.qlora import apply_qlora_to_model, load_qlora_state_dict, fuse_qlora_model

            q_cfg = {}
            q_cfg_path = os.path.join(ckpt_dir, "qlora_config.json")
            if os.path.isfile(q_cfg_path):
                with open(q_cfg_path, "r") as f:
                    q_cfg = json.load(f)
            target_base = q_cfg.get("model_id") or base_id
            if not target_base:
                raise ValueError(f"Cannot determine base model_id for QLoRA adapter in {ckpt_dir}")
            model = AutoModelForCausalLM.from_pretrained(target_base, torch_dtype=torch.float32)
            apply_qlora_to_model(
                model,
                lora_r=q_cfg.get("lora_r", llm_cfg.get("lora_r", 8)),
                lora_alpha=q_cfg.get("lora_alpha", llm_cfg.get("lora_alpha", 16)),
                lora_dropout=q_cfg.get("lora_dropout", llm_cfg.get("lora_dropout", 0.05)),
                quant_type=q_cfg.get("quant_type", llm_cfg.get("quantization_type", "nf4")),
            )
            state = torch.load(qlora_pt, map_location="cpu", weights_only=True)
            load_qlora_state_dict(model, state)
            if merge_adapters:
                fuse_qlora_model(model)
            return model.to(device), tokenizer

        if os.path.isfile(adapter_cfg_json) and not os.path.isfile(hf_cfg_json):
            from peft import PeftModel

            with open(adapter_cfg_json, "r") as f:
                a_cfg = json.load(f)
            target_base = a_cfg.get("base_model_name_or_path") or base_id
            if not target_base:
                raise ValueError(f"Cannot determine base model_id for PEFT adapter in {ckpt_dir}")
            base_model = AutoModelForCausalLM.from_pretrained(target_base, torch_dtype=torch.float32)
            peft_model = PeftModel.from_pretrained(base_model, ckpt_dir)
            model = peft_model.merge_and_unload() if merge_adapters else peft_model
            return model.to(device), tokenizer

        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.float32).to(device)
        return model, tokenizer

    def load_experiment(self, experiment_id: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Loads an experiment record and its associated BEST model state_dict (for MLP/CNN)
        or resolved checkpoint descriptor (for LLM).
        """
        record = self.get_experiment_record(experiment_id)
        arch_type = record.get("config", {}).get("architecture", {}).get("type", "mlp")
        if arch_type == "llm":
            resolved_dir = self.resolve_checkpoint_path(record)
            record["checkpoint_path"] = resolved_dir
            return record, {"checkpoint_path": resolved_dir}

        resolved_pt = self.resolve_checkpoint_path(record)
        model_state = self.load_classical_state_dict(record)
        record["checkpoint_path"] = resolved_pt
        return record, model_state

    def save_experiment_record(self, experiment_id: str, record: Dict[str, Any]):
        experiment_id = self._validate_id(experiment_id)
        file_path = self.resolve_safe_path(f"exp_{experiment_id}.json")
        with open(file_path, "w") as f:
            json.dump(record, f, indent=4)

    def get_experiment_record(self, experiment_id: str) -> Dict[str, Any]:
        """Loads only the JSON record without the weights."""
        experiment_id = self._validate_id(experiment_id)
        file_path = self.resolve_safe_path(f"exp_{experiment_id}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Experiment record not found for ID: {experiment_id}")
            
        with open(file_path, "r") as f:
            return json.load(f)
        
    def load_training_state(self, experiment_id: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Loads the experiment record and the FULL training state for resuming.
        """
        experiment_id = self._validate_id(experiment_id)
        file_path = self.resolve_safe_path(f"exp_{experiment_id}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Experiment record not found for ID: {experiment_id}")
            
        with open(file_path, "r") as f:
            record = json.load(f)
            
        training_state_path = record.get("training_state_path")
        training_state = None
        
        if training_state_path and os.path.exists(training_state_path):
            # weights_only=False because this dict contains non-tensor objects (optimizer state, dicts)
            # This is safe assuming we trust the files we wrote locally.
            training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        else:
            raise FileNotFoundError(f"Training state checkpoint not found at: {training_state_path}")
            
        return record, training_state

    def list_experiments(self) -> List[Dict[str, Any]]:
        experiments = []
        for filename in os.listdir(self.registry_dir):
            if filename.startswith("exp_") and filename.endswith(".json"):
                with open(os.path.join(self.registry_dir, filename), "r") as f:
                    try:
                        record = json.load(f)
                        experiments.append(record)
                    except json.JSONDecodeError:
                        print(f"Warning: Corrupted JSON record {filename}")
        experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return experiments

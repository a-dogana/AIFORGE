import os
import sys
import shutil
import struct
import subprocess
import tempfile
import json
from typing import Dict, Any, Optional, List, Tuple
from aiforge.experiments.registry import ExperimentRegistry


SUPPORTED_GGUF_QUANTIZATIONS = [
    "F16",       # Full 16-bit float
    "Q8_0",      # 8-bit standard quantization (near zero loss)
    "Q5_K_M",    # 5-bit Medium K-quant (balanced quality/speed)
    "Q4_K_M",    # 4-bit Medium K-quant (recommended for 8GB Mac M1)
    "Q4_0",      # 4-bit legacy standard
    "Q3_K_M",    # 3-bit aggressive compression
]

GGUF_MAGIC = b"GGUF"
SUPPORTED_GGUF_VERSIONS = (2, 3)


def validate_gguf_file(file_path: str) -> Dict[str, Any]:
    """
    Validates that a file on disk is a non-empty binary GGUF file with a valid
    GGUF header (magic 'GGUF', version 2 or 3, and 64-bit tensor/KV counts).
    """
    if not file_path or not os.path.isfile(file_path):
        raise FileNotFoundError(f"GGUF file not found: {file_path}")

    file_size = os.path.getsize(file_path)
    if file_size < 24:
        raise ValueError(
            f"Invalid GGUF file '{file_path}': size ({file_size} bytes) is smaller than the 24-byte header."
        )

    with open(file_path, "rb") as f:
        header = f.read(24)

    magic, version, tensor_count, metadata_kv_count = struct.unpack("<4sIQQ", header)
    if magic != GGUF_MAGIC:
        raise ValueError(
            f"Invalid GGUF magic bytes in '{file_path}': expected {GGUF_MAGIC!r}, got {magic!r}."
        )
    if version not in SUPPORTED_GGUF_VERSIONS:
        raise ValueError(
            f"Unsupported GGUF version {version} in '{file_path}' (supported: {SUPPORTED_GGUF_VERSIONS})."
        )

    return {
        "valid": True,
        "file_size": file_size,
        "version": version,
        "tensor_count": tensor_count,
        "metadata_kv_count": metadata_kv_count,
    }


class GGUFExporter:
    """
    Exports AIFORGE trained/merged LLM checkpoints into GGUF-compatible packages
    ready for execution in llama.cpp, LM Studio, Ollama, and LocalAI.
    """

    def __init__(
        self,
        registry: Optional[ExperimentRegistry] = None,
        llama_cpp_dir: Optional[str] = None,
        convert_script_path: Optional[str] = None,
        llama_quantize_bin: Optional[str] = None,
        python_executable: Optional[str] = None,
    ):
        self.registry = registry or ExperimentRegistry()
        self.llama_cpp_dir = llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR")
        self.convert_script_path = convert_script_path or os.environ.get("AIFORGE_CONVERT_HF_TO_GGUF")
        self.llama_quantize_bin = llama_quantize_bin or os.environ.get("AIFORGE_LLAMA_QUANTIZE")
        self.python_executable = python_executable or sys.executable

    def resolve_tools(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Locates `convert_hf_to_gguf.py` and `llama-quantize` via explicit config,
        `LLAMA_CPP_DIR`, or system PATH.
        """
        convert_script = self.convert_script_path
        if convert_script and not os.path.isfile(convert_script):
            convert_script = None

        if not convert_script and self.llama_cpp_dir:
            candidate = os.path.join(self.llama_cpp_dir, "convert_hf_to_gguf.py")
            if os.path.isfile(candidate):
                convert_script = candidate

        if not convert_script:
            convert_script = (
                shutil.which("convert_hf_to_gguf.py")
                or shutil.which("convert-hf-to-gguf")
            )

        quantize_bin = self.llama_quantize_bin
        if quantize_bin and not os.path.isfile(quantize_bin):
            quantize_bin = shutil.which(quantize_bin)

        if not quantize_bin and self.llama_cpp_dir:
            exe_names = ["llama-quantize.exe", "llama-quantize"] if os.name == "nt" else ["llama-quantize"]
            for exe in exe_names:
                for sub in (("build", "bin", exe), ("bin", exe), (exe,)):
                    cand = os.path.join(self.llama_cpp_dir, *sub)
                    if os.path.isfile(cand):
                        quantize_bin = cand
                        break
                if quantize_bin:
                    break

        if not quantize_bin:
            quantize_bin = shutil.which("llama-quantize")

        return convert_script, quantize_bin

    @staticmethod
    def _is_adapter_only_checkpoint(checkpoint_path: str) -> Tuple[bool, str]:
        """
        Returns (is_adapter_only, adapter_type) where adapter_type is 'peft', 'qlora', or 'full'.
        """
        has_full_cfg = os.path.isfile(os.path.join(checkpoint_path, "config.json"))
        has_peft_cfg = os.path.isfile(os.path.join(checkpoint_path, "adapter_config.json"))
        has_qlora = os.path.isfile(os.path.join(checkpoint_path, "qlora_adapter.pt"))
        if has_qlora and not has_full_cfg:
            return True, "qlora"
        if has_peft_cfg and not has_full_cfg:
            return True, "peft"
        return False, "full"

    def _materialize_merged_hf_checkpoint(
        self, checkpoint_path: str, rec: Dict[str, Any], target_dir: str
    ) -> str:
        """
        Fuses a PEFT LoRA or AIFORGE QLoRA adapter into its base HuggingFace model
        and saves a standalone HuggingFace checkpoint in `target_dir` for GGUF conversion.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _, adapter_type = self._is_adapter_only_checkpoint(checkpoint_path)
        cfg = rec.get("config", {})
        llm_cfg = cfg.get("architecture", {}).get("llm_config", {})
        allow_download = bool(llm_cfg.get("allow_download", False))
        trust_remote = bool(llm_cfg.get("trust_remote_code", False))

        if adapter_type == "qlora":
            from aiforge.llm.qlora import (
                apply_qlora_to_model,
                load_qlora_state_dict,
                fuse_qlora_model,
            )

            qlora_cfg_path = os.path.join(checkpoint_path, "qlora_config.json")
            qlora_cfg = {}
            if os.path.isfile(qlora_cfg_path):
                with open(qlora_cfg_path, "r", encoding="utf-8") as f:
                    qlora_cfg = json.load(f)
            base_id = qlora_cfg.get("model_id") or llm_cfg.get("model_id")
            if not base_id:
                raise ValueError("Cannot fuse QLoRA adapter: base model_id is missing.")

            base_model = AutoModelForCausalLM.from_pretrained(
                base_id,
                local_files_only=not allow_download,
                trust_remote_code=trust_remote,
            )
            apply_qlora_to_model(
                base_model,
                lora_r=qlora_cfg.get("lora_r", llm_cfg.get("lora_r", 8)),
                lora_alpha=qlora_cfg.get("lora_alpha", llm_cfg.get("lora_alpha", 16.0)),
                lora_dropout=0.0,
                quant_type=qlora_cfg.get("quant_type", "nf4"),
            )
            adapter_state = torch.load(
                os.path.join(checkpoint_path, "qlora_adapter.pt"),
                map_location="cpu",
                weights_only=True,
            )
            load_qlora_state_dict(base_model, adapter_state)
            merged_model = fuse_qlora_model(base_model)
            merged_model.save_pretrained(target_dir)

            tok_source = (
                checkpoint_path
                if os.path.isfile(os.path.join(checkpoint_path, "tokenizer_config.json"))
                else base_id
            )
            tokenizer = AutoTokenizer.from_pretrained(
                tok_source,
                local_files_only=not allow_download,
                trust_remote_code=trust_remote,
            )
            tokenizer.save_pretrained(target_dir)
            return target_dir

        if adapter_type == "peft":
            from peft import PeftModel

            peft_cfg_path = os.path.join(checkpoint_path, "adapter_config.json")
            with open(peft_cfg_path, "r", encoding="utf-8") as f:
                peft_meta = json.load(f)
            base_id = peft_meta.get("base_model_name_or_path") or llm_cfg.get("model_id")
            if not base_id:
                raise ValueError("Cannot fuse PEFT adapter: base_model_name_or_path is missing.")

            base_model = AutoModelForCausalLM.from_pretrained(
                base_id,
                local_files_only=not allow_download,
                trust_remote_code=trust_remote,
            )
            peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
            merged_model = peft_model.merge_and_unload()
            merged_model.save_pretrained(target_dir)

            tok_source = (
                checkpoint_path
                if os.path.isfile(os.path.join(checkpoint_path, "tokenizer_config.json"))
                else base_id
            )
            tokenizer = AutoTokenizer.from_pretrained(
                tok_source,
                local_files_only=not allow_download,
                trust_remote_code=trust_remote,
            )
            tokenizer.save_pretrained(target_dir)
            return target_dir

        return checkpoint_path

    def _run_conversion_pipeline(
        self,
        source_checkpoint: str,
        rec: Dict[str, Any],
        target_gguf_path: str,
        quantization: str,
        convert_script: str,
        quantize_bin: Optional[str],
    ) -> Dict[str, Any]:
        is_adapter, _ = self._is_adapter_only_checkpoint(source_checkpoint)
        with tempfile.TemporaryDirectory(prefix="aiforge_gguf_") as tmp_dir:
            hf_dir = (
                self._materialize_merged_hf_checkpoint(source_checkpoint, rec, os.path.join(tmp_dir, "merged_hf"))
                if is_adapter
                else source_checkpoint
            )

            if quantization == "F16":
                cmd_convert = [
                    self.python_executable,
                    convert_script,
                    hf_dir,
                    "--outtype",
                    "f16",
                    "--outfile",
                    target_gguf_path,
                ]
                proc = subprocess.run(cmd_convert, capture_output=True, text=True, check=False)
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"convert_hf_to_gguf.py failed (exit {proc.returncode}): {proc.stderr or proc.stdout}"
                    )
            else:
                if not quantize_bin:
                    raise RuntimeError(
                        f"llama-quantize binary is required for quantization '{quantization}' but was not found."
                    )
                temp_f16_path = os.path.join(tmp_dir, "model_f16.gguf")
                cmd_convert = [
                    self.python_executable,
                    convert_script,
                    hf_dir,
                    "--outtype",
                    "f16",
                    "--outfile",
                    temp_f16_path,
                ]
                proc_conv = subprocess.run(cmd_convert, capture_output=True, text=True, check=False)
                if proc_conv.returncode != 0:
                    raise RuntimeError(
                        f"convert_hf_to_gguf.py failed (exit {proc_conv.returncode}): {proc_conv.stderr or proc_conv.stdout}"
                    )
                validate_gguf_file(temp_f16_path)

                cmd_quant = [
                    quantize_bin,
                    temp_f16_path,
                    target_gguf_path,
                    quantization,
                ]
                proc_quant = subprocess.run(cmd_quant, capture_output=True, text=True, check=False)
                if proc_quant.returncode != 0:
                    raise RuntimeError(
                        f"llama-quantize failed (exit {proc_quant.returncode}): {proc_quant.stderr or proc_quant.stdout}"
                    )

        return validate_gguf_file(target_gguf_path)

    def export(
        self,
        experiment_id: str,
        quantization: str = "Q4_K_M",
        output_dir: Optional[str] = None,
        custom_name: Optional[str] = None,
        run_conversion: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Prepares a GGUF export package for a registered LLM experiment and executes
        the binary GGUF conversion pipeline when llama.cpp tools are available
        (or raises explicitly when `run_conversion=True` and tools are missing).
        """
        if quantization.upper() not in SUPPORTED_GGUF_QUANTIZATIONS:
            raise ValueError(
                f"Unsupported quantization '{quantization}'. "
                f"Supported: {', '.join(SUPPORTED_GGUF_QUANTIZATIONS)}"
            )

        quantization = quantization.upper()
        rec = self.registry.get_experiment_record(experiment_id)
        if not rec:
            raise ValueError(f"Experiment {experiment_id} not found in registry.")

        cfg = rec.get("config", {})
        arch_type = cfg.get("architecture", {}).get("type", "")
        if arch_type != "llm":
            raise ValueError(f"Only LLM models can be exported to GGUF (found '{arch_type}').")

        if isinstance(self.registry, ExperimentRegistry):
            checkpoint_path = self.registry.resolve_checkpoint_path(rec)
        else:
            checkpoint_path = rec.get("checkpoint_path", "")
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint directory '{checkpoint_path}' does not exist.")

        export_target = output_dir or os.path.join(checkpoint_path, "gguf_export")
        os.makedirs(export_target, exist_ok=True)

        clean_id = experiment_id.lower().replace(" ", "_")
        model_name = custom_name or f"aiforge-{clean_id}-{quantization.lower()}"
        gguf_filename = f"{model_name}.gguf"
        target_gguf_path = os.path.join(export_target, gguf_filename)

        # Build GGUF metadata descriptor
        llm_cfg = cfg.get("architecture", {}).get("llm_config", {})
        base_model_id = llm_cfg.get("model_id", "custom-llm")
        max_seq_len = llm_cfg.get("max_seq_length", 2048)

        gguf_meta = {
            "format": "GGUF",
            "version": 3,
            "architecture": "llama" if "qwen" in base_model_id.lower() or "llama" in base_model_id.lower() else "gpt2",
            "base_model": base_model_id,
            "quantization": quantization,
            "context_length": max_seq_len,
            "experiment_id": experiment_id,
            "source_checkpoint": os.path.abspath(checkpoint_path),
            "output_file": target_gguf_path,
            "chat_template": cfg.get("backend_args", {}).get("chat_template", "chatml")
        }

        meta_path = os.path.join(export_target, "gguf_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(gguf_meta, f, indent=2)

        # Generate standalone Python conversion helper script
        conversion_script = f"""#!/usr/bin/env python3
# Auto-generated by AIFORGE v2.1.1 GGUF Exporter
import sys
import subprocess
import os

print("=== AIFORGE GGUF Conversion Tool ===")
print("Model ID: {experiment_id}")
print("Target Quantization: {quantization}")
print("Output GGUF: {target_gguf_path}")
"""
        script_path = os.path.join(export_target, "convert_to_gguf.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(conversion_script)

        # Generate Modelfile for Ollama GGUF direct import
        modelfile_content = f"""# GGUF Modelfile generated by AIFORGE
FROM {os.path.abspath(target_gguf_path)}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx {max_seq_len}
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>\"\"\"
"""
        modelfile_path = os.path.join(export_target, "Modelfile")
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(modelfile_content)

        convert_script, quantize_bin = self.resolve_tools()
        tools_ready = bool(convert_script) and (quantization == "F16" or bool(quantize_bin))

        if run_conversion is True and not tools_ready:
            missing = []
            if not convert_script:
                missing.append("convert_hf_to_gguf.py")
            if quantization != "F16" and not quantize_bin:
                missing.append("llama-quantize")
            raise RuntimeError(
                f"Cannot execute GGUF conversion: missing required llama.cpp tool(s): {', '.join(missing)}."
            )

        conversion_completed = False
        gguf_validation = None
        should_convert = run_conversion if run_conversion is not None else tools_ready
        if should_convert:
            gguf_validation = self._run_conversion_pipeline(
                source_checkpoint=checkpoint_path,
                rec=rec,
                target_gguf_path=target_gguf_path,
                quantization=quantization,
                convert_script=convert_script,
                quantize_bin=quantize_bin,
            )
            conversion_completed = True

        return {
            "status": "success",
            "conversion_completed": conversion_completed,
            "gguf_validation": gguf_validation,
            "experiment_id": experiment_id,
            "quantization": quantization,
            "export_dir": export_target,
            "target_gguf_path": target_gguf_path,
            "meta_path": meta_path,
            "script_path": script_path,
            "modelfile_path": modelfile_path,
            "convert_hf_to_gguf_available": convert_script is not None,
            "llama_quantize_available": quantize_bin is not None,
            "gguf_meta": gguf_meta,
        }

import time
import os
import math
import torch
from typing import Dict, Any, List, Optional
from aiforge.experiments.registry import ExperimentRegistry
from aiforge.models import ArchitectureEngine
from aiforge.data.dataset import DataEngine

class MergeEvaluator:
    def __init__(self, registry: ExperimentRegistry):
        self.registry = registry

    @staticmethod
    def compute_llm_perplexity(model, tokenizer, texts: List[str], max_length: int = 128, device: str = "cpu") -> float:
        """
        Computes average perplexity of an LLM on a sample slice of text.
        Perplexity = exp(average cross-entropy loss).
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in texts:
                if not text.strip():
                    continue
                encodings = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
                input_ids = encodings.input_ids.to(device)
                if input_ids.shape[1] < 2:
                    continue
                
                # In causal LM, labels are equal to input_ids
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                num_tokens = input_ids.shape[1]
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        if total_tokens == 0:
            return 999.0
            
        avg_loss = total_loss / total_tokens
        # Cap loss to prevent overflow in math.exp
        avg_loss = min(avg_loss, 20.0)
        return float(math.exp(avg_loss))

    def evaluate_merge(self, model_a_id: str, model_b_id: str, merged_model_id: str, sample_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluates Model A, Model B, and Merged Model.
        For LLM: evaluates Perplexity (PPL) on reference evaluation prompts.
        For Classical (MLP/CNN): evaluates Accuracy and Loss on validation loader.
        Sets release_gate: PROMOTED or FLAGGED_REGRESSION.
        """
        rec_a = self.registry.get_experiment_record(model_a_id) if not model_a_id.startswith("BASE:") else None
        rec_b = self.registry.get_experiment_record(model_b_id) if not model_b_id.startswith("BASE:") else None
        rec_c = self.registry.get_experiment_record(merged_model_id)

        if not rec_c:
            raise ValueError(f"Merged experiment {merged_model_id} not found in registry.")

        cfg_c = rec_c.get("config", {})
        arch_type = cfg_c.get("architecture", {}).get("type", "unknown")

        results = {
            "model_a_id": model_a_id,
            "model_b_id": model_b_id,
            "merged_model_id": merged_model_id,
            "architecture": arch_type,
            "metrics": {},
            "regression_detected": False,
            "release_gate": "PROMOTED"
        }

        # ------------------- 1. LLM Evaluation via Perplexity -------------------
        if arch_type == "llm":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            test_prompts = sample_texts or [
                "Artificial intelligence and machine learning are transforming software engineering.",
                "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:",
                "In computer systems, memory management and file descriptors are critical resources.",
                "The quick brown fox jumps over the lazy dog."
            ]

            merged_path = rec_c["checkpoint_path"]
            device = "mps" if torch.backends.mps.is_available() else "cpu"

            try:
                # Load merged model
                tokenizer_c = AutoTokenizer.from_pretrained(merged_path)
                model_c = AutoModelForCausalLM.from_pretrained(merged_path, torch_dtype=torch.float32).to(device)
                t0 = time.time()
                ppl_c = self.compute_llm_perplexity(model_c, tokenizer_c, test_prompts, device=device)
                t1 = time.time()

                results["metrics"]["Merged"] = {
                    "perplexity": ppl_c,
                    "eval_time_s": t1 - t0
                }

                # Evaluate baseline model A if present
                ppl_a = None
                if model_a_id.startswith("BASE:"):
                    base_id = model_a_id.replace("BASE:", "")
                    tok_a = AutoTokenizer.from_pretrained(base_id)
                    mod_a = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32).to(device)
                    ppl_a = self.compute_llm_perplexity(mod_a, tok_a, test_prompts, device=device)
                    del mod_a
                elif rec_a and "checkpoint_path" in rec_a:
                    tok_a = AutoTokenizer.from_pretrained(rec_a["checkpoint_path"])
                    mod_a = AutoModelForCausalLM.from_pretrained(rec_a["checkpoint_path"], torch_dtype=torch.float32).to(device)
                    ppl_a = self.compute_llm_perplexity(mod_a, tok_a, test_prompts, device=device)
                    del mod_a

                if ppl_a is not None:
                    results["metrics"]["Model A"] = {"perplexity": ppl_a}

                # Evaluate baseline model B if present and full model
                ppl_b = None
                if rec_b and "checkpoint_path" in rec_b and os.path.exists(os.path.join(rec_b["checkpoint_path"], "config.json")):
                    tok_b = AutoTokenizer.from_pretrained(rec_b["checkpoint_path"])
                    mod_b = AutoModelForCausalLM.from_pretrained(rec_b["checkpoint_path"], torch_dtype=torch.float32).to(device)
                    ppl_b = self.compute_llm_perplexity(mod_b, tok_b, test_prompts, device=device)
                    del mod_b
                    results["metrics"]["Model B"] = {"perplexity": ppl_b}

                del model_c
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                # Regression rule: if merged PPL > 1.35 * baseline PPL (significant quality degradation)
                baseline_ppl = ppl_a or 50.0
                if ppl_c > (baseline_ppl * 1.35):
                    results["regression_detected"] = True
                    results["release_gate"] = "FLAGGED_REGRESSION"
                else:
                    results["regression_detected"] = False
                    results["release_gate"] = "PROMOTED"

            except Exception as e:
                results["error"] = str(e)
                results["release_gate"] = "UNKNOWN"

        # ------------------- 2. Classical Evaluation (MLP / CNN) -------------------
        elif arch_type in ["mlp", "cnn"]:
            from aiforge.core.config import ExperimentConfig
            cfg_obj = ExperimentConfig(**cfg_c)
            _, val_loader, _, _ = DataEngine.get_dataloaders(cfg_obj)

            criterion = torch.nn.CrossEntropyLoss()
            eval_targets = [("Merged", rec_c)]
            if rec_a:
                eval_targets.append(("Model A", rec_a))
            if rec_b:
                eval_targets.append(("Model B", rec_b))

            for name, rec in eval_targets:
                model = ArchitectureEngine.get_model(cfg_obj)
                model.load_state_dict(torch.load(f"{rec['checkpoint_path']}/model.pt", weights_only=True))
                model.eval()

                correct = 0
                total = 0
                loss = 0.0

                t0 = time.time()
                device = next(model.parameters()).device
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss += criterion(outputs, targets).item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
                t1 = time.time()

                results["metrics"][name] = {
                    "accuracy": correct / total if total > 0 else 0.0,
                    "loss": loss / len(val_loader) if len(val_loader) > 0 else 0.0,
                    "inference_time_s": t1 - t0
                }

            # Check accuracy regression
            acc_c = results["metrics"]["Merged"]["accuracy"]
            baseline_accs = [v["accuracy"] for k, v in results["metrics"].items() if k != "Merged"]
            if baseline_accs and acc_c < min(baseline_accs):
                results["regression_detected"] = True
                results["release_gate"] = "FLAGGED_REGRESSION"
            else:
                results["regression_detected"] = False
                results["release_gate"] = "PROMOTED"

        # Persist release gate and evaluation metrics in registry record
        rec_c["evaluation"] = results
        rec_c["release_gate"] = results["release_gate"]
        self.registry.save_experiment_record(merged_model_id, rec_c)

        return results

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
    def compute_llm_perplexity(model, tokenizer, texts: List[str], max_length: int = 128, device: str = "cpu") -> Optional[float]:
        """
        Computes average perplexity of an LLM on a sample slice of text.
        Perplexity = exp(average cross-entropy loss).
        Returns None if no valid tokens were evaluated or if loss is non-finite.
        """
        if not texts:
            return None

        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in texts:
                if not isinstance(text, str) or not text.strip():
                    continue
                encodings = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
                input_ids = encodings.input_ids.to(device)
                if input_ids.shape[1] < 2:
                    continue
                
                # In causal LM, labels are equal to input_ids
                outputs = model(input_ids, labels=input_ids)
                loss_val = float(outputs.loss.item())
                if not math.isfinite(loss_val):
                    return None
                num_tokens = int(input_ids.shape[1])
                
                total_loss += loss_val * num_tokens
                total_tokens += num_tokens

        if total_tokens == 0 or not math.isfinite(total_loss):
            return None
            
        avg_loss = total_loss / total_tokens
        if not math.isfinite(avg_loss):
            return None
        # Cap loss to prevent overflow in math.exp while keeping finite JSON float
        avg_loss = min(avg_loss, 20.0)
        ppl = float(math.exp(avg_loss))
        return ppl if math.isfinite(ppl) else None

    def evaluate_merge(self, model_a_id: str, model_b_id: str, merged_model_id: str, sample_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluates Model A, Model B, and Merged Model.
        For LLM: evaluates Perplexity (PPL) on reference evaluation prompts.
        For Classical (MLP/CNN): evaluates Accuracy and Loss on validation loader.
        Sets release_gate: PROMOTED, FLAGGED_REGRESSION, or UNKNOWN (default when evaluation is invalid or uncomparable).
        """
        rec_a = None
        if not model_a_id.startswith("BASE:"):
            try:
                rec_a = self.registry.get_experiment_record(model_a_id)
            except FileNotFoundError:
                rec_a = None

        rec_b = None
        if not model_b_id.startswith("BASE:"):
            try:
                rec_b = self.registry.get_experiment_record(model_b_id)
            except FileNotFoundError:
                rec_b = None

        rec_c = self.registry.get_experiment_record(merged_model_id)

        if not rec_c:
            raise ValueError(f"Merged experiment {merged_model_id} not found in registry.")

        cfg_c = rec_c.get("config", {})
        arch_type = cfg_c.get("architecture", {}).get("type", "unknown")

        results: Dict[str, Any] = {
            "model_a_id": model_a_id,
            "model_b_id": model_b_id,
            "merged_model_id": merged_model_id,
            "architecture": arch_type,
            "metrics": {},
            "regression_detected": False,
            "release_gate": "UNKNOWN",
        }

        # ------------------- 1. LLM Evaluation via Perplexity -------------------
        if arch_type == "llm":
            default_prompts = [
                "Artificial intelligence and machine learning are transforming software engineering.",
                "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:",
                "In computer systems, memory management and file descriptors are critical resources.",
                "The quick brown fox jumps over the lazy dog."
            ]
            test_prompts = default_prompts if sample_texts is None else sample_texts

            device = "mps" if torch.backends.mps.is_available() else "cpu"

            try:
                # Load merged model via centralized registry loader
                model_c, tokenizer_c = self.registry.load_llm_for_inference(rec_c, device=device)
                t0 = time.time()
                ppl_c = self.compute_llm_perplexity(model_c, tokenizer_c, test_prompts, device=device)
                t1 = time.time()

                results["metrics"]["Merged"] = {
                    "perplexity": ppl_c,
                    "eval_time_s": round(t1 - t0, 6),
                }

                # Evaluate baseline model A if present
                ppl_a = None
                if model_a_id.startswith("BASE:"):
                    try:
                        mod_a, tok_a = self.registry.load_llm_for_inference(model_a_id, device=device)
                        ppl_a = self.compute_llm_perplexity(mod_a, tok_a, test_prompts, device=device)
                        del mod_a
                    except Exception:
                        ppl_a = None
                elif rec_a:
                    try:
                        mod_a, tok_a = self.registry.load_llm_for_inference(rec_a, device=device)
                        ppl_a = self.compute_llm_perplexity(mod_a, tok_a, test_prompts, device=device)
                        del mod_a
                    except Exception:
                        ppl_a = None

                if ppl_a is not None and math.isfinite(ppl_a):
                    results["metrics"]["Model A"] = {"perplexity": ppl_a}
                else:
                    ppl_a = None

                # Evaluate baseline model B if present
                ppl_b = None
                if rec_b:
                    try:
                        mod_b, tok_b = self.registry.load_llm_for_inference(rec_b, device=device)
                        ppl_b = self.compute_llm_perplexity(mod_b, tok_b, test_prompts, device=device)
                        del mod_b
                        if ppl_b is not None and math.isfinite(ppl_b):
                            results["metrics"]["Model B"] = {"perplexity": ppl_b}
                        else:
                            ppl_b = None
                    except Exception:
                        ppl_b = None

                del model_c
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                baseline_ppls = [p for p in (ppl_a, ppl_b) if p is not None and math.isfinite(p)]
                if ppl_c is None or not math.isfinite(ppl_c) or not baseline_ppls:
                    results["regression_detected"] = False
                    results["release_gate"] = "UNKNOWN"
                else:
                    # Regression rule: if merged PPL > 1.35 * baseline PPL (significant quality degradation)
                    baseline_ppl = ppl_a if ppl_a is not None else min(baseline_ppls)
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
            try:
                from aiforge.core.config import ExperimentConfig
                cfg_obj = ExperimentConfig(**cfg_c)
                _, val_loader, _, _ = DataEngine.get_dataloaders(cfg_obj)

                if val_loader is None or len(val_loader) == 0:
                    results["error"] = "Validation loader is empty."
                    results["release_gate"] = "UNKNOWN"
                else:
                    criterion = torch.nn.CrossEntropyLoss()
                    eval_targets = [("Merged", rec_c)]
                    if rec_a:
                        eval_targets.append(("Model A", rec_a))
                    if rec_b:
                        eval_targets.append(("Model B", rec_b))

                    valid_eval = True
                    for name, rec in eval_targets:
                        model = ArchitectureEngine.get_model(cfg_obj)
                        model.load_state_dict(self.registry.load_classical_state_dict(rec))
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
                                batch_loss = float(criterion(outputs, targets).item())
                                if not math.isfinite(batch_loss):
                                    valid_eval = False
                                    break
                                loss += batch_loss
                                _, predicted = torch.max(outputs.data, 1)
                                total += int(targets.size(0))
                                correct += int((predicted == targets).sum().item())
                        t1 = time.time()

                        if not valid_eval or total == 0:
                            valid_eval = False
                            results["metrics"][name] = {
                                "accuracy": None,
                                "loss": None,
                                "inference_time_s": round(t1 - t0, 6),
                            }
                            break

                        acc_val = float(correct / total)
                        loss_val = float(loss / len(val_loader))
                        if not (math.isfinite(acc_val) and math.isfinite(loss_val)):
                            valid_eval = False
                            break

                        results["metrics"][name] = {
                            "accuracy": acc_val,
                            "loss": loss_val,
                            "inference_time_s": round(t1 - t0, 6),
                        }

                    merged_acc = results["metrics"].get("Merged", {}).get("accuracy")
                    baseline_accs = [
                        v["accuracy"]
                        for k, v in results["metrics"].items()
                        if k != "Merged" and v.get("accuracy") is not None and math.isfinite(v["accuracy"])
                    ]
                    if not valid_eval or merged_acc is None or not math.isfinite(merged_acc) or not baseline_accs:
                        results["regression_detected"] = False
                        results["release_gate"] = "UNKNOWN"
                    elif merged_acc < min(baseline_accs):
                        results["regression_detected"] = True
                        results["release_gate"] = "FLAGGED_REGRESSION"
                    else:
                        results["regression_detected"] = False
                        results["release_gate"] = "PROMOTED"
            except Exception as e:
                results["error"] = str(e)
                results["release_gate"] = "UNKNOWN"

        # Persist release gate and evaluation metrics in registry record
        rec_c["evaluation"] = results
        rec_c["release_gate"] = results["release_gate"]
        self.registry.save_experiment_record(merged_model_id, rec_c)

        return results

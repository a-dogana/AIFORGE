import json
import math
import pytest
import torch

from aiforge.core.config import ExperimentConfig
from aiforge.evaluation.evaluator import EvaluationEngine
from aiforge.evaluation.merge_eval import MergeEvaluator
from aiforge.experiments.registry import ExperimentRegistry
from aiforge.models import ArchitectureEngine


class _DummyOutputs:
    def __init__(self, loss_val: float):
        self.loss = torch.tensor(loss_val, dtype=torch.float32)


class _DummyTokenizer:
    def __call__(self, text, truncation=True, max_length=128, return_tensors="pt"):
        tokens = [1, 2, 3, 4] if len(text.strip()) > 1 else [1]

        class _Enc:
            input_ids = torch.tensor([tokens], dtype=torch.long)

        return _Enc()


class _DummyLLM:
    def __init__(self, loss_val: float):
        self.loss_val = loss_val

    def eval(self):
        return self

    def __call__(self, input_ids, labels=None):
        return _DummyOutputs(self.loss_val)


def _register_llm(registry: ExperimentRegistry, exp_id: str, tmp_path):
    ckpt_dir = tmp_path / f"ckpt_{exp_id}"
    ckpt_dir.mkdir(exist_ok=True)
    cfg = {
        "experiment_name": f"LLM_{exp_id}",
        "domain": "llm",
        "task": "generation",
        "architecture": {"type": "llm", "llm_config": {"model_id": "gpt2"}},
    }
    registry.save_experiment_record(
        exp_id,
        {
            "id": exp_id,
            "experiment_name": f"LLM_{exp_id}",
            "config": cfg,
            "checkpoint_path": str(ckpt_dir),
        },
    )


def test_llm_perplexity_empty_or_nan_returns_none_and_unknown_gate(tmp_path, monkeypatch):
    """Empty prompts or NaN/Inf loss return None (not 999.0) and set release_gate to UNKNOWN with valid JSON."""
    registry = ExperimentRegistry(str(tmp_path / "registry"))
    _register_llm(registry, "mod_a", tmp_path)
    _register_llm(registry, "mod_b", tmp_path)
    _register_llm(registry, "merged", tmp_path)

    tok = _DummyTokenizer()
    assert MergeEvaluator.compute_llm_perplexity(_DummyLLM(1.0), tok, []) is None
    assert MergeEvaluator.compute_llm_perplexity(_DummyLLM(1.0), tok, ["   ", "x"]) is None
    assert MergeEvaluator.compute_llm_perplexity(_DummyLLM(float("nan")), tok, ["valid prompt text"]) is None
    assert MergeEvaluator.compute_llm_perplexity(_DummyLLM(float("inf")), tok, ["valid prompt text"]) is None

    monkeypatch.setattr(
        registry,
        "load_llm_for_inference",
        lambda rec_or_id, device="cpu", merge_adapters=False: (_DummyLLM(float("nan")), tok),
    )

    evaluator = MergeEvaluator(registry)
    res_nan = evaluator.evaluate_merge("mod_a", "mod_b", "merged", sample_texts=["valid prompt text"])
    assert res_nan["release_gate"] == "UNKNOWN"
    assert res_nan["regression_detected"] is False
    # Strictly verify no NaN or Inf exists in serialized JSON
    json.dumps(res_nan, allow_nan=False)

    # Empty sample_texts list must also yield UNKNOWN
    monkeypatch.setattr(
        registry,
        "load_llm_for_inference",
        lambda rec_or_id, device="cpu", merge_adapters=False: (_DummyLLM(1.0), tok),
    )
    res_empty = evaluator.evaluate_merge("mod_a", "mod_b", "merged", sample_texts=[])
    assert res_empty["release_gate"] == "UNKNOWN"
    assert res_empty["metrics"]["Merged"]["perplexity"] is None
    json.dumps(res_empty, allow_nan=False)


def test_llm_missing_baseline_sets_unknown_instead_of_arbitrary_50(tmp_path, monkeypatch):
    """When baseline models cannot be evaluated, release_gate is UNKNOWN rather than using fallback 50.0."""
    registry = ExperimentRegistry(str(tmp_path / "registry"))
    _register_llm(registry, "merged_only", tmp_path)
    tok = _DummyTokenizer()

    def _loader(rec_or_id, device="cpu", merge_adapters=False):
        if isinstance(rec_or_id, dict) and rec_or_id.get("id") == "merged_only":
            # Merged has loss=2.5 -> PPL ~12.18 (would have been < 50*1.35 under old bug!)
            return _DummyLLM(2.5), tok
        raise FileNotFoundError("Baseline missing")

    monkeypatch.setattr(registry, "load_llm_for_inference", _loader)
    evaluator = MergeEvaluator(registry)
    res = evaluator.evaluate_merge("missing_a", "missing_b", "merged_only", sample_texts=["hello world test"])
    assert res["release_gate"] == "UNKNOWN"
    assert res["regression_detected"] is False


def test_llm_valid_baseline_35_percent_regression_threshold(tmp_path, monkeypatch):
    """With a real baseline, PPL <= 1.35 * baseline is PROMOTED and PPL > 1.35 * baseline is FLAGGED_REGRESSION."""
    registry = ExperimentRegistry(str(tmp_path / "registry"))
    _register_llm(registry, "base_a", tmp_path)
    _register_llm(registry, "lora_b", tmp_path)
    _register_llm(registry, "merged_good", tmp_path)
    _register_llm(registry, "merged_bad", tmp_path)
    tok = _DummyTokenizer()

    # base_a loss = 1.0 -> PPL = e^1.0 ≈ 2.71828
    # 1.35 * e^1.0 = e^(1.0 + ln(1.35)) ≈ e^1.3001
    losses = {
        "base_a": 1.0,
        "lora_b": 1.1,
        "merged_good": 1.15,  # PPL < 1.35 * PPL_a -> PROMOTED
        "merged_bad": 1.60,   # PPL > 1.35 * PPL_a -> FLAGGED_REGRESSION
    }

    monkeypatch.setattr(
        registry,
        "load_llm_for_inference",
        lambda rec, device="cpu", merge_adapters=False: (_DummyLLM(losses[rec["id"]]), tok),
    )

    evaluator = MergeEvaluator(registry)
    good = evaluator.evaluate_merge("base_a", "lora_b", "merged_good", sample_texts=["valid test sentence"])
    assert good["release_gate"] == "PROMOTED"
    assert good["regression_detected"] is False

    bad = evaluator.evaluate_merge("base_a", "lora_b", "merged_bad", sample_texts=["valid test sentence"])
    assert bad["release_gate"] == "FLAGGED_REGRESSION"
    assert bad["regression_detected"] is True


def test_classical_empty_loader_or_missing_baseline_sets_unknown(tmp_path, monkeypatch):
    """Classical evaluation sets UNKNOWN when validation loader is empty or baselines are absent."""
    registry = ExperimentRegistry(str(tmp_path / "registry"))
    cfg = ExperimentConfig(
        experiment_name="MLP_Gate",
        domain="tabular",
        task="classification",
        device="cpu",
        data={"type": "digits_builtin"},
        architecture={
            "type": "mlp",
            "mlp_config": {"input_size": 64, "hidden_sizes": [8], "output_size": 10},
        },
    )
    model = ArchitectureEngine.get_model(cfg)
    exp_m = registry.save_experiment(cfg, {"best_model_state": model.state_dict()}, {}, device_used="cpu")

    evaluator = MergeEvaluator(registry)
    # Missing baselines -> UNKNOWN
    res_no_base = evaluator.evaluate_merge("nonexistent_a", "nonexistent_b", exp_m)
    assert res_no_base["release_gate"] == "UNKNOWN"

    # Empty validation loader -> UNKNOWN
    from aiforge.data.dataset import DataEngine

    monkeypatch.setattr(DataEngine, "get_dataloaders", staticmethod(lambda c, saved_artifacts=None: ([], [], [], {})))
    res_empty = evaluator.evaluate_merge(exp_m, exp_m, exp_m)
    assert res_empty["release_gate"] == "UNKNOWN"

    # EvaluationEngine on empty test_loader -> UNKNOWN
    ev_engine = EvaluationEngine(cfg)
    ev_res = ev_engine.evaluate(model, [])
    assert ev_res["release_gate"] == "UNKNOWN"
    assert ev_res["accuracy"] is None

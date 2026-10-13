import pytest
import torch
import math
from unittest.mock import MagicMock
from aiforge.experiments.registry import ExperimentRegistry
from aiforge.evaluation.merge_eval import MergeEvaluator

def test_compute_llm_perplexity_math():
    # Create mock model and tokenizer
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # When called with text, return tensor with 10 tokens
    mock_tokenizer.return_value.input_ids = torch.ones((1, 10), dtype=torch.long)
    # Output loss = 2.0
    mock_output = MagicMock()
    mock_output.loss = torch.tensor(2.0)
    mock_model.return_value = mock_output

    ppl = MergeEvaluator.compute_llm_perplexity(mock_model, mock_tokenizer, ["Hello world", "Test sentence"])
    # Perplexity should equal exp(2.0)
    expected_ppl = math.exp(2.0)
    assert abs(ppl - expected_ppl) < 1e-4

def test_classical_merge_eval_release_gate(tmp_path):
    registry = ExperimentRegistry(str(tmp_path))
    # Fake MLP experiments
    config = {
        "experiment_name": "MLP_Merged",
        "domain": "tabular",
        "task": "classification",
        "data": {"type": "digits_builtin"},
        "architecture": {
            "type": "mlp",
            "mlp_config": {"input_size": 64, "hidden_sizes": [16], "output_size": 10}
        }
    }
    from aiforge.models import ArchitectureEngine
    from aiforge.core.config import ExperimentConfig
    cfg_obj = ExperimentConfig(**config)
    m = ArchitectureEngine.get_model(cfg_obj)

    import os
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    torch.save(m.state_dict(), ckpt_dir / "model.pt")

    registry.save_experiment_record("m_merged", {"id": "m_merged", "config": config, "checkpoint_path": str(ckpt_dir)})

    evaluator = MergeEvaluator(registry)
    res = evaluator.evaluate_merge("m_merged", "m_merged", "m_merged")
    assert "metrics" in res
    assert "Merged" in res["metrics"]
    assert "release_gate" in res
    assert res["release_gate"] in ["PROMOTED", "FLAGGED_REGRESSION"]

    # Verify updated in registry
    saved = registry.get_experiment_record("m_merged")
    assert saved.get("release_gate") == res["release_gate"]

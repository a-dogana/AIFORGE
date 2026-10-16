"""Regression tests for Item 4: Confinement of registry writes, safe IDs, and strategy validation."""

import os
import pytest
import torch

from aiforge.core.config import ExperimentConfig
from aiforge.core.merge_lab import MergeLab, MergeRequest
from aiforge.core.unified_training import UnifiedOrchestrator
from aiforge.experiments.registry import ExperimentRegistry


def _make_mlp_config(name: str = "Safe Experiment") -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name=name,
        domain="tabular",
        task="classification",
        data={"type": "digits_builtin"},
        architecture={"type": "mlp", "mlp_config": {"input_size": 64, "hidden_sizes": [8], "output_size": 10}},
        epochs=1,
        batch_size=16,
    )


def test_safe_id_separated_from_display_name(tmp_path):
    reg_dir = tmp_path / "registry"
    registry = ExperimentRegistry(str(reg_dir))

    display_name = "My Complex Display Name! (v2.1) #test"
    exp_id = registry.generate_experiment_id(display_name)
    assert exp_id.isalnum()
    assert " " not in exp_id
    assert "!" not in exp_id


@pytest.mark.parametrize(
    "bad_name",
    [
        "../escape",
        "../../escape_root",
        "/tmp/absolute_escape",
        "sub/dir/escape",
        "..\\windows_escape",
        "C:\\windows\\temp",
    ],
)
def test_traversal_and_absolute_paths_produce_no_artifacts(tmp_path, bad_name):
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    reg_dir = tmp_path / "registry"
    registry = ExperimentRegistry(str(reg_dir))

    # 1. Direct registry save with bad ID
    cfg = _make_mlp_config("Valid Name")
    with pytest.raises(ValueError):
        registry.save_experiment(cfg, {"model_state": {"w": torch.tensor([1.0])}}, {}, "cpu", experiment_id=bad_name)

    # 2. UnifiedOrchestrator with bad display name or bad experiment_id
    bad_cfg = _make_mlp_config(bad_name)
    with pytest.raises(ValueError):
        UnifiedOrchestrator(bad_cfg, registry)

    orch = UnifiedOrchestrator(cfg, registry)
    with pytest.raises(ValueError):
        orch.run(experiment_id=bad_name)

    # 3. MergeLab with bad output_name
    lab = MergeLab(registry)
    req = MergeRequest(strategy="linear_weight", model_a_id="A", model_b_id="B", output_name=bad_name)
    with pytest.raises(ValueError):
        lab.execute_merge(req)

    # Confirm nothing was created outside or inside registry (except checkpoints dir)
    assert list(outside_dir.iterdir()) == []
    assert registry.list_experiments() == []
    reg_entries = {p.name for p in reg_dir.iterdir()}
    assert reg_entries == {"checkpoints"}
    assert list((reg_dir / "checkpoints").iterdir()) == []


def test_symlink_pointing_outside_registry_rejected_without_writes(tmp_path):
    outside_dir = tmp_path / "outside_target"
    outside_dir.mkdir()
    reg_dir = tmp_path / "registry"
    registry = ExperimentRegistry(str(reg_dir))

    # Create a symlink inside registry pointing to outside_dir
    symlink_dir = reg_dir / "linked_exp"
    os.symlink(outside_dir, symlink_dir)

    cfg = _make_mlp_config("Symlink Test")
    orch = UnifiedOrchestrator(cfg, registry)
    with pytest.raises(ValueError, match="Symlink"):
        orch.run(experiment_id="linked_exp")

    # Also test symlinked JSON record
    symlink_json = reg_dir / "exp_linkedjson.json"
    outside_json = outside_dir / "stolen.json"
    os.symlink(outside_json, symlink_json)
    with pytest.raises(ValueError, match="Symlink"):
        registry.save_experiment_record("linkedjson", {"id": "linkedjson"})

    assert list(outside_dir.iterdir()) == []


def test_invalid_merge_strategy_creates_no_artifacts_or_experiment(tmp_path):
    reg_dir = tmp_path / "registry"
    registry = ExperimentRegistry(str(reg_dir))
    lab = MergeLab(registry)

    cfg = _make_mlp_config("MLP_Source").model_dump()
    registry.save_experiment_record("A", {"id": "A", "config": cfg, "checkpoint_path": str(reg_dir / "A")})
    registry.save_experiment_record("B", {"id": "B", "config": cfg, "checkpoint_path": str(reg_dir / "B")})

    before_files = set(os.listdir(reg_dir))

    req = MergeRequest(
        strategy="nonexistent_quantum_merge",
        model_a_id="A",
        model_b_id="B",
        alpha=0.5,
        output_name="ShouldNotExist",
    )
    with pytest.raises(ValueError, match="strategy"):
        lab.execute_merge(req)

    after_files = set(os.listdir(reg_dir))
    assert after_files == before_files
    assert len(registry.list_experiments()) == 2

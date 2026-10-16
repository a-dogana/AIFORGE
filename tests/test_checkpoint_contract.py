import json
import os
import pytest
import torch

from aiforge.core.config import ExperimentConfig
from aiforge.core.merge_lab import MergeLab, MergeRequest
from aiforge.core.unified_training import UnifiedOrchestrator, LLMAdapter
from aiforge.data.dataset import DataEngine
from aiforge.deployment.deployer import DeploymentEngine
from aiforge.deployment.inference import InferenceEngine
from aiforge.evaluation.merge_eval import MergeEvaluator
from aiforge.experiments.registry import ExperimentRegistry
from aiforge.models import ArchitectureEngine
from aiforge.training.trainer import TrainingEngine


def _mlp_config(name: str = "ContractMLP") -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name=name,
        domain="tabular",
        task="classification",
        seed=42,
        epochs=1,
        batch_size=32,
        learning_rate=0.01,
        device="cpu",
        data={"type": "digits_builtin"},
        architecture={
            "type": "mlp",
            "mlp_config": {"input_size": 64, "hidden_sizes": [16], "output_size": 10},
        },
    )


def test_classical_full_pipeline_train_registry_merge_eval_export_inference(tmp_path):
    """End-to-end contract test: training -> registry -> merge -> evaluation -> export -> inference."""
    reg_dir = tmp_path / "registry"
    registry = ExperimentRegistry(str(reg_dir))

    # 1. Train Model A via standard TrainingEngine + registry.save_experiment
    cfg_a = _mlp_config("MLP_Standard_A")
    train_loader, val_loader, _, artifacts = DataEngine.get_dataloaders(cfg_a)
    model_a = ArchitectureEngine.get_model(cfg_a)
    trainer = TrainingEngine(cfg_a)
    train_res_a = trainer.train(model_a, train_loader, val_loader)
    exp_a = registry.save_experiment(
        cfg_a,
        train_res_a,
        {"accuracy": 0.85, "loss": 0.4},
        device_used="cpu",
        preprocessing_artifacts=artifacts,
    )
    rec_a = registry.get_experiment_record(exp_a)
    assert rec_a["checkpoint_path"].endswith(".pt")
    assert os.path.isfile(rec_a["checkpoint_path"])

    # 2. Train Model B via UnifiedOrchestrator
    cfg_b = _mlp_config("MLP_Unified_B")
    orch = UnifiedOrchestrator(cfg_b, registry)
    exp_b = orch.run()
    rec_b = registry.get_experiment_record(exp_b)
    assert rec_b["checkpoint_path"].endswith(".pt")
    assert os.path.isfile(rec_b["checkpoint_path"])

    # 3. Merge A + B in MergeLab (previously failed when A had a .pt file path and B had a dir path)
    lab = MergeLab(registry)
    merged_id = lab.execute_merge(
        MergeRequest(
            strategy="linear_weight",
            model_a_id=exp_a,
            model_b_id=exp_b,
            alpha=0.5,
            output_name="Merged_Contract_MLP",
        )
    )
    rec_m = registry.get_experiment_record(merged_id)
    assert rec_m["checkpoint_path"].endswith(".pt")
    assert os.path.isfile(rec_m["checkpoint_path"])

    # 4. Evaluate merged model vs A and B in MergeEvaluator
    evaluator = MergeEvaluator(registry)
    eval_res = evaluator.evaluate_merge(exp_a, exp_b, merged_id)
    assert "Merged" in eval_res["metrics"]
    assert "Model A" in eval_res["metrics"]
    assert "Model B" in eval_res["metrics"]
    assert 0.0 <= eval_res["metrics"]["Merged"]["accuracy"] <= 1.0

    # 5. Export merged model with DeploymentEngine and run InferenceEngine
    export_dir = tmp_path / "deployed_merged"
    deployer = DeploymentEngine(str(reg_dir))
    manifest = deployer.export(merged_id, str(export_dir), formats=["pytorch"])
    assert "pytorch" in manifest["formats"]

    inf = InferenceEngine(str(export_dir), backend="pytorch")
    preds = inf.predict(torch.randn(4, 64).numpy())
    assert preds.shape == (4, 10)


def test_legacy_directory_checkpoint_read_compatibility(tmp_path):
    """Legacy classical records whose checkpoint_path points to a directory with model.pt remain readable."""
    registry = ExperimentRegistry(str(tmp_path / "registry"))
    cfg = _mlp_config("LegacyDirMLP")
    model = ArchitectureEngine.get_model(cfg)

    legacy_dir = tmp_path / "legacy_ckpt"
    legacy_dir.mkdir()
    torch.save(model.state_dict(), legacy_dir / "model.pt")

    registry.save_experiment_record(
        "legacy01",
        {
            "id": "legacy01",
            "experiment_name": "LegacyDirMLP",
            "config": cfg.model_dump(),
            "checkpoint_path": str(legacy_dir),
            "results": {},
        },
    )

    resolved = registry.resolve_checkpoint_path("legacy01")
    assert resolved == str(legacy_dir / "model.pt")

    rec, state = registry.load_experiment("legacy01")
    assert state is not None
    _, loaded_model = registry.load_classical_model("legacy01")
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2)


def test_llm_checkpoint_contract_and_soup_pop(tmp_path):
    """save_experiment extracts checkpoint_path from training_results (Soup format) and stores directory path at root."""
    registry = ExperimentRegistry(str(tmp_path / "registry"))
    adapter_dir = tmp_path / "soup_adapter_out"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "gpt2"}))

    cfg = ExperimentConfig(
        experiment_name="Soup_LLM_Run",
        domain="llm",
        task="generation",
        data={"type": "jsonl", "csv_config": {"path": "dummy.jsonl", "target_column": "output"}},
        architecture={
            "type": "llm",
            "llm_config": {"model_id": "gpt2", "lora_enabled": True},
        },
    )
    soup_res = {
        "history": {"train_loss": [1.5, 1.1]},
        "checkpoint_path": str(adapter_dir),
        "soup_version": "0.1.0",
    }
    exp_id = registry.save_experiment(cfg, soup_res, {}, device_used="cpu")
    rec = registry.get_experiment_record(exp_id)
    assert rec["checkpoint_path"] == str(adapter_dir)
    assert "checkpoint_path" not in rec["results"]["training"]
    assert registry.resolve_checkpoint_path(exp_id) == str(adapter_dir)


def test_llm_adapter_supports_jsonl_and_saves_adapters(tmp_path, monkeypatch):
    """LLMAdapter accepts jsonl data type and delegates adapter saving to engine.save_adapters."""
    cfg = ExperimentConfig(
        experiment_name="LLMAdapterJSONL",
        domain="llm",
        task="generation",
        data={"type": "jsonl", "csv_config": {"path": str(tmp_path / "data.jsonl"), "target_column": "output"}},
        architecture={
            "type": "llm",
            "llm_config": {"model_id": "gpt2", "lora_enabled": True, "qlora": True},
        },
    )
    adapter = LLMAdapter(cfg)

    saved_dirs = []

    class DummyEngine:
        def load(self, local_path=None):
            pass

        def train(self, dataset_path):
            return {"history": {"train_loss": [1.0]}}

        def save_adapters(self, save_dir):
            saved_dirs.append(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "qlora_config.json"), "w") as f:
                json.dump({"model_id": "gpt2", "qlora": True}, f)

    adapter.engine = DummyEngine()
    res = adapter.train()
    assert res["history"]["train_loss"] == [1.0]

    target_dir = str(tmp_path / "saved_adapter")
    out_path = adapter.save(target_dir)
    assert out_path == target_dir
    assert saved_dirs == [target_dir]
    assert os.path.isfile(os.path.join(target_dir, "qlora_config.json"))

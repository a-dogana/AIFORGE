import datetime
import json
import os
import pytest
import torch
from fastapi.testclient import TestClient

from aiforge.core.config import ExperimentConfig
from aiforge.core.merge_lab import MergeLab, MergeRequest
from aiforge.core.unified_training import UnifiedOrchestrator
from aiforge.experiments.registry import ExperimentRegistry
from aiforge.models import ArchitectureEngine


def test_list_experiments_mixed_and_invalid_timestamps(tmp_path, capsys):
    """list_experiments handles ISO UTC, legacy naive ISO, float epoch, and invalid/missing timestamps without TypeError."""
    reg_dir = tmp_path / "registry"
    registry = ExperimentRegistry(str(reg_dir))

    # Define 4 valid timestamps with known chronological order:
    # t1 (oldest): float epoch = 1700000000.0
    # t2: ISO UTC = 1700001000.0
    # t3: legacy naive ISO in local system timezone = 1700002000.0
    # t4 (newest): ISO UTC with Z = 1700003000.0
    t1_epoch = 1700000000.0
    t2_iso_utc = datetime.datetime.fromtimestamp(1700001000.0, tz=datetime.timezone.utc).isoformat()
    t3_naive_local = datetime.datetime.fromtimestamp(1700002000.0).replace(tzinfo=None).isoformat()
    t4_iso_z = (
        datetime.datetime.fromtimestamp(1700003000.0, tz=datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    records = [
        {"id": "rec_float", "experiment_name": "FloatEpoch", "timestamp": t1_epoch},
        {"id": "rec_iso_utc", "experiment_name": "IsoUtc", "timestamp": t2_iso_utc},
        {"id": "rec_naive", "experiment_name": "NaiveIso", "timestamp": t3_naive_local},
        {"id": "rec_iso_z", "experiment_name": "IsoZ", "timestamp": t4_iso_z},
        {"id": "rec_bad", "experiment_name": "Malformed", "timestamp": "not-a-valid-date"},
        {"id": "rec_missing", "experiment_name": "MissingTs"},
    ]

    raw_Snapshots = {}
    for r in records:
        p = reg_dir / f"exp_{r['id']}.json"
        content = json.dumps(r, indent=2)
        p.write_text(content, encoding="utf-8")
        raw_Snapshots[r["id"]] = content

    listed = registry.list_experiments()
    ids = [r["id"] for r in listed]

    # Valid records ordered newest -> oldest, invalid/missing placed deterministically at the bottom
    assert ids[:4] == ["rec_iso_z", "rec_naive", "rec_iso_utc", "rec_float"]
    assert ids[4:] == ["rec_bad", "rec_missing"]

    out = capsys.readouterr().out
    assert "Warning:" in out
    assert "rec_bad" in out
    assert "rec_missing" in out

    # Ensure JSON files on disk were NOT rewritten during read
    for rec_id, before_text in raw_Snapshots.items():
        after_text = (reg_dir / f"exp_{rec_id}.json").read_text(encoding="utf-8")
        assert after_text == before_text


def test_new_writers_emit_iso8601_utc_and_ui_endpoint_does_not_fail(tmp_path, monkeypatch):
    """save_experiment, UnifiedOrchestrator, and MergeLab emit timezone-aware ISO 8601 UTC strings."""
    reg_dir = tmp_path / "registry"
    registry = ExperimentRegistry(str(reg_dir))

    cfg = ExperimentConfig(
        experiment_name="TsCheckMLP",
        domain="tabular",
        task="classification",
        epochs=1,
        batch_size=32,
        device="cpu",
        data={"type": "digits_builtin"},
        architecture={
            "type": "mlp",
            "mlp_config": {"input_size": 64, "hidden_sizes": [8], "output_size": 10},
        },
    )

    model = ArchitectureEngine.get_model(cfg)
    id_save = registry.save_experiment(
        cfg,
        {"best_model_state": model.state_dict()},
        {"accuracy": 0.5},
        device_used="cpu",
    )
    id_orch = UnifiedOrchestrator(cfg, registry).run()
    id_merge = MergeLab(registry).execute_merge(
        MergeRequest(
            strategy="linear_weight",
            model_a_id=id_save,
            model_b_id=id_orch,
            alpha=0.5,
            output_name="MergedTsMLP",
        )
    )

    for exp_id in (id_save, id_orch, id_merge):
        rec = registry.get_experiment_record(exp_id)
        ts = rec["timestamp"]
        assert isinstance(ts, str)
        parsed = datetime.datetime.fromisoformat(ts)
        assert parsed.tzinfo == datetime.timezone.utc

    # Inject a legacy float timestamp record and verify /api/experiments succeeds without TypeError
    registry.save_experiment_record(
        "legacy_epoch",
        {"id": "legacy_epoch", "experiment_name": "OldEpoch", "timestamp": 1600000000.0},
    )

    import aiforge.ui.app as ui_app

    monkeypatch.setattr(ui_app, "registry", registry)
    monkeypatch.setenv("AIFORGE_TESTING", "1")
    client = TestClient(ui_app.app)
    resp = client.get("/api/experiments")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 4
    assert data[-1]["id"] == "legacy_epoch"

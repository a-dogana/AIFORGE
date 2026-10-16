import pytest

import aiforge.core.resources as res_mod
from aiforge.core.resources import DeviceUnavailableError, ResourceChecker, select_device


def _patch(monkeypatch, cuda: bool, mps: bool):
    monkeypatch.setattr(res_mod, "_cuda_available", lambda: cuda)
    monkeypatch.setattr(res_mod, "_mps_available", lambda: mps)


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [
        (True, True, "cuda"),
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_auto_priority_cuda_mps_cpu(monkeypatch, cuda, mps, expected):
    _patch(monkeypatch, cuda, mps)
    assert select_device("auto") == expected
    assert ResourceChecker.get_available_device("auto") == expected


def test_explicit_devices_validated(monkeypatch):
    _patch(monkeypatch, cuda=False, mps=False)
    assert select_device("cpu") == "cpu"
    with pytest.raises(DeviceUnavailableError):
        select_device("mps")
    with pytest.raises(DeviceUnavailableError):
        select_device("cuda")
    with pytest.raises(ValueError):
        select_device("tpu")

    _patch(monkeypatch, cuda=True, mps=True)
    assert select_device("mps") == "mps"
    assert select_device("cuda") == "cuda"
    assert select_device("cuda:0") == "cuda:0"


def test_llm_engine_uses_shared_selector(monkeypatch):
    from aiforge.core.config import ExperimentConfig
    from aiforge.llm.engine import LLMEngine

    _patch(monkeypatch, cuda=False, mps=False)
    cfg = ExperimentConfig(
        experiment_name="dev_sel",
        domain="llm",
        task="generation",
        device="auto",
        data={"type": "jsonl", "csv_config": {"path": "dummy.jsonl", "target_column": "output"}},
        architecture={"type": "llm", "llm_config": {"model_id": "gpt2"}},
    )
    assert LLMEngine(cfg).device == "cpu"

    cfg_mps = cfg.model_copy(update={"device": "mps"})
    with pytest.raises(DeviceUnavailableError):
        LLMEngine(cfg_mps)


def test_ui_base_serve_config_has_no_hardcoded_mps(monkeypatch, tmp_path):
    import yaml
    import aiforge.ui.app as ui_app

    monkeypatch.setattr(ui_app, "TMP_CONFIG_DIR", str(tmp_path))
    cmd = ui_app._build_serve_command("BASE:gpt2")
    cfg_path = cmd[cmd.index("--config") + 1]
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["device"] == "auto"

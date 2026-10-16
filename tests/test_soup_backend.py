import os
import pytest
from unittest.mock import patch, MagicMock
from aiforge.core.config import ExperimentConfig
from aiforge.llm.backends.soup import SoupBackend
from aiforge.core.run_forge import resolve_soup_env

def test_resolve_soup_env(monkeypatch, tmp_path):
    # Case 1: SOUP_ENV_PATH set and exists
    test_env = tmp_path / "custom_soup"
    test_env.mkdir()
    monkeypatch.setenv("SOUP_ENV_PATH", str(test_env))
    assert resolve_soup_env() == str(test_env)

    # Case 2: SOUP_ENV_PATH not set, fallback
    monkeypatch.delenv("SOUP_ENV_PATH", raising=False)
    resolved = resolve_soup_env()
    assert resolved is not None
    assert "venv_soup" in resolved

def test_soup_backend_init_missing_env():
    cfg = ExperimentConfig(
        experiment_name="test_soup",
        domain="llm",
        task="generation",
        architecture={"type": "llm", "llm_config": {"model_id": "test_id"}},
        data={"type": "jsonl", "csv_config": {"path": "dummy.jsonl", "target_column": "text"}},
        backend="soup",
        backend_args={}
    )
    with pytest.raises(ValueError, match="soup_env_path is required"):
        SoupBackend(cfg)

def test_soup_backend_translate_config(tmp_path):
    env_dir = tmp_path / "mock_soup_env"
    env_dir.mkdir()
    (env_dir / "bin").mkdir()
    (env_dir / "bin" / "python").touch()
    (env_dir / "bin" / "soup").touch()

    cfg = ExperimentConfig(
        experiment_name="test_soup_run",
        domain="llm",
        task="generation",
        architecture={
            "type": "llm",
            "llm_config": {
                "model_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                "lora_enabled": True,
                "lora_r": 8,
                "lora_alpha": 16,
                "max_seq_length": 512
            }
        },
        data={"type": "jsonl", "csv_config": {"path": "datasets/test.jsonl", "target_column": "output"}},
        batch_size=2,
        epochs=3,
        learning_rate=0.0002,
        backend="soup",
        backend_args={
            "soup_env_path": str(env_dir),
            "chat_template": "chatml",
            "quantization": "4bit"
        }
    )

    backend = SoupBackend(cfg)
    soup_cfg = backend._translate_config()

    assert soup_cfg["base"] == "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    assert soup_cfg["task"] == "sft"
    assert soup_cfg["data"]["chat_template"] == "chatml"
    assert soup_cfg["data"]["max_length"] == 512
    assert soup_cfg["training"]["batch_size"] == 2
    assert soup_cfg["training"]["epochs"] == 3
    assert soup_cfg["training"]["lr"] == 0.0002
    assert soup_cfg["training"]["lora"]["r"] == 8
    assert soup_cfg["training"]["lora"]["alpha"] == 16
    assert soup_cfg["training"]["quantization"] == "4bit"

def test_soup_backend_train_mocked(tmp_path):
    env_dir = tmp_path / "mock_soup_env"
    env_dir.mkdir()
    (env_dir / "bin").mkdir()
    (env_dir / "bin" / "python").touch()
    (env_dir / "bin" / "soup").touch()

    cfg = ExperimentConfig(
        experiment_name="test_soup_train",
        domain="llm",
        task="generation",
        architecture={
            "type": "llm",
            "llm_config": {
                "model_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                "lora_enabled": True
            }
        },
        data={"type": "jsonl", "csv_config": {"path": "datasets/test.jsonl", "target_column": "output"}},
        backend="soup",
        backend_args={
            "soup_env_path": str(env_dir)
        }
    )

    backend = SoupBackend(cfg)
    
    def fake_run(cmd, *args, **kwargs):
        if "train" in cmd:
            # Simulate Soup writing an adapter to the configured output dir
            import yaml
            cfg_path = cmd[cmd.index("--config") + 1]
            with open(cfg_path) as f:
                out_dir = yaml.safe_load(f)["output"]
            os.makedirs(out_dir, exist_ok=True)
            open(os.path.join(out_dir, "adapter_model.safetensors"), "w").close()
            return MagicMock(stdout="Epoch 1/3 - loss: 0.25\n", returncode=0)
        return MagicMock(stdout="Soup 1.0.0\n", returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        res = backend.train("datasets/sample.jsonl")
        assert "checkpoint_path" in res
        assert res["soup_version"] == "Soup 1.0.0"
        assert res["backend"] == "soup"


def test_soup_backend_windows_and_posix_paths(tmp_path, monkeypatch):
    """Verify SoupBackend resolves Scripts/python.exe and Scripts/soup.exe on Windows (os.name == 'nt') and bin/* on POSIX."""
    from aiforge.llm.backends.soup import resolve_soup_executables

    win_env = tmp_path / "win_soup_env"
    (win_env / "Scripts").mkdir(parents=True)
    (win_env / "Scripts" / "python.exe").touch()
    (win_env / "Scripts" / "soup.exe").touch()

    py_win, cli_win = resolve_soup_executables(str(win_env), is_windows=True)
    assert py_win.endswith(os.path.join("Scripts", "python.exe"))
    assert cli_win.endswith(os.path.join("Scripts", "soup.exe"))

    posix_env = tmp_path / "posix_soup_env"
    (posix_env / "bin").mkdir(parents=True)
    (posix_env / "bin" / "python").touch()
    (posix_env / "bin" / "soup").touch()

    py_pos, cli_pos = resolve_soup_executables(str(posix_env), is_windows=False)
    assert py_pos.endswith(os.path.join("bin", "python"))
    assert cli_pos.endswith(os.path.join("bin", "soup"))

    # Simulate Windows runtime via monkeypatch on os.name
    import aiforge.llm.backends.soup as soup_mod
    monkeypatch.setattr(soup_mod.os, "name", "nt")
    cfg = ExperimentConfig(
        experiment_name="test_soup_win",
        domain="llm",
        task="generation",
        architecture={"type": "llm", "llm_config": {"model_id": "gpt2", "lora_enabled": True}},
        data={"type": "jsonl", "csv_config": {"path": "datasets/test.jsonl", "target_column": "output"}},
        backend="soup",
        backend_args={"soup_env_path": str(win_env)},
    )
    backend = SoupBackend(cfg)
    assert backend.soup_python == py_win
    assert backend.soup_cli == cli_win

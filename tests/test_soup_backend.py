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
    
    with patch("subprocess.Popen") as mock_popen, patch("subprocess.run") as mock_run:
        # Mock version call
        mock_run.return_value = MagicMock(stdout="Soup 1.0.0\n")
        
        # Mock training execution
        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = [
            b"Starting Soup training...\n",
            b"Epoch 1/3 - loss: 0.25\n",
            b""
        ]
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process

        res = backend.train("datasets/sample.jsonl")
        assert "checkpoint_path" in res
        assert "soup_version" in res
        assert res["backend"] == "soup"

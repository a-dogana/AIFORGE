import os
import subprocess
import yaml
import tempfile
import json
from typing import Dict, Any, Optional, Tuple

from aiforge.core.config import ExperimentConfig
from aiforge.experiments.registry import ExperimentRegistry


def resolve_soup_env() -> str:
    """Resolve the Soup virtual environment path dynamically across CLI, Forge, and SoupBackend."""
    env_soup = os.environ.get("SOUP_ENV_PATH")
    if env_soup and os.path.exists(env_soup):
        return os.path.abspath(env_soup)

    local_soup = os.path.abspath("./venv_soup")
    if os.path.exists(local_soup):
        return local_soup

    home_soup = os.path.expanduser("~/.aiforge/venv_soup")
    if os.path.exists(home_soup):
        return home_soup

    return local_soup


def resolve_soup_executables(soup_env_path: str, is_windows: Optional[bool] = None) -> Tuple[str, str]:
    """
    Resolve (python_executable, soup_executable) inside ``soup_env_path``
    in a cross-platform way (Windows ``Scripts/*.exe`` vs POSIX ``bin/*``).
    """
    win = (os.name == "nt") if is_windows is None else bool(is_windows)
    if win:
        scripts_dir = os.path.join(soup_env_path, "Scripts")
        soup_py = os.path.join(scripts_dir, "python.exe")
        soup_cli = os.path.join(scripts_dir, "soup.exe")
    else:
        bin_dir = os.path.join(soup_env_path, "bin")
        soup_py = os.path.join(bin_dir, "python")
        soup_cli = os.path.join(bin_dir, "soup")
    return soup_py, soup_cli


class SoupBackend:
    """Optional backend for delegating LLM training to Soup in an isolated environment."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.soup_env_path = config.backend_args.get("soup_env_path")
        if self.soup_env_path == "auto":
            self.soup_env_path = resolve_soup_env()
        if not self.soup_env_path:
            raise ValueError("soup_env_path is required in backend_args when using the 'soup' backend.")
            
        self.soup_python, self.soup_cli = resolve_soup_executables(self.soup_env_path)
        
        self.check_availability()
        
    def check_availability(self):
        """Verify the Soup environment and executable are available."""
        if not os.path.exists(self.soup_env_path):
            raise FileNotFoundError(f"Soup environment not found at {self.soup_env_path}")
        if not os.path.exists(self.soup_python):
            raise FileNotFoundError(f"Python executable not found in Soup environment: {self.soup_python}")
        if not os.path.exists(self.soup_cli):
            raise FileNotFoundError(f"Soup executable not found in Soup environment: {self.soup_cli}")
            
    def _translate_config(self) -> Dict[str, Any]:
        """Convert AIFORGE ExperimentConfig to a Soup-compatible dictionary."""
        llm = self.config.architecture.llm_config
        if not llm:
            raise ValueError("LLMConfig is missing.")
            
        # Check unsupported arguments
        if self.config.distillation.enabled:
            raise ValueError("Distillation is not natively supported via the Soup adapter yet.")
        if self.config.tuning.enabled:
            raise ValueError("Tuning is not natively supported via the Soup adapter yet.")
            
        # Map parameters to soup.yaml
        data_cfg = {
            "train": self.config.data.csv_config.path if self.config.data.csv_config else None,
            "format": "auto",
            "max_length": llm.max_seq_length
        }
        
        if "chat_template" in self.config.backend_args:
            data_cfg["chat_template"] = self.config.backend_args["chat_template"]

        soup_cfg = {
            "base": llm.model_id,
            "task": "sft",
            "data": data_cfg,
            "training": {
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "lr": self.config.learning_rate,
                "lora": {
                    "r": llm.lora_r,
                    "alpha": llm.lora_alpha,
                    "target_modules": "auto"
                },
                "quantization": self.config.backend_args.get("quantization", "none")
            },
            "output": "./output"
        }
        return soup_cfg
        
    def train(self, dataset_path: str) -> Dict[str, Any]:
        """Runs the Soup training process."""
        soup_cfg = self._translate_config()
        # Override dataset path
        soup_cfg["data"]["train"] = dataset_path
        
        # Temporary dir for the soup config and output
        temp_dir = tempfile.mkdtemp(prefix="aiforge_soup_")
        config_path = os.path.join(temp_dir, "soup.yaml")
        output_dir = os.path.join(temp_dir, "output")
        soup_cfg["output"] = output_dir
        
        with open(config_path, "w") as f:
            yaml.dump(soup_cfg, f)
            
        # Build command: soup train --config <path> -y
        cmd = [self.soup_cli, "train", "--config", config_path, "-y"]
        
        # Get Soup version
        soup_version = "unknown"
        try:
            ver_res = subprocess.run([self.soup_cli, "--version"], capture_output=True, text=True)
            soup_version = ver_res.stdout.strip()
        except Exception:
            pass

        print(f"Starting Soup ({soup_version}) training process: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            training_log = result.stdout
            exit_code = result.returncode
            print(training_log)
        except subprocess.CalledProcessError as e:
            print("Soup training failed:")
            print(e.stdout)
            print(e.stderr)
            raise RuntimeError(f"Soup training process failed with exit code {e.returncode}.\nLog:\n{e.stderr}") from e
            
        if not os.path.exists(output_dir) or not os.listdir(output_dir):
            raise FileNotFoundError(f"Soup did not produce any output in {output_dir}")
            
        return {
            "backend": "soup",
            "history": {"train_loss": [], "val_loss": []},
            "test_results": {},
            "checkpoint_path": output_dir,
            "soup_version": soup_version,
            "exit_code": exit_code,
            "training_log": training_log
        }

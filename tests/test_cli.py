import pytest
import subprocess
import os

import sys
import shutil

CLI_CMD = ["aiforge"] if shutil.which("aiforge") else [sys.executable, "aiforge_cli.py"]

def test_cli_help():
    result = subprocess.run(CLI_CMD + ["--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "AIFORGE CLI" in result.stdout
    assert "run" in result.stdout
    assert "list" in result.stdout
    assert "show" in result.stdout
    assert "evaluate" in result.stdout
    assert "forge" in result.stdout
    assert "lima-generate" in result.stdout
    assert "export-gguf" in result.stdout

def test_cli_list():
    result = subprocess.run(CLI_CMD + ["list"], capture_output=True, text=True)
    assert result.returncode == 0

def test_cli_run_invalid_config():
    result = subprocess.run(CLI_CMD + ["run", "--config", "nonexistent.yaml"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Failed to load configuration" in result.stdout

def test_cli_forge_help():
    result = subprocess.run(CLI_CMD + ["forge", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--name" in result.stdout
    assert "--model" in result.stdout
    assert "--topic" in result.stdout
    assert "--qlora" in result.stdout
    assert "--method" in result.stdout
    assert "--target-samples" in result.stdout

def test_cli_lima_generate_help():
    result = subprocess.run(CLI_CMD + ["lima-generate", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--topic" in result.stdout
    assert "--target-samples" in result.stdout
    assert "--batch-size" in result.stdout
    assert "--turn-type" in result.stdout
    assert "--output" in result.stdout

def test_cli_export_gguf_help():
    result = subprocess.run(CLI_CMD + ["export-gguf", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--quantization" in result.stdout
    assert "Q4_K_M" in result.stdout




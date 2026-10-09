import pytest
import subprocess
import os

def test_cli_help():
    result = subprocess.run(["aiforge", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "AIFORGE CLI" in result.stdout
    assert "run" in result.stdout
    assert "list" in result.stdout
    assert "show" in result.stdout
    assert "evaluate" in result.stdout

def test_cli_list():
    result = subprocess.run(["aiforge", "list"], capture_output=True, text=True)
    assert result.returncode == 0

def test_cli_run_invalid_config():
    result = subprocess.run(["aiforge", "run", "--config", "nonexistent.yaml"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Failed to load configuration" in result.stdout

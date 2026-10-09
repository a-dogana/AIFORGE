# AIFORGE

AIFORGE is an autonomous laboratory for creating, training, specializing, and evaluating neural networks based on user-defined goals. 

It is NOT a single AI model, but a framework designed to formulate hypotheses, train models, measure results, and iteratively improve them.

## Requirements
- Python 3.9+
- Designed to run on Apple Silicon (MPS supported) or standard CPUs.

## Installation

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install AIFORGE and dependencies
pip install -e .
```

## Usage (CLI)
AIFORGE v0.2 provides a Command Line Interface (CLI) for running experiments from YAML configurations, listing them, and reloading models.

### 1. Run an experiment
You can run an experiment by providing a valid YAML configuration.
```bash
aiforge run --config configs/digits_mlp.yaml
```
This will train the model, evaluate it, and save the metrics in the `registry/` folder, alongside the PyTorch weights (`.pt`) in `registry/checkpoints/`.

### 2. List experiments
To see all past experiments and their evaluation accuracy:
```bash
aiforge list
```

### 3. Show experiment details
To dump the JSON record of a specific experiment (including the used configuration and detailed metrics):
```bash
aiforge show <EXPERIMENT_ID>
```

### 4. Evaluate a saved model
To load a saved model's weights and re-evaluate it on the test set:
```bash
aiforge evaluate <EXPERIMENT_ID>
```

## Running Tests
A full suite of unit and integration tests is provided:
```bash
pytest tests/
```

## Current Status (v0.2)
- ✅ Configuration Engine (YAML & Pydantic)
- ✅ Resource Checker
- ✅ Data Engine (Digits)
- ✅ Architecture Engine (MLP)
- ✅ Training & Evaluation Engines
- ✅ Experiment Registry (JSON Metrics & PyTorch Checkpoints)
- ✅ CLI & Reproducibility
- ⏳ Future: Teacher Engine, Advanced Data, CNN/LLM Support

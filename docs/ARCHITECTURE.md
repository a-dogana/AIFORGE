# AIFORGE Architecture

AIFORGE is designed as a modular laboratory for creating, training, and validating neural networks.

## Core Principles
- **Modularity**: Each engine has a specific responsibility and can be substituted.
- **Reproducibility**: Experiments save configurations and outcomes.
- **Hardware Awareness**: Validates resources (RAM, MPS, CPU) before execution.

## Components (v0.1.0)
Currently implemented modules:
1. **Configuration Engine** (`aiforge.core.config`): Uses Pydantic to parse and validate experiment parameters.
2. **Resource Checker** (`aiforge.core.resources`): Detects CPU, Memory, and MPS compatibility.
3. **Data Engine** (`aiforge.data.dataset`): Handles data loading, scaling, and splitting into train/val/test.
4. **Architecture Engine** (`aiforge.models`): Provides base neural network architectures (e.g., MLP).
5. **Training Engine** (`aiforge.training.trainer`): Orchestrates the PyTorch training loop and metric logging.
6. **Evaluation Engine** (`aiforge.evaluation.evaluator`): Tests the model against hold-out sets.
7. **Experiment Registry** (`aiforge.experiments.registry`): Persists experiment records as JSON files.

## Future Motors
To be implemented in subsequent phases:
- Research Engine (For automated literature & architecture search)
- Teacher Engine (For synthetic data and distillation)
- Experiment Engine (For iterative automated hypothesis testing)
- Deployment Engine (For exporting models)
- Full Model Registry (For saving checkpoints and model lineage)

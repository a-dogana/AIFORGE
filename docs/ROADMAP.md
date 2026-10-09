# AIFORGE Roadmap

## Phase B & C - Foundations & End-to-End Pipeline [DONE]
- [x] Configuration Engine (Pydantic)
- [x] Resource Checks (psutil, PyTorch MPS detection)
- [x] Dataset loading (Scikit-learn Digits)
- [x] Simple Model Architecture (PyTorch MLP)
- [x] Training loop execution
- [x] Quantitative evaluation (scikit-learn metrics)
- [x] Experiment Registry (JSON persistence)
- [x] Automated tests (Pytest)

## Phase D - Persistence & Iteration [NEXT]
- [ ] Model weights persistence (saving and loading `.pt` checkpoints)
- [ ] Command Line Interface (CLI) using `argparse` or `typer` to load experiments from YAML.
- [ ] Iterative execution: comparing a candidate configuration with a baseline.

## Phase E - Consolidation & Extensibility
- [ ] Expand dataset engine to support local CSVs and HuggingFace datasets.
- [ ] Add CNN architectures for vision tasks.
- [ ] Implement Logging & Error Recovery for long-training jobs.

## Phase F - Advanced Capabilities
- [ ] Teacher Engine: Connect to an LLM provider for synthetic data generation.
- [ ] Fine-Tuning: Support for loading pre-trained weights and applying LoRA.
- [ ] Advanced Resource Estimation: Pre-calculate FLOPs and VRAM requirements before training starts.

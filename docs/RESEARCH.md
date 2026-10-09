# AIFORGE Research & Decisions

## Context
AIFORGE must support varying hardware limitations. The first target is Apple Silicon with 8GB RAM.

## Hardware Utilization
- **Apple Silicon (MPS)**: PyTorch provides the `mps` backend. Tests confirm `mps` significantly accelerates simple MLPs and convolutions, but may not support all operators. Fallback to `cpu` must be guaranteed.
- **Memory Limits**: With 8GB total RAM, large Language Models (LLMs) > 3-4B parameters might cause memory swapping or OOM errors. Small models (MLPs, small CNNs) fit comfortably.

## Architectural Decisions
- **Pydantic**: Chosen for robust type checking and automatic validation of experiment configurations.
- **Scikit-learn for Base Datasets**: Used `load_digits` as a safe, built-in dataset for the first increment to avoid downloading external data in unpredictable environments.
- **PyTorch**: Selected as the foundational tensor and autograd library due to its MPS support and widespread research adoption.

## Future Research Areas
- **Small Language Models (SLMs)**: Qwen 1.5B, Llama-3 8B (quantized) for fine-tuning.
- **Quantization**: Exploring QLoRA for 8GB environments to allow SFT on 7B models.
- **Knowledge Distillation**: Utilizing external teacher APIs to train smaller, domain-specific models locally.

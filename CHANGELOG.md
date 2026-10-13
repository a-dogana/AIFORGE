# Changelog

All notable changes to the AIFORGE platform are documented in this file.

## [Unreleased] - Consolidation & Security Hardening (v2.0-dev)

### Security & Robustness
- **Path Traversal Defense**: Added `_validate_id` security guard in `aiforge/experiments/registry.py` to prevent directory traversal (`../`, `/`, `\`) on experiment lookups, checkpoint saves, and model loads.
- **macOS File Descriptor Guard**: Implemented programmatic `RLIMIT_NOFILE` soft limit elevation (up to 10,240) in `aiforge/core/resources.py` (`ResourceChecker.check_and_raise_nofile_limit()`) to prevent `OSError: [Errno 24] Too many open files` on macOS.
- **Zombie & Orphan Process Reaping**:
  - Hardened `/api/server/stop` with a two-stage process termination (`SIGTERM` followed by a 3.0s timeout and fallback to `SIGKILL`).
  - Wrapped `/api/forge/start` Server-Sent Events (SSE) stream in a `try...finally` block to terminate background training subprocesses immediately if a client disconnects.
- **Loopback Proxy Isolation**: Configured `httpx.AsyncClient(trust_env=False, verify=False)` to prevent environment `HTTP_PROXY`/`HTTPS_PROXY` variables from intercepting internal loopback calls to `127.0.0.1:8001`.
- **SSL Environment Sanitization**: Added startup check to remove non-existent `SSL_CERT_FILE` paths from the process environment.

### Added
- **Model Merge Lab (`aiforge/core/merge_lab.py`)**:
  - **SLERP (Spherical Linear Interpolation)**: Vector/tensor-level interpolation on the unit hypersphere with cosine clamping and collinear fallback to LERP.
  - **DARE-Linear (Drop And REscale)**: Weight delta pruning using a Bernoulli probability mask with conservative $1/(1-p)$ scaling to reduce interference.
  - **Multi-LoRA Adapter Fusion**: Multi-adapter composition using PEFT `add_weighted_adapter` with automatic weight normalization.
  - **Classical Merge Strategies**: Support for linear weighted averaging, ties merging, and max magnitude pooling.
- **Perplexity & Release Gate Evaluation (`aiforge/evaluation/merge_eval.py`)**:
  - LLM Perplexity evaluation (`compute_llm_perplexity`) on validation slices with negative log-likelihood cross-entropy loss tracking.
  - Automated release gating tagging merged checkpoints with `PROMOTED` or `FLAGGED_REGRESSION` based on threshold degradation.
- **Ollama One-Click Modelfile Export (`aiforge/deployment/ollama_export.py`)**:
  - Generator for Ollama `Modelfile` with system prompts, temperature parameters, and weights mapping.
  - Integrated CLI command: `aiforge export-ollama <experiment_id>`.
- **Advanced Web Studio UI**:
  - Dedicated Model Merge Lab view with real-time strategy selector, interactive $\alpha$ slider (0.00–1.00), and dynamic balance badge.
  - Post-merge evaluation toolbar with instant perplexity scoring and one-click Ollama export.
- **Unified Training Adapters (`aiforge/core/unified_training.py`)**:
  - Standardized interface decoupling tabular, vision, and causal language modeling workloads.

### CLI Enhancements
- Updated `aiforge merge` to support `--alpha` parameter and 5 merging strategies (`slerp`, `dare_linear`, `linear`, `multi_lora`, `ties`).
- Added `aiforge export-ollama` for direct CLI-driven Ollama packaging.
- Added `aiforge ui` command for launching the local dashboard.

### Verification & Testing
- Automated test suite covering 65 test cases: **62 passed, 3 skipped, 0 failed** in 14.5s.
- Tested on Apple Silicon M1 (8GB unified memory) maintaining peak RAM usage under 1.5GB.

---

## [1.0.0] - 2026-10-09
### Added
- Complete end-to-end Machine Learning pipeline (Data, Architecture, Training, Tuning, Distillation, Export, Serving).
- **Vision (CNN)**: Image dataset support with multi-layer CNN capabilities.
- **AutoTuning**: Built-in hyperparameter tuning via Optuna.
- **Knowledge Distillation**: Teacher-Student knowledge transfer.
- **Deployment Engine**: JIT/ONNX weights-only exportation and decoupling of models from the `DataEngine`.
- **API Server**: Fully functional FastAPI endpoints (`aiforge serve`) for tabular and computer vision models.
- **LLM Specialization**: Autoregressive causal language modeling via `transformers` with `peft` LoRA.
- **LLM Server**: A dedicated HTTP generation server (`aiforge serve-llm`) with sequence limit safety and explicit download authorization.

### Security
- The framework prevents stealth downloads unless `allow_download: true`.
- Deserialization ignores untrusted pickles, enforcing `weights_only=True`.
- API servers bounded strictly to `127.0.0.1`.

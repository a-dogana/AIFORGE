# Changelog

All notable changes to the AIFORGE platform are documented in this file.

## [2.1.0] - 2026-10-14

### Highlights
Introduces Direct GGUF Export for llama.cpp & LM Studio, Multi-Turn Conversational Dialogue Synthesis in LIMA, and Direct Preference Optimization (DPO) preference alignment engine.

### Added
- **GGUF Direct Exporter (`aiforge/deployment/gguf_export.py`)**:
  - Complete packaging engine supporting `F16`, `Q8_0`, `Q5_K_M`, `Q4_K_M`, `Q4_0`, and `Q3_K_M` quantizations.
  - Automatic generation of conversion recipes (`convert_to_gguf.py`), quantization parameters, and GGUF Modelfiles.
  - CLI command: `aiforge export-gguf <experiment_id> [--quantization Q4_K_M]`.
  - Web UI endpoint `/api/export/gguf` and one-click toolbar action button.
- **LIMA Multi-Turn Dialogues (`aiforge/data/lima_pipeline.py`)**:
  - Support for synthesizing multi-turn conversational datasets (`--turn-type multi_turn`) with deep follow-ups, debugging, and corner-case resolution.
  - Extended `LIMAQualityFilter` to validate dialogue structure, alternating roles, and semantic user prompt deduplication.
- **DPO (Direct Preference Optimization) Engine (`aiforge/llm/dpo.py`)**:
  - `DPOLoss`: Reference policy log-ratio alignment with implicit reward metrics (`chosen_reward`, `rejected_reward`, `reward_margin`, `accuracy`).
  - Autoregressive causal token log-prob extraction with label masking (`compute_sequence_log_probs`).
  - `DPOBatchValidator`: Strict schema validation and degenerate pair filtering for `{prompt, chosen, rejected}` datasets.

### Verification & Testing
- Test suite expanded to 92 items: **89 passed, 3 skipped, 0 failed** in 25.79s (100% pass rate).

---

## [2.0.0] - 2026-10-14

### Highlights
Major release introducing Native 4-bit NF4 QLoRA, LIMA Synthetic Data Generation Pipeline, Alpine.js Reactive Web Studio, Model Merge Lab (SLERP/DARE), Ollama One-Click Export, and full Apple Silicon M1/Metal optimization.

### Added
- **Native 4-bit NF4 QLoRA (`aiforge/llm/qlora.py`)**:
  - `NF4QuantizedLinear`: PyTorch linear layer wrapping double-quantized NormalFloat4 weights (2x 4-bit nibbles packed per byte) and low-rank adapter (LoRA $\Delta W$).
  - Dynamic type casting (`self.lora_A.to(x.dtype)`) ensuring complete numerical stability and eliminating mixed-precision `c10::Half != float` runtime crashes on Apple Silicon Metal/MPS.
- **LIMA Synthetic Data Pipeline (`aiforge/data/lima_pipeline.py`)**:
  - Curated generation methodology with automatic taxonomy expansion and critique-based quality filtering.
  - `LIMARateLimiter`: Token bucket algorithm and exponential backoff with jitter to eliminate HTTP 429 rate limit errors against LLM providers.
  - `LIMAStagingManager`: Checkpoint-based staging manager with atomic JSONL appending and metadata state tracking.
- **Alpine.js Reactive Web Studio (`aiforge/ui/static/app_alpine.js`)**:
  - Modern zero-build reactive architecture replacing over 940 lines of monolithic DOM-manipulation JavaScript.
  - Centralized reactive stores (`trainingState`, `metricsStore`, `modalState`).
  - Strict lifecycle cleanup of SSE streams (`EventSource`) and intervals preventing memory leaks.
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
- **Unified Training Adapters (`aiforge/core/unified_training.py`)**:
  - Standardized interface decoupling tabular, vision, and causal language modeling workloads.

### CLI Enhancements
- Added `aiforge forge`: Complete end-to-end synthetic dataset generation (LIMA/Quick) + LoRA/QLoRA fine-tuning + registry registration.
- Added `aiforge lima-generate`: Standalone CLI tool to generate and curate LIMA datasets without training.
- Added `aiforge export-ollama`: Direct CLI-driven Ollama packaging.
- Added `aiforge ui`: Start the local Web Dashboard.
- Updated `aiforge merge`: Support for `--alpha` parameter and 5 merging strategies (`slerp`, `dare_linear`, `linear`, `multi_lora`, `ties`).

### Security & Robustness
- **Dynamic Path Decoupling**: Dynamic resolution of `SOUP_ENV_PATH` with automatic fallback to local virtual environments.
- **Path Traversal Defense**: Added `_validate_id` and `_validate_job_id` security guards in `aiforge/experiments/registry.py` and `aiforge/data/lima_pipeline.py` to prevent directory traversal (`../`, `/`, `\`).
- **macOS File Descriptor Guard**: Implemented programmatic `RLIMIT_NOFILE` soft limit elevation (up to 10,240) in `aiforge/core/resources.py` (`ResourceChecker.check_and_raise_nofile_limit()`) to prevent `OSError: [Errno 24] Too many open files` on macOS.
- **Zombie & Orphan Process Reaping**:
  - Hardened `/api/server/stop` with a two-stage process termination (`SIGTERM` followed by a 3.0s timeout and fallback to `SIGKILL`).
  - Wrapped `/api/forge/start` Server-Sent Events (SSE) stream in a `try...finally` block to terminate background training subprocesses immediately if a client disconnects.
- **Loopback Proxy Isolation**: Configured `httpx.AsyncClient(trust_env=False, verify=False)` to prevent environment `HTTP_PROXY`/`HTTPS_PROXY` variables from intercepting internal loopback calls to `127.0.0.1:8001`.

### Verification & Testing
- Automated test suite covering 78 test cases: **75 passed, 3 skipped, 0 failed** (100% pass rate).
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

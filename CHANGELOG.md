# Changelog

All notable changes to the AIFORGE platform are documented in this file.

## [2.1.1] - 2026-10-16

### Highlights
Complete architectural, security, numerical, and cross-platform remediation release resolving all 17 items from GitHub Issue #1.

### Fixed & Improved
- **Packaging & Dependencies (`setup.py`, `MANIFEST.in`, `requirements.txt`)**:
  - Added missing `__init__.py` in `aiforge/ui` and `aiforge/llm/backends`, included static/template assets via `MANIFEST.in` + `include_package_data=True`, and declared `psutil`, `httpx`, `keyring`, `PyYAML`, `accelerate`, and `pillow`.
- **Process Management (`aiforge/ui/app.py`)**:
  - Replaced indiscriminate `lsof` + `kill -9` port killing with safe TCP socket binding checks (`_is_port_in_use`, HTTP 409) and managed subprocess tree termination via `psutil` (`_terminate_process_tree`).
- **Registry Confinement & Safe Experiment IDs (`aiforge/experiments/registry.py`)**:
  - Separated display `experiment_name` from filesystem `experiment_id` (`generate_experiment_id`), added `resolve_safe_path` confinement before any directory creation, and hardened legacy ID validation.
- **Unified `checkpoint_path` Contract & Centralized Loaders (`aiforge/experiments/registry.py`)**:
  - Standardized `checkpoint_path` (`.pt` file for `mlp`/`cnn`, directory for `llm`) with backward-compatible resolution (`resolve_checkpoint_path`, `load_classical_state_dict`, `load_llm_for_inference`) across training, merging, evaluation, and export.
- **Timestamp Normalization (`aiforge/experiments/registry.py`)**:
  - Standardized new experiment timestamps to ISO 8601 UTC (`now_utc_iso`) and implemented deterministic descending epoch sorting (`parse_timestamp_epoch`) compatible with legacy float and naive ISO timestamps.
- **Evaluation Release Gate (`aiforge/evaluation/merge_eval.py`, `aiforge/evaluation/evaluator.py`)**:
  - Removed sentinel metrics (`999.0`, `50.0`), defaulted `release_gate` to `"UNKNOWN"` on non-finite metrics or missing baselines, and enforced promotion only on finite baseline comparisons.
- **Cross-Platform Credentials, Soup Backend & Hardware Selection**:
  - Migrated secret storage in `aiforge/core/secrets.py` to `keyring` with explicit `SecretBackendUnavailableError`.
  - Added POSIX/Windows path resolution (`resolve_soup_executables`) in `aiforge/llm/backends/soup.py`.
  - Centralized hardware device selection (`select_device`: `cuda` -> `mps` -> `cpu`) in `aiforge/core/resources.py`.
- **Conversational SFT & Truncation Filtering (`aiforge/llm/sft_data.py`, `aiforge/llm/engine.py`)**:
  - Added native `messages`/`turns` SFT encoding with `tokenizer.apply_chat_template`, `-100` masking on non-assistant turns, prefix-consistency validation (`ChatTemplateError`), pre-split filtering of truncated prompt-only samples (`filter_supervised_examples`), and chat-templated inference in `LLMEngine.generate()` and `/api/chat`.
- **Data Leakage Prevention (`aiforge/data/dataset.py`, `aiforge/deployment/inference.py`)**:
  - Moved train/val/test splitting before `StandardScaler` fitting and mean imputation in `digits_builtin` and `csv` pipelines, rejecting all-`NaN` training columns and persisting `feature_columns`, `imputation`, and `scaler` in `preprocessing_artifacts`.
- **QLoRA Memory & Autograd Optimization (`aiforge/llm/qlora.py`)**:
  - Implemented chunked CPU NF4/FP4 quantization, removed persistent `_cached_dequant_w`, and introduced `_NF4BaseLinearFunction` to dequantize on the fly during forward/backward without retaining full-precision weight matrices in the autograd graph.
- **Real GGUF Conversion & Binary Validation (`aiforge/deployment/gguf_export.py`)**:
  - Added execution of `convert_hf_to_gguf.py` and `llama-quantize`, automatic temporary fusion of LoRA/QLoRA adapters, direct `F16` output handling, and binary GGUF v2/v3 header validation (`validate_gguf_file`).

### Verification & Testing
- Full regression & end-to-end suite expanded to 153 items: **152 passed, 1 skipped, 0 failed** (100% pass rate).

---

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

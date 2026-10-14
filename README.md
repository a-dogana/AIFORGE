# AIFORGE v2.1 — Autonomous AI Laboratory & Model Forge

AIFORGE è una piattaforma completa, autonoma e verificabile per l'addestramento, il fine-tuning con QLoRA/LoRA, l'allineamento con preferenze (DPO), la fusione avanzata di modelli (SLERP/DARE), la generazione di dataset sintetici curati (LIMA single-turn & multi-turn) e il deployment ottimizzato per **Apple Silicon (M1/M2/M3/M4)** e CUDA.

---

## 🌟 Funzionalità Chiave

- 📦 **GGUF Direct Exporter**: Esportazione diretta di modelli fine-tuned o fusi in pacchetti GGUF compatibili con `llama.cpp` e LM Studio (`Q4_K_M`, `Q5_K_M`, `Q8_0`, `F16`).
- 🎯 **DPO (Direct Preference Optimization)**: Motore nativo di allineamento da preferenze umane (`prompt`, `chosen`, `rejected`) con funzione di perdita log-ratio e calcolo automatico dei margini di ricompensa.
- 🔬 **LIMA Synthetic Data Pipeline (Single & Multi-Turn)**: Generazione strutturata di dataset sintetici con espansione tassonomica, token-bucket rate limiter, backoff adattivo con jitter e supporto per conversazioni multi-turno a più livelli di approfondimento.
- ⚡ **Native 4-bit NF4 QLoRA**: Quantizzazione NormalFloat4 a doppio stadio con low-rank adapter su CPU e Apple Silicon Metal (MPS), abbattendo fino al 75% l'uso di memoria RAM.
- 🧬 **Model Merge Lab**: Algoritmi geometrici avanzati:
  - **SLERP** (Spherical Linear Interpolation) con clamping e fallback lineare;
  - **DARE-Linear** (Drop And REscale delta pruning);
  - **Multi-LoRA Adapter Fusion**;
  - Valutazione scientifica automatica della **Perplexity (PPL)** con Release Gate (`PROMOTED` / `FLAGGED_REGRESSION`).
- 🦙 **Ollama One-Click Export**: Esportazione automatica di `Modelfile` pronti per l'importazione istantanea nel runtime Ollama locale.
- 🎛️ **Dashboard Studio Reattiva (Alpine.js)**: Nuova interfaccia grafica reattiva a zero dipendenze di build (`aiforge ui`), con monitoraggio in tempo reale, gestione esperimenti, chat playground e toolbar per export Ollama/GGUF.
- 🍲 **Soup Isolated Fine-Tuning Backend**: Motore delegato di addestramento LoRA/SFT in ambiente isolato (`SoupBackend`), con traduzione automatica delle configurazioni (`soup.yaml`), supporto chat template e protezione da conflitti di dipendenze.
- 🛡️ **Hardening & Security-by-Design**: Elevazione dinamica preventiva del limite soft dei file descriptor su macOS (10.240 FD), isolamento loopback da proxy HTTP e protezione assoluta contro il path traversal.

---

## 🚀 Installazione Rapida

È possibile installare la libreria e le sue dipendenze via pip nel proprio ambiente Python 3.9+:

```bash
# Clone e installazione locale
git clone https://github.com/alessandrodogana/AIFORGE.git
cd AIFORGE
pip install -e .
```

### Dipendenze Opzionali
- **Serving & UI**: `pip install aiforge[serving]` (FastAPI, Uvicorn, Jinja2)
- **LLM Specialization & PEFT**: `pip install aiforge[llm]` (Transformers, PEFT, Torch)
- **Hyperparameter Tuning**: `pip install aiforge[tuning]` (Optuna)
- **All Features**: `pip install aiforge[all]`

---

## 🖥️ Interfaccia Web Studio

Avvia la dashboard locale sul browser:
```bash
aiforge ui --host 127.0.0.1 --port 8500
```
La dashboard offre:
1. **La Forgia (LoRA / QLoRA • Soup Backend)**: Generazione dati LIMA (single-turn o multi-turno) e fine-tuning delegato al motore isolato **Soup** con terminale di log streaming.
2. **Merge Lab**: Slider interattivo $\alpha$, verifica di preflight con analisi di compatibilità e RAM, e pulsanti di export rapido (Perplexity, Ollama, GGUF).
3. **Playground Chat**: Interfaccia conversazionale reattiva con badge di stato server e accelerazione Apple Silicon (MPS).
4. **Catalogo Modelli Base**: Selezione rapida di foundation model con azioni dirette "Usa in Chat" e "Usa nella Forgia".
5. **Connettori API**: Preset rapidi (Ollama, LM Studio, Gemini, OpenAI) con verifica istantanea della connessione e storage cifrato nel portachiavi.

---

## 💻 Guida all'Uso della CLI

### 1. La Forgia (Generazione Dati + Fine-Tuning + Registrazione)
```bash
# Esecuzione completa con metodo LIMA, dialogo multi-turno e QLoRA 4-bit
aiforge forge \
  --name "SQL_Specialist" \
  --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
  --topic "PostgreSQL Query Optimization and Indexing" \
  --method lima \
  --turn-type multi_turn \
  --target-samples 100 \
  --qlora
```

### 2. Generazione Standalone Dataset Curato LIMA
```bash
# Genera dataset multi-turno senza avviare il training
aiforge lima-generate \
  --topic "Architettura Software Concorrente in C" \
  --turn-type multi_turn \
  --target-samples 50 \
  --output datasets/c_concurrency_multi_turn.jsonl
```

### 3. Model Merge Lab
```bash
# SLERP tra modello A e modello B
aiforge merge \
  --model-a "exp_model_a" \
  --model-b "exp_model_b" \
  --strategy slerp \
  --alpha 0.5 \
  --output "Merged_SLERP_v1"

# DARE Linear delta merge
aiforge merge \
  --model-a "base_model" \
  --model-b "expert_adapter" \
  --strategy dare_linear \
  --alpha 0.7
```

### 4. Esportazione per LM Studio / llama.cpp (GGUF) e Ollama
```bash
# Esporta pacchetto GGUF quantizzato a 4-bit Medium K-quant
aiforge export-gguf <experiment_id> --quantization Q4_K_M

# Esporta Modelfile per Ollama
aiforge export-ollama <experiment_id> --name "my-custom-coder"
```

### 5. Addestramento Classico e Hyperparameter Tuning
```bash
# Training da file di configurazione YAML
aiforge run --config configs/csv_mlp.yaml

# Auto-Tuning con Optuna
aiforge tune --config configs/csv_mlp.yaml --evaluate-best
```

---

## 🍲 Motore di Fine-Tuning Isolato: Soup Backend

AIFORGE adotta un'architettura a **processi isolati** per l'addestramento dei modelli linguistici:
- **Delega & Isolamento**: Il fine-tuning SFT e LoRA viene delegato a **Soup** (`SoupBackend`), eseguito in un ambiente virtuale dedicato (`SOUP_ENV_PATH` o `./venv_soup`). Questo previene conflitti di versioni tra le librerie di training e il server Web/API.
- **Traduzione Trasparente**: AIFORGE compila dinamicamente la configurazione `soup.yaml`, mappando automaticamente iperparametri, quantizzazione 4-bit NF4 (`Q4_K_M` o `none`), chat template e target modules LoRA.
- **Configurazione Percorso**:
  ```bash
  # Imposta l'ambiente Soup (opzionale se presente ./venv_soup)
  export SOUP_ENV_PATH="/Users/alessandrodogana/AIFORGE/venv_soup"
  ```

---

## 🧪 Verifica e Test Suite

La piattaforma include una test suite esaustiva con oltre 90 casi di test certificati:
```bash
pytest tests/ -v
```

---

## 🛡️ Sicurezza e Isolamento

- **Protezione Deserializzazione**: Esecuzione rigorosa con `weights_only=True` su tutti i caricamenti PyTorch per prevenire attacchi via pickle non attendibili.
- **Isolamento di Rete**: Server di inferenza e studio vincolati rigidamente all'interfaccia loopback locale `127.0.0.1`.
- **Prevenzione Path Traversal**: Tutte le operazioni di caricamento, staging e registry sono validate da controlli anti-traversal.
- **Resource Discipline**: Gestione rigorosa dei limiti di memoria su Mac M1 (8GB) con rilascio immediato dei tensori e dei processi orfani.

---

## 📄 Licenza
Rilasciato sotto licenza MIT.

"""AIFORGE Dashboard API (FastAPI)."""

import asyncio
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from typing import List, Optional

import psutil

if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]

try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 10240:
        target = min(10240, hard) if hard > 0 else 10240
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
except Exception:
    pass

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from aiforge.core.merge_lab import MergeLab, MergeRequest
from aiforge.core.secrets import (
    SecretBackendUnavailableError,
    check_plaintext_migration,
    get_secret,
    get_secret_backend_status,
    has_secret,
    save_secret,
)
from aiforge.experiments.registry import ExperimentRegistry

logger = logging.getLogger("aiforge.ui")

# --- Configuration ---------------------------------------------------------

LLM_SERVER_PORT = 8001
LLM_SERVER_URL = f"http://127.0.0.1:{LLM_SERVER_PORT}"
TMP_CONFIG_DIR = os.path.join("logs", "tmp_configs")
LLM_SERVER_LOG = os.path.join("logs", "llm_server.log")
KEY_PLACEHOLDER = "●●●●●●●● (Configured)"
DEFAULT_CONNECTOR_MODEL = "gemini-1.5-flash"
SECRET_PROVIDERS = ("openai", "anthropic", "google")

_UI_PORTS = (8000, 8500)
ALLOWED_HOSTS = [f"{h}:{p}" for p in _UI_PORTS for h in ("127.0.0.1", "localhost")]
ALLOWED_ORIGINS = [f"http://{h}" for h in ALLOWED_HOSTS]

app = FastAPI(title="AIFORGE Dashboard")

registry_path = os.environ.get("AIFORGE_REGISTRY_PATH", "registry")
registry = ExperimentRegistry(registry_path)

# Handle of the LLM server subprocess started from the UI
llm_server_process: Optional[subprocess.Popen] = None


# --- Security --------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def local_only_guard(request: Request, call_next):
    """Protect every state-changing endpoint against DNS rebinding and CSRF.

    - Host must be a loopback UI address (blocks DNS-rebinding attacks).
    - If a browser sends an Origin header on a mutating request, it must be ours.
    The test client host ("testserver") is accepted only when explicitly enabled.
    """
    host = request.headers.get("host", "")
    allowed_hosts = ALLOWED_HOSTS + (["testserver"] if os.environ.get("AIFORGE_TESTING") else [])
    if host not in allowed_hosts:
        return JSONResponse({"detail": "Invalid Host header. Binding strictly local."}, status_code=403)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin is not None and origin not in ALLOWED_ORIGINS:
            return JSONResponse({"detail": "Invalid Origin header. CSRF protection active."}, status_code=403)
    return await call_next(request)


def _safe_name(value: str) -> str:
    return re.sub(r"[^\w\-]", "_", value)[:100]


# --- LLM server lifecycle --------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=2000)
    max_new_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class StartServerRequest(BaseModel):
    experiment_id: str


def _is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if a TCP listener is already accepting connections on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _terminate_process_tree(pid: int, timeout: float = 3.0) -> bool:
    """Gracefully terminate ``pid`` and all its descendants, then SIGKILL stragglers after ``timeout``s."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return False

    try:
        children = parent.children(recursive=True)
    except psutil.Error:
        children = []

    targets = children + [parent]
    for proc in targets:
        try:
            proc.terminate()
        except psutil.Error:
            pass

    _, alive = psutil.wait_procs(targets, timeout=timeout)
    if alive:
        for proc in alive:
            try:
                proc.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(alive, timeout=1.0)
    return True


def _write_tmp_config(name: str, config: dict) -> str:
    os.makedirs(TMP_CONFIG_DIR, exist_ok=True)
    path = os.path.join(TMP_CONFIG_DIR, f"{_safe_name(name)}.yaml")
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


def _build_serve_command(experiment_id: str) -> List[str]:
    adapter_path = None
    if experiment_id.startswith("BASE:"):
        base_model_id = experiment_id[len("BASE:"):]
        config = {
            "experiment_name": f"Base_{base_model_id.replace('/', '_')}",
            "domain": "llm",
            "task": "generation",
            "device": "auto",
            "architecture": {
                "type": "llm",
                "llm_config": {"model_id": base_model_id, "allow_download": True, "lora_enabled": False},
            },
        }
        config_path = _write_tmp_config("base", config)
    else:
        try:
            exp = registry.get_experiment_record(experiment_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(status_code=404, detail="Experiment not found")
        import copy
        config = copy.deepcopy(exp.get("config", {}))
        if config.get("domain") != "llm":
            raise HTTPException(status_code=400, detail="Only LLM experiments can be started via UI")
        try:
            adapter_path = registry.resolve_checkpoint_path(exp)
        except FileNotFoundError:
            adapter_path = ""
            llm_cfg = config.get("architecture", {}).get("llm_config")
            if isinstance(llm_cfg, dict):
                llm_cfg["lora_enabled"] = False
                llm_cfg["qlora"] = False
        config_path = _write_tmp_config(experiment_id, config)

    cmd = [sys.executable, "-m", "aiforge_cli", "serve-llm", "--config", config_path, "--port", str(LLM_SERVER_PORT)]
    if adapter_path and os.path.exists(adapter_path):
        cmd.extend(["--adapter", adapter_path])
    return cmd


@app.post("/api/server/start")
def start_server(req: StartServerRequest):
    global llm_server_process

    if llm_server_process is not None:
        if llm_server_process.poll() is None:
            raise HTTPException(status_code=400, detail="Server is already running")
        llm_server_process = None

    cmd = _build_serve_command(req.experiment_id)
    if _is_port_in_use(LLM_SERVER_PORT):
        raise HTTPException(
            status_code=409,
            detail=f"Port {LLM_SERVER_PORT} is already in use by another process.",
        )

    try:
        os.makedirs(os.path.dirname(LLM_SERVER_LOG), exist_ok=True)
        with open(LLM_SERVER_LOG, "a") as log_file:
            # The child inherits the descriptor; the parent can close its copy.
            llm_server_process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
        return {"status": "starting", "pid": llm_server_process.pid}
    except Exception as e:
        logger.error(f"Failed to launch LLM server: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/server/stop")
def stop_server():
    global llm_server_process
    if llm_server_process is None:
        return {"status": "not running"}

    proc = llm_server_process
    llm_server_process = None
    was_running = proc.poll() is None
    stopped = _terminate_process_tree(proc.pid, timeout=3.0) if was_running else False
    try:
        proc.wait(timeout=1.0)
    except Exception:
        pass
    return {"status": "stopped" if (stopped or was_running) else "not running"}


@app.get("/api/status")
def get_status():
    llm_status = "offline"
    try:
        req = urllib.request.Request(f"{LLM_SERVER_URL}/ready", method="GET")
        with urllib.request.urlopen(req, timeout=1.0) as response:
            if response.status == 200:
                llm_status = "online"
    except Exception:
        pass

    from aiforge.core.resources import ResourceChecker
    env_guard = ResourceChecker.check_and_raise_nofile_limit()

    return {
        "status": "online",
        "llm_server": llm_status,
        "registry_path": registry_path,
        "experiments_count": len(registry.list_experiments()),
        "env_guard": env_guard,
    }


@app.get("/api/experiments")
def list_experiments():
    return registry.list_experiments()


@app.get("/api/experiments/{exp_id}")
def get_experiment(exp_id: str):
    try:
        return registry.get_experiment_record(exp_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="Experiment not found")


@app.post("/api/generate")
async def generate_text(req: GenerateRequest):
    # Proxy to the local LLM server (loopback only, no TLS involved)
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            resp = await client.post(f"{LLM_SERVER_URL}/v1/generate", json=req.model_dump(), timeout=180.0)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()
    except httpx.RequestError as e:
        logger.error(f"Generate proxy error: {type(e).__name__} - {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM Server non raggiungibile sulla porta {LLM_SERVER_PORT} ({type(e).__name__}). "
                   f"Se è in fase di avvio, attendi qualche secondo.",
        )


# --- API keys & connectors -------------------------------------------------

def _safe_has_secret(provider: str) -> bool:
    try:
        return has_secret(provider)
    except SecretBackendUnavailableError:
        return False


@app.get("/api/keys")
def get_keys():
    backend_info = get_secret_backend_status()
    keys = {}
    backend_available = bool(backend_info.get("available", False))
    backend_error = backend_info.get("error")
    for p in SECRET_PROVIDERS:
        if not backend_available:
            keys[p] = False
        else:
            try:
                keys[p] = has_secret(p)
            except SecretBackendUnavailableError as e:
                backend_available = False
                backend_error = str(e)
                keys[p] = False
    keys["legacy_plaintext_warning"] = check_plaintext_migration()
    keys["backend_available"] = backend_available
    keys["backend"] = backend_info.get("backend", "unknown")
    if backend_error:
        keys["backend_error"] = backend_error
    return keys


@app.post("/api/keys")
def save_keys(keys: dict, request: Request):
    # Stricter than the global guard: secrets may only be written from the dashboard itself.
    if request.headers.get("origin", "") not in ALLOWED_ORIGINS:
        raise HTTPException(status_code=403, detail="Invalid Origin header. CSRF protection active.")

    try:
        for provider in SECRET_PROVIDERS:
            val = keys.get(provider)
            if isinstance(val, str):
                val = val.strip()
                if val and val != KEY_PLACEHOLDER:
                    save_secret(provider, val)
    except SecretBackendUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"status": "success"}


class ConnectorConfig(BaseModel):
    id: str
    name: str
    base_url: str
    model: Optional[str] = DEFAULT_CONNECTOR_MODEL
    api_key: Optional[str] = None


def _connectors_path() -> str:
    return os.path.join(registry_path, "connectors.json")


def _load_connectors() -> list:
    path = _connectors_path()
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def _save_connectors(data: list) -> None:
    with open(_connectors_path(), "w") as f:
        json.dump(data, f, indent=2)


def _default_model_for(conn_id: str) -> str:
    return {"google": DEFAULT_CONNECTOR_MODEL, "openai": "gpt-4o-mini"}.get(conn_id, "local-model")


@app.get("/api/connectors")
def get_connectors():
    if not os.path.exists(_connectors_path()):
        return [
            {"id": "google", "name": "Google Gemini (OpenAI Std)", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "model": DEFAULT_CONNECTOR_MODEL, "has_key": _safe_has_secret("google")},
            {"id": "openai", "name": "OpenAI Default", "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "has_key": _safe_has_secret("openai")},
            {"id": "anthropic", "name": "Anthropic (via Proxy)", "base_url": "", "model": "claude-3-5-sonnet", "has_key": _safe_has_secret("anthropic")},
        ]

    data = _load_connectors()
    for c in data:
        c["has_key"] = _safe_has_secret(c["id"])
        if not c.get("model"):
            c["model"] = _default_model_for(c["id"])
    return data


@app.post("/api/connectors")
def save_connector(conn: ConnectorConfig):
    if conn.api_key and conn.api_key != KEY_PLACEHOLDER:
        try:
            save_secret(conn.id, conn.api_key)
        except SecretBackendUnavailableError as e:
            raise HTTPException(status_code=503, detail=str(e))

    data = _load_connectors()
    entry = {"id": conn.id, "name": conn.name, "base_url": conn.base_url, "model": conn.model or DEFAULT_CONNECTOR_MODEL}
    for c in data:
        if c["id"] == conn.id:
            c.update(entry)
            break
    else:
        data.append(entry)
    _save_connectors(data)
    return {"status": "success"}


@app.delete("/api/connectors/{conn_id}")
def delete_connector(conn_id: str):
    if os.path.exists(_connectors_path()):
        _save_connectors([c for c in _load_connectors() if c["id"] != conn_id])
    # The keychain secret is intentionally left intact.
    return {"status": "success"}


class TestConnectorRequest(BaseModel):
    id: Optional[str] = None
    base_url: str
    api_key: Optional[str] = None


@app.post("/api/connectors/test")
async def test_connector(req: TestConnectorRequest):
    """Test connectivity and latency to an OpenAI-compatible endpoint."""
    if not req.base_url.startswith(("http://", "https://")):
        return {"status": "error", "message": "URL non valido: deve iniziare con http:// o https://"}

    key = req.api_key
    if (not key or key == KEY_PLACEHOLDER) and req.id:
        try:
            key = get_secret(req.id)
        except SecretBackendUnavailableError as e:
            return {"status": "error", "message": f"Credential store non disponibile: {e}"}
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    base = req.base_url.rstrip("/")
    start_time = time.time()
    last_err = "Nessuna risposta dal server"

    async with httpx.AsyncClient(trust_env=False, timeout=6.0) as client:
        for u in (f"{base}/models", base):
            try:
                resp = await client.get(u, headers=headers)
                latency_ms = int((time.time() - start_time) * 1000)
                if resp.status_code == 200:
                    return {"status": "ok", "latency_ms": latency_ms, "message": f"Connesso ({latency_ms} ms)"}
                if resp.status_code in (401, 403):
                    return {"status": "auth_error", "latency_ms": latency_ms, "message": f"Server raggiungibile ma chiave API non valida (HTTP {resp.status_code})"}
                if resp.status_code < 500:
                    return {"status": "ok", "latency_ms": latency_ms, "message": f"Endpoint raggiungibile (HTTP {resp.status_code}, {latency_ms} ms)"}
            except Exception as e:
                last_err = str(e)

    return {"status": "error", "message": f"Connessione fallita: {last_err}"}


# --- Forge & model management ---------------------------------------------

class ForgeRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    model: str
    topic: str = Field("", max_length=500)
    qlora: bool = False
    method: str = Field("quick", pattern="^(quick|lima|custom)$")
    target_samples: int = Field(100, ge=1, le=10000)
    turn_type: str = Field("single_turn", pattern="^(single_turn|multi_turn)$")
    dataset: Optional[str] = None


@app.post("/api/forge/start")
async def start_forge(req: ForgeRequest):
    async def event_generator():
        cmd = [
            sys.executable, "-m", "aiforge.core.run_forge",
            "--name", req.name,
            "--model", req.model,
            "--topic", req.topic or "",
            "--method", req.method,
            "--target-samples", str(req.target_samples),
            "--turn-type", req.turn_type,
        ]
        if req.qlora:
            cmd.append("--qlora")
        if req.dataset:
            cmd.extend(["--dataset", req.dataset])
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    yield f"data: {text}\n\n"
            yield "data: [DONE]\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            if process.returncode is None:
                _terminate_process_tree(process.pid, timeout=3.0)
                try:
                    process.kill()
                except Exception:
                    pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


SUPPORTED_BASE_MODELS = [
    {
        "id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "name": "Qwen 2.5 Coder 0.5B Instruct",
        "desc": "Architettura avanzata (2024), perfetta per Mac da 8 GB. Prestazioni di codice eccezionali con ~1 GB di RAM.",
        "params": "0.5B",
        "recommended": True,
    },
    {
        "id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "name": "Qwen 2.5 Coder 1.5B Instruct",
        "desc": "Capacità di ragionamento superiore per codice C/Python complesso. Richiede circa 2.5 GB di RAM.",
        "params": "1.5B",
        "recommended": False,
    },
    {
        "id": "openai-community/gpt2",
        "name": "GPT-2 Base (124M)",
        "desc": "Modello leggero classico, usato come banco di prova iniziale.",
        "params": "124M",
        "recommended": False,
    },
]
_SUPPORTED_MODEL_IDS = {m["id"] for m in SUPPORTED_BASE_MODELS}


@app.get("/api/models")
def get_available_models():
    # Check the HuggingFace cache dynamically
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    results = []
    for m in SUPPORTED_BASE_MODELS:
        folder_name = "models--" + m["id"].replace("/", "--")
        results.append({**m, "downloaded": os.path.exists(os.path.join(hf_cache_dir, folder_name))})
    return results


@app.post("/api/models/download")
def download_model(payload: dict):
    # Sync handler: FastAPI runs it in a threadpool so the event loop is not blocked.
    model_id = payload.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="Missing model_id")
    if model_id not in _SUPPORTED_MODEL_IDS:
        raise HTTPException(status_code=400, detail="Model not in the supported list")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        AutoTokenizer.from_pretrained(model_id)
        AutoModelForCausalLM.from_pretrained(model_id)
        return {"status": "success", "message": f"Modello {model_id} scaricato con successo in cache!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore download: {e}")


# --- Static UI -------------------------------------------------------------

_UI_DIR = os.path.dirname(__file__)
static_dir = os.path.join(_UI_DIR, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(_UI_DIR, "templates", "index.html"))


# --- Merge, export & evaluation -------------------------------------------

@app.post("/api/merge/preflight")
def merge_preflight_post(req: MergeRequest):
    return MergeLab(registry).validate_pre_merge(req)


@app.get("/api/merge/preflight")
def merge_preflight(model_a: str, model_b: str, strategy: str):
    req = MergeRequest(strategy=strategy, model_a_id=model_a, model_b_id=model_b)
    return MergeLab(registry).validate_pre_merge(req)


@app.post("/api/merge/execute")
def merge_execute(req: MergeRequest):
    try:
        exp_id = MergeLab(registry).execute_merge(req)
        return {"status": "success", "experiment_id": exp_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class OllamaExportRequest(BaseModel):
    experiment_id: str
    ollama_model_name: Optional[str] = None
    system_prompt: Optional[str] = None


@app.post("/api/export/ollama")
def export_to_ollama(req: OllamaExportRequest):
    from aiforge.deployment.ollama_export import OllamaExporter
    try:
        return OllamaExporter(registry).export(
            experiment_id=req.experiment_id,
            ollama_model_name=req.ollama_model_name,
            system_prompt=req.system_prompt,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class GGUFExportRequest(BaseModel):
    experiment_id: str
    quantization: str = "Q4_K_M"
    custom_name: Optional[str] = None


@app.post("/api/export/gguf")
def export_to_gguf(req: GGUFExportRequest):
    from aiforge.deployment.gguf_export import GGUFExporter
    try:
        return GGUFExporter(registry).export(
            experiment_id=req.experiment_id,
            quantization=req.quantization,
            custom_name=req.custom_name,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class MergeEvaluateRequest(BaseModel):
    model_a_id: str
    model_b_id: str
    merged_model_id: str
    sample_texts: Optional[List[str]] = None


@app.post("/api/merge/evaluate")
def evaluate_merge(req: MergeEvaluateRequest):
    from aiforge.evaluation.merge_eval import MergeEvaluator
    try:
        return MergeEvaluator(registry).evaluate_merge(
            model_a_id=req.model_a_id,
            model_b_id=req.model_b_id,
            merged_model_id=req.merged_model_id,
            sample_texts=req.sample_texts,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

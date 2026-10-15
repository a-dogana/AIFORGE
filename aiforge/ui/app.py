import os
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
from aiforge.core.merge_lab import MergeLab, MergeRequest
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import subprocess
import sys
import logging
import yaml
import json
import time

logger = logging.getLogger("aiforge.ui")

from aiforge.experiments.registry import ExperimentRegistry

app = FastAPI(title="AIFORGE Dashboard")

# Setup registry
registry_path = os.environ.get("AIFORGE_REGISTRY_PATH", "registry")
registry = ExperimentRegistry(registry_path)

# Global subprocess reference
llm_server_process = None

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)

class StartServerRequest(BaseModel):
    experiment_id: str

@app.post("/api/server/start")
def start_server(req: StartServerRequest):
    global llm_server_process
    
    if llm_server_process is not None and llm_server_process.poll() is None:
        raise HTTPException(status_code=400, detail="Server is already running")
        
    if req.experiment_id.startswith("BASE:"):
        base_model_id = req.experiment_id.split("BASE:")[1]
        config = {
            "experiment_name": f"Base_{base_model_id.replace('/', '_')}",
            "domain": "llm",
            "task": "generation",
            "device": "mps",
            "architecture": {
                "type": "llm",
                "llm_config": {
                    "model_id": base_model_id,
                    "allow_download": True,
                    "lora_enabled": False
                }
            }
        }
        temp_config_path = f".tmp_config_base.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)
        cmd = [sys.executable, "-m", "aiforge_cli", "serve-llm", "--config", temp_config_path, "--port", "8001"]
    else:
        exp = registry.get_experiment_record(req.experiment_id)
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")
            
        config = exp.get("config", {})
        if config.get("domain") != "llm":
            raise HTTPException(status_code=400, detail="Only LLM experiments can be started via UI")
            
        temp_config_path = f".tmp_config_{req.experiment_id}.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)
            
        adapter_path = exp.get("checkpoint_path", "")
        if adapter_path and adapter_path.endswith(".pt"):
            adapter_path = os.path.dirname(adapter_path)
        
        cmd = [sys.executable, "-m", "aiforge_cli", "serve-llm", "--config", temp_config_path, "--port", "8001"]
        if adapter_path and os.path.exists(adapter_path):
            cmd.extend(["--adapter", adapter_path])
        
    try:
        # Clean up any lingering process on port 8001 before starting fresh
        try:
            pid_bytes = subprocess.check_output(["lsof", "-ti:8001"], stderr=subprocess.DEVNULL)
            for pid_str in pid_bytes.decode().strip().split():
                if pid_str.isdigit() and int(pid_str) != os.getpid():
                    os.kill(int(pid_str), 9)
        except Exception:
            pass

        os.makedirs("logs", exist_ok=True)
        log_file = open("logs/llm_server.log", "a")
        llm_server_process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
        return {"status": "starting"}
    except Exception as e:
        logger.error(f"Failed to launch LLM server: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/server/stop")
def stop_server():
    global llm_server_process
    stopped = False
    if llm_server_process is not None:
        try:
            llm_server_process.terminate()
            llm_server_process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            llm_server_process.kill()
            llm_server_process.wait(timeout=1.0)
        except Exception:
            pass
        finally:
            llm_server_process = None
        stopped = True

    # Also clean up any lingering process on port 8001
    try:
        pid_bytes = subprocess.check_output(["lsof", "-ti:8001"], stderr=subprocess.DEVNULL)
        for pid_str in pid_bytes.decode().strip().split():
            if pid_str.isdigit() and int(pid_str) != os.getpid():
                os.kill(int(pid_str), 9)
                stopped = True
    except Exception:
        pass

    return {"status": "stopped" if stopped else "not running"}

@app.get("/api/status")
def get_status():
    llm_status = "offline"
    try:
        import urllib.request
        # Check if LLM server is running on default port
        req = urllib.request.Request("http://127.0.0.1:8001/ready", method="GET")
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
        "env_guard": env_guard
    }

@app.get("/api/experiments")
def list_experiments():
    exps = registry.list_experiments()
    # Sort by date descending (latest first)
    exps.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return exps

@app.get("/api/experiments/{exp_id}")
def get_experiment(exp_id: str):
    try:
        exp = registry.get_experiment_record(exp_id)
        return exp
    except FileNotFoundError as e:
        print(f"DEBUG FileNotFound: {e}")
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp

@app.post("/api/generate")
async def generate_text(req: GenerateRequest):
    # Proxy to the local LLM server
    try:
        async with httpx.AsyncClient(trust_env=False, verify=False) as client:
            resp = await client.post(
                "http://127.0.0.1:8001/v1/generate",
                json=req.model_dump(),
                timeout=180.0
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()
    except httpx.RequestError as e:
        logger.error(f"Generate proxy error: {type(e).__name__} - {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"LLM Server non raggiungibile sulla porta 8001 ({type(e).__name__}). Se è in fase di avvio, attendi qualche secondo."
        )

from fastapi.middleware.cors import CORSMiddleware
from aiforge.core.merge_lab import MergeLab, MergeRequest
from fastapi import Request
from aiforge.core.secrets import save_secret, has_secret, check_plaintext_migration

# Add strict CORS to prevent cross-origin requests from malicious sites
ALLOWED_ORIGINS = [
    "http://127.0.0.1:8000", "http://localhost:8000",
    "http://127.0.0.1:8500", "http://localhost:8500"
]
ALLOWED_HOSTS = [
    "127.0.0.1:8000", "localhost:8000",
    "127.0.0.1:8500", "localhost:8500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

@app.get("/api/keys")
def get_keys():
    return {
        "openai": has_secret("openai"),
        "anthropic": has_secret("anthropic"),
        "google": has_secret("google"),
        "legacy_plaintext_warning": check_plaintext_migration()
    }

@app.post("/api/keys")
def save_keys(keys: dict, request: Request):
    host = request.headers.get("host", "")
    if host not in ALLOWED_HOSTS:
        raise HTTPException(status_code=403, detail="Invalid Host header. Binding strictly local.")
        
    origin = request.headers.get("origin", "")
    if origin not in ALLOWED_ORIGINS:
        raise HTTPException(status_code=403, detail="Invalid Origin header. CSRF protection active.")

    placeholder = "●●●●●●●● (Configured)"
    for provider in ["openai", "anthropic", "google"]:
        val = keys.get(provider)
        if val and isinstance(val, str):
            val = val.strip()
            if val and val != placeholder:
                save_secret(provider, val)
    return {"status": "success"}

class ConnectorConfig(BaseModel):
    id: str
    name: str
    base_url: str
    model: Optional[str] = "gemini-1.5-flash"
    api_key: Optional[str] = None

@app.get("/api/connectors")
def get_connectors():
    connectors_path = os.path.join(registry_path, "connectors.json")
    if not os.path.exists(connectors_path):
        # Default connectors
        return [
            {"id": "google", "name": "Google Gemini (OpenAI Std)", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "model": "gemini-1.5-flash", "has_key": has_secret("google")},
            {"id": "openai", "name": "OpenAI Default", "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "has_key": has_secret("openai")},
            {"id": "anthropic", "name": "Anthropic (via Proxy)", "base_url": "", "model": "claude-3-5-sonnet", "has_key": has_secret("anthropic")}
        ]
    
    with open(connectors_path, "r") as f:
        data = json.load(f)
    
    for c in data:
        c["has_key"] = has_secret(c["id"])
        if "model" not in c or not c["model"]:
            c["model"] = "gemini-1.5-flash" if c["id"] == "google" else ("gpt-4o-mini" if c["id"] == "openai" else "local-model")
    return data

@app.post("/api/connectors")
def save_connector(conn: ConnectorConfig, request: Request):
    host = request.headers.get("host", "")
    if host not in ALLOWED_HOSTS:
        raise HTTPException(status_code=403, detail="Invalid Host header.")
        
    connectors_path = os.path.join(registry_path, "connectors.json")
    data = []
    if os.path.exists(connectors_path):
        with open(connectors_path, "r") as f:
            data = json.load(f)
            
    # Update or append
    updated = False
    for i, c in enumerate(data):
        if c["id"] == conn.id:
            data[i]["name"] = conn.name
            data[i]["base_url"] = conn.base_url
            data[i]["model"] = conn.model or "gemini-1.5-flash"
            updated = True
            break
    if not updated:
        data.append({"id": conn.id, "name": conn.name, "base_url": conn.base_url, "model": conn.model or "gemini-1.5-flash"})
        
    with open(connectors_path, "w") as f:
        json.dump(data, f, indent=2)
        
    if conn.api_key and conn.api_key != "●●●●●●●● (Configured)":
        save_secret(conn.id, conn.api_key)
        
    return {"status": "success"}

@app.delete("/api/connectors/{conn_id}")
def delete_connector(conn_id: str, request: Request):
    host = request.headers.get("host", "")
    if host not in ALLOWED_HOSTS:
        raise HTTPException(status_code=403, detail="Invalid Host header.")
        
    connectors_path = os.path.join(registry_path, "connectors.json")
    if os.path.exists(connectors_path):
        with open(connectors_path, "r") as f:
            data = json.load(f)
        data = [c for c in data if c["id"] != conn_id]
        with open(connectors_path, "w") as f:
            json.dump(data, f, indent=2)
            
    # We leave the keychain secret intact to be safe, or we could overwrite it.
    return {"status": "success"}

class TestConnectorRequest(BaseModel):
    id: Optional[str] = None
    base_url: str
    api_key: Optional[str] = None

@app.post("/api/connectors/test")
async def test_connector(req: TestConnectorRequest):
    """Test connectivity and latency to an OpenAI-compatible endpoint."""
    key = req.api_key
    if not key or key == "●●●●●●●● (Configured)":
        if req.id:
            from aiforge.core.secrets import get_secret
            key = get_secret(req.id)
            
    headers = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
        
    base = req.base_url.rstrip("/")
    test_urls = [f"{base}/models", base]
    start_time = time.time()
    last_err = "Nessuna risposta dal server"
    
    async with httpx.AsyncClient(trust_env=False, verify=False, timeout=6.0) as client:
        for u in test_urls:
            try:
                resp = await client.get(u, headers=headers)
                latency_ms = int((time.time() - start_time) * 1000)
                if resp.status_code == 200:
                    return {"status": "ok", "latency_ms": latency_ms, "message": f"Connesso ({latency_ms} ms)"}
                elif resp.status_code in [401, 403]:
                    return {"status": "auth_error", "latency_ms": latency_ms, "message": f"Server raggiungibile ma chiave API non valida (HTTP {resp.status_code})"}
                elif resp.status_code < 500:
                    return {"status": "ok", "latency_ms": latency_ms, "message": f"Endpoint raggiungibile (HTTP {resp.status_code}, {latency_ms} ms)"}
            except Exception as e:
                last_err = str(e)
                
    return {"status": "error", "message": f"Connessione fallita: {last_err}"}

        
# --- Model Management API ---

import sys
import asyncio
from fastapi.responses import StreamingResponse

class ForgeRequest(BaseModel):
    name: str
    model: str
    topic: str
    qlora: bool = False
    method: str = "quick"
    target_samples: int = 100
    turn_type: str = "single_turn"

@app.post("/api/forge/start")
async def start_forge(req: ForgeRequest):
    async def event_generator():
        cmd = [
            sys.executable,
            "-m", "aiforge.core.run_forge",
            "--name", req.name,
            "--model", req.model,
            "--topic", req.topic,
            "--method", req.method,
            "--target-samples", str(req.target_samples),
            "--turn-type", req.turn_type
        ]
        if req.qlora:
            cmd.append("--qlora")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                text = line.decode('utf-8', errors='replace').strip()
                if text:
                    yield f"data: {text}\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            if process.returncode is None:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except Exception:
                    process.kill()
                    await process.wait()
            yield f"data: [DONE]\n\n"
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")

SUPPORTED_BASE_MODELS = [
    {
        "id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "name": "Qwen 2.5 Coder 0.5B Instruct",
        "desc": "Architettura avanzata (2024), perfetta per Mac da 8 GB. Prestazioni di codice eccezionali con ~1 GB di RAM.",
        "params": "0.5B",
        "recommended": True
    },
    {
        "id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "name": "Qwen 2.5 Coder 1.5B Instruct",
        "desc": "Capacità di ragionamento superiore per codice C/Python complesso. Richiede circa 2.5 GB di RAM.",
        "params": "1.5B",
        "recommended": False
    },
    {
        "id": "openai-community/gpt2",
        "name": "GPT-2 Base (124M)",
        "desc": "Modello leggero classico, usato come banco di prova iniziale.",
        "params": "124M",
        "recommended": False
    }
]

@app.get("/api/models")
def get_available_models():
    # Controlla dinamicamente la cache di HuggingFace
    import os
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    results = []
    for m in SUPPORTED_BASE_MODELS:
        model_copy = dict(m)
        folder_name = "models--" + m["id"].replace("/", "--")
        is_downloaded = os.path.exists(os.path.join(hf_cache_dir, folder_name))
        model_copy["downloaded"] = is_downloaded
        results.append(model_copy)
        
    return results

@app.post("/api/models/download")
async def download_model(payload: dict):
    model_id = payload.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="Missing model_id")
    # Scarica tokenizer e pesi tramite HuggingFace transformers in background
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # Download tokenizer
        AutoTokenizer.from_pretrained(model_id)
        # Download model
        AutoModelForCausalLM.from_pretrained(model_id)
        return {"status": "success", "message": f"Modello {model_id} scaricato con successo in cache!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore download: {str(e)}")

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "templates", "index.html"))

@app.post("/api/merge/preflight")
def merge_preflight_post(req: MergeRequest):
    lab = MergeLab(registry)
    return lab.validate_pre_merge(req)

@app.get("/api/merge/preflight")
def merge_preflight(model_a: str, model_b: str, strategy: str):
    lab = MergeLab(registry)
    req = MergeRequest(strategy=strategy, model_a_id=model_a, model_b_id=model_b)
    return lab.validate_pre_merge(req)

@app.post("/api/merge/execute")
def merge_execute(req: MergeRequest):
    lab = MergeLab(registry)
    try:
        exp_id = lab.execute_merge(req)
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
    exporter = OllamaExporter(registry)
    try:
        res = exporter.export(
            experiment_id=req.experiment_id,
            ollama_model_name=req.ollama_model_name,
            system_prompt=req.system_prompt
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class GGUFExportRequest(BaseModel):
    experiment_id: str
    quantization: str = "Q4_K_M"
    custom_name: Optional[str] = None

@app.post("/api/export/gguf")
def export_to_gguf(req: GGUFExportRequest):
    from aiforge.deployment.gguf_export import GGUFExporter
    exporter = GGUFExporter(registry)
    try:
        res = exporter.export(
            experiment_id=req.experiment_id,
            quantization=req.quantization,
            custom_name=req.custom_name
        )
        return res
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
    evaluator = MergeEvaluator(registry)
    try:
        res = evaluator.evaluate_merge(
            model_a_id=req.model_a_id,
            model_b_id=req.model_b_id,
            merged_model_id=req.merged_model_id,
            sample_texts=req.sample_texts
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


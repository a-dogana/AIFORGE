import os
import sys
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aiforge.core.config import ExperimentConfig
from aiforge.llm.engine import LLMEngine

# --- Models ---
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to generate text for.", max_length=2000)
    max_new_tokens: int = Field(50, description="Max tokens to generate.", gt=0, le=512)
    temperature: float = Field(0.7, description="Sampling temperature.", ge=0.0, le=2.0)

class GenerateResponse(BaseModel):
    text: str
    model_version: str
    inference_ms: float

# --- State ---
class AppState:
    engine: Optional[LLMEngine] = None
    model_id: str = "unknown"
    req_count: int = 0
    err_count: int = 0

state = AppState()
logger = logging.getLogger("aiforge.llm_server")

@asynccontextmanager
async def lifespan(app: FastAPI):
    adapter_dir = os.environ.get("AIFORGE_LLM_ADAPTER")
    config_path = os.environ.get("AIFORGE_LLM_CONFIG")
    
    if not config_path:
        logger.warning("No config provided. Starting in unready state.")
    else:
        try:
            config = ExperimentConfig.from_yaml(config_path)
            engine = LLMEngine(config)
            engine.load(local_path=adapter_dir)
            state.engine = engine
            state.model_id = adapter_dir if adapter_dir else config.architecture.llm_config.model_id
            logger.info("LLM loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            
    yield
    state.engine = None

app = FastAPI(title="AIFORGE LLM Generation Server", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.10.0"}

@app.get("/ready")
def readiness_check():
    if state.engine is None:
        raise HTTPException(status_code=503, detail="LLM not loaded.")
    return {"status": "ready", "model_id": state.model_id}

@app.post("/v1/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    if state.engine is None:
        raise HTTPException(status_code=503, detail="LLM not loaded.")
        
    import time
    try:
        start = time.time()
        out_text = state.engine.generate(
            request.prompt, 
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature
        )
        latency = (time.time() - start) * 1000
        state.req_count += 1
        return GenerateResponse(text=out_text, model_version=state.model_id, inference_ms=latency)
    except ValueError as ve:
        state.err_count += 1
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        state.err_count += 1
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during generation.")

def run_llm_server(config_path: str, adapter_dir: Optional[str] = None, host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    os.environ["AIFORGE_LLM_CONFIG"] = config_path
    if adapter_dir:
        os.environ["AIFORGE_LLM_ADAPTER"] = adapter_dir
        
    if host != "127.0.0.1":
        print("WARNING: Binding to non-loopback interface. Ensure this service is not publicly exposed without authentication.")
        
    uvicorn.run(app, host=host, port=port, log_level="info")

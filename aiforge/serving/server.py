import os
import sys
import json
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from aiforge.deployment.inference import InferenceEngine

# --- Models ---
class PredictRequest(BaseModel):
    data: List[List[float]] = Field(..., description="Batch of tabular feature rows.")

class PredictResponse(BaseModel):
    predictions: List[float]
    model_version: str
    backend: str
    inference_ms: float

# --- State ---
class AppState:
    engine: Optional[InferenceEngine] = None
    manifest: Dict[str, Any] = {}
    backend: str = "pytorch"
    package_dir: str = ""
    req_count: int = 0
    err_count: int = 0

state = AppState()
logger = logging.getLogger("aiforge.server")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    package_dir = os.environ.get("AIFORGE_DEPLOY_PACKAGE")
    backend = os.environ.get("AIFORGE_DEPLOY_BACKEND", "pytorch")
    
    if not package_dir:
        logger.warning("No AIFORGE_DEPLOY_PACKAGE provided. Starting in unready state.")
    else:
        try:
            logger.info(f"Loading deployment package from {package_dir} using {backend}...")
            state.engine = InferenceEngine(package_dir, backend=backend)
            state.manifest = state.engine.manifest
            state.backend = backend
            state.package_dir = package_dir
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # We don't crash here so `/health` still works, but `/ready` will fail
            
    yield
    # Shutdown
    state.engine = None

app = FastAPI(title="AIFORGE Inference Server", lifespan=lifespan)

# --- Endpoints ---

@app.get("/health")
def health_check():
    """Basic process health check."""
    return {"status": "ok", "version": "0.8.0"}

@app.get("/ready")
def readiness_check():
    """Readiness check to confirm the model is loaded."""
    if state.engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded or failed to load.")
    return {"status": "ready", "experiment_id": state.manifest.get("experiment_id")}

@app.get("/v1/model")
def get_model_metadata():
    """Returns safe model metadata without exposing internal paths."""
    if state.engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
        
    return {
        "experiment_id": state.manifest.get("experiment_id"),
        "experiment_name": state.manifest.get("experiment_name"),
        "architecture": state.manifest.get("architecture"),
        "task": state.manifest.get("task"),
        "domain": state.manifest.get("domain"),
        "active_backend": state.backend,
        "schema_version": state.manifest.get("schema_version")
    }

@app.post("/v1/predict", response_model=PredictResponse)
def predict_tabular(request: PredictRequest):
    if state.engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
        
    if state.manifest.get("domain") != "vision" and state.manifest.get("architecture") != "mlp":
        if state.manifest.get("domain") != "tabular":
            # If vision, we probably expect images, but AIFORGE mlp might be used on images flattened.
            if state.manifest.get("architecture") == "cnn":
                raise HTTPException(status_code=400, detail="This endpoint only supports tabular/MLP data. For CNN use /v1/predict/images")
            
    try:
        start_time = time.time()
        
        # Batch size limit
        if len(request.data) > 1024:
            raise HTTPException(status_code=400, detail="Batch size exceeds limit of 1024.")
            
        # Inference
        out = state.engine.predict(request.data)
        
        # Determine class predictions
        if len(out.shape) > 1 and out.shape[1] > 1:
            # Classification
            preds = np.argmax(out, axis=1).tolist()
        else:
            # Regression / Binary
            preds = out.flatten().tolist()
            
        latency_ms = (time.time() - start_time) * 1000
        state.req_count += 1
        
        return PredictResponse(
            predictions=preds,
            model_version=state.manifest.get("experiment_id", "unknown"),
            backend=state.backend,
            inference_ms=latency_ms
        )
    except HTTPException:
        # Re-raise HTTP exceptions so they retain their status code
        state.err_count += 1
        raise
    except ValueError as ve:
        state.err_count += 1
        raise HTTPException(status_code=400, detail=f"Input validation error: {ve}")
    except Exception as e:
        state.err_count += 1
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")

@app.get("/metrics")
def get_metrics():
    return {
        "requests_total": state.req_count,
        "errors_total": state.err_count,
        "backend": state.backend
    }

def run_server(package_dir: str, host: str = "127.0.0.1", port: int = 8000, backend: str = "pytorch"):
    import uvicorn
    os.environ["AIFORGE_DEPLOY_PACKAGE"] = package_dir
    os.environ["AIFORGE_DEPLOY_BACKEND"] = backend
    
    if host != "127.0.0.1":
        print("WARNING: Binding to non-loopback interface. Ensure this service is not publicly exposed without authentication.")
        
    uvicorn.run(app, host=host, port=port, log_level="info")

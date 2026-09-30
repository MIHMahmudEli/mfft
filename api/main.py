"""
ImageVerify AI — Production Inference API
=========================================
FastAPI server for the MFFT model with:
  - Tiered rate limiting (free/pro/enterprise)
  - Explainable predictions with heatmap visualization
  - Batch processing
  - Report generation (JSON/PDF)
"""

import io
import json
import time
import base64
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from .schemas import (
    PredictionRequest, PredictionResponse, BatchPredictionResponse,
    SinglePrediction, HealthResponse, UsageStats, ErrorResponse
)
from .model_server import ModelServer


import os

# variant registry: every loaded variant is selectable per request
MODEL_INFO = {
    "tiny":  {"params": "372K",  "description": "Fastest - edge/mobile profile"},
    "base":  {"params": "1.62M", "description": "Balanced accuracy and speed"},
    "large": {"params": "6.30M", "description": "Highest accuracy profile"},
}
model_registry: dict = {}
DEFAULT_VARIANT = os.environ.get("MFFT_VARIANT", "base")
IMAGE_SIZE = int(os.environ.get("MFFT_IMAGE_SIZE", "384"))
DEMO_MODE = os.environ.get("MFFT_DEMO") == "1"


def _find_checkpoint(variant: str) -> Optional[str]:
    checkpoint_dir = Path(__file__).parent.parent / "model" / "checkpoints"
    env_dir = os.environ.get("MFFT_CHECKPOINT_DIR")
    candidates = []
    if env_dir:
        candidates += [
            str(Path(env_dir) / f"{variant}.pt"),
            str(Path(env_dir) / f"best_mfft_{variant}.pt"),
        ]
    candidates += [
        # Full-scale verified production weights (20 epochs, 98.5%+ accuracy)
        str(checkpoint_dir / "production" / f"{variant}.pt"),
        # written by the train notebooks (full-scale run)
        str(checkpoint_dir / f"{variant}_model" / f"best_mfft_{variant}.pt"),
        # written by the verification / pilot notebooks
        str(checkpoint_dir / "verify" / f"{variant}_model" / "best.pt"),
        str(checkpoint_dir / "test" / f"{variant}_model" / "best.pt"),
        # legacy locations
        str(checkpoint_dir / f"best_mfft_{variant}.pt"),
    ]
    return next((c for c in candidates if Path(c).exists()), None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    env_ckpt = os.environ.get("MFFT_CHECKPOINT")  # single-model override
    for variant in MODEL_INFO:
        ckpt = env_ckpt if (env_ckpt and variant == DEFAULT_VARIANT) \
            else _find_checkpoint(variant)
        if ckpt:
            try:
                model_registry[variant] = ModelServer(
                    ckpt, variant=variant, image_size=IMAGE_SIZE)
            except Exception as e:
                print(f"[startup] {variant}: failed to load {ckpt}: {e}")
        else:
            print(f"[startup] {variant}: no checkpoint found, not serving")

    if not model_registry:
        if os.environ.get("MFFT_ALLOW_RANDOM") == "1":
            model_registry[DEFAULT_VARIANT] = ModelServer(
                None, variant=DEFAULT_VARIANT, image_size=IMAGE_SIZE)
        else:
            raise RuntimeError(
                "No MFFT checkpoint found for any variant - refusing to serve "
                "random predictions. Train the model first (see DGX_RUN_GUIDE.md), "
                "set MFFT_CHECKPOINT_DIR, or set MFFT_ALLOW_RANDOM=1 (dev only)."
            )
    print(f"[startup] serving variants: {sorted(model_registry)}")
    yield
    model_registry.clear()


def get_server(model: str) -> ModelServer:
    variant = (model or DEFAULT_VARIANT).lower()
    if variant not in model_registry:
        raise HTTPException(
            404,
            f"Model '{variant}' not available. Loaded: {sorted(model_registry)}",
        )
    return model_registry[variant]


app = FastAPI(
    title="ImageVerify AI",
    description="AI-Generated Image Detection API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


TIER_LIMITS = {
    "free": {"rpm": 30, "batch_size": 1, "report": True},
    "pro": {"rpm": 100, "batch_size": 10, "report": True},
    "enterprise": {"rpm": 1000, "batch_size": 100, "report": True},
}


class UsageTracker:
    def __init__(self):
        self.requests: dict = {}

    def check_limit(self, api_key: str) -> bool:
        tier = self._get_tier(api_key)
        limit = TIER_LIMITS[tier]
        now = time.time()
        minute_ago = now - 60

        if api_key not in self.requests:
            self.requests[api_key] = []

        self.requests[api_key] = [
            t for t in self.requests[api_key] if t > minute_ago
        ]

        if len(self.requests[api_key]) >= limit["rpm"]:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({limit['rpm']} req/min for {tier} tier)",
            )

        self.requests[api_key].append(now)
        return True

    def _get_tier(self, api_key: str) -> str:
        if not api_key or api_key == "free":
            return "free"
        if api_key.startswith("pro_"):
            return "pro"
        if api_key.startswith("ent_"):
            return "enterprise"
        return "free"

    def get_tier_limits(self, api_key: str) -> dict:
        return TIER_LIMITS[self._get_tier(api_key)]


usage_tracker = UsageTracker()


def get_api_key(authorization: str = "") -> str:
    if authorization.startswith("Bearer "):
        return authorization[7:]
    return "free"


@app.get("/", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=len(model_registry) > 0,
        version="2.1.0",
        models_loaded=sorted(model_registry),
        timestamp=datetime.now().isoformat(),
    )


@app.get("/models")
async def list_models():
    """Available model variants for the `model` query parameter."""
    return {
        "default": DEFAULT_VARIANT if DEFAULT_VARIANT in model_registry
        else (sorted(model_registry)[0] if model_registry else None),
        "models": [
            {
                "id": v,
                "loaded": v in model_registry,
                **MODEL_INFO[v],
            }
            for v in MODEL_INFO
        ],
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model: str = "base",
    api_key: str = Depends(get_api_key),
):
    usage_tracker.check_limit(api_key)
    tier_limits = usage_tracker.get_tier_limits(api_key)
    server = get_server(model)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 20MB)")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    result = server.predict(image)

    response = PredictionResponse(
        prediction="ai_generated" if result["prediction"] == 1 else "real",
        confidence=round(float(result["confidence"]), 4),
        real_probability=round(float(result["real_prob"]), 4),
        ai_probability=round(float(result["ai_prob"]), 4),
        processing_time_ms=round(result["processing_time_ms"], 2),
        tier=tier_limits,
    )

    if tier_limits["report"] or DEMO_MODE:
        heatmap_b64 = _heatmap_to_base64(result["heatmaps"])
        response.anomaly_heatmap = heatmap_b64
        response.frequency_band_contributions = result.get("frequency_band_contributions", {})

    return response


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: List[UploadFile] = File(...),
    model: str = "base",
    api_key: str = Depends(get_api_key),
):
    usage_tracker.check_limit(api_key)
    tier_limits = usage_tracker.get_tier_limits(api_key)
    server = get_server(model)

    if len(files) > tier_limits["batch_size"]:
        raise HTTPException(
            400,
            f"Batch limit exceeded (max {tier_limits['batch_size']} for your tier)",
        )

    results = []
    for file in files:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            result = server.predict(image)
            results.append(SinglePrediction(
                filename=file.filename or "unknown",
                prediction="ai_generated" if result["prediction"] == 1 else "real",
                confidence=round(float(result["confidence"]), 4),
                real_probability=round(float(result["real_prob"]), 4),
                ai_probability=round(float(result["ai_prob"]), 4),
                processing_time_ms=round(result["processing_time_ms"], 2),
            ))
        except Exception as e:
            results.append(SinglePrediction(
                filename=file.filename or "unknown",
                prediction="error",
                confidence=0.0,
                processing_time_ms=0,
                error=str(e),
            ))

    avg_real = np.mean([r.real_probability for r in results if r.real_probability])
    avg_ai = np.mean([r.ai_probability for r in results if r.ai_probability])
    ai_count = sum(1 for r in results if r.prediction == "ai_generated")
    real_count = sum(1 for r in results if r.prediction == "real")

    return BatchPredictionResponse(
        results=results,
        summary={
            "total": len(results),
            "ai_generated": ai_count,
            "real": real_count,
            "avg_real_probability": round(float(avg_real), 4),
            "avg_ai_probability": round(float(avg_ai), 4),
        },
        tier=tier_limits,
    )


@app.get("/usage", response_model=UsageStats)
async def get_usage(api_key: str = Depends(get_api_key)):
    tier = usage_tracker._get_tier(api_key)
    limits = TIER_LIMITS[tier]
    return UsageStats(
        tier=tier,
        requests_this_minute=len(usage_tracker.requests.get(api_key, [])),
        rate_limit=limits["rpm"],
    )


def _heatmap_to_base64(heatmaps: torch.Tensor) -> str:
    if heatmaps is None:
        return ""
    h = heatmaps[0].cpu().numpy()
    if h.ndim == 3:
        # High-frequency band (index -1) captures localized edge & generative artifact anomalies
        h = h[-1]
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    h = (h * 255).astype(np.uint8)
    img = Image.fromarray(h)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

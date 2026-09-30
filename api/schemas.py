from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class PredictionRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64-encoded image")


class SinglePrediction(BaseModel):
    filename: str = Field(..., description="Input filename")
    prediction: str = Field(..., description="real or ai_generated")
    confidence: float = Field(..., ge=0.0, le=1.0)
    real_probability: float = Field(0.0, ge=0.0, le=1.0)
    ai_probability: float = Field(0.0, ge=0.0, le=1.0)
    processing_time_ms: float = 0.0
    anomaly_heatmap: Optional[str] = None
    error: Optional[str] = None
    frequency_band_contributions: Optional[Dict[str, float]] = None


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="real or ai_generated")
    confidence: float = Field(..., ge=0.0, le=1.0)
    real_probability: float = Field(..., ge=0.0, le=1.0)
    ai_probability: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: float
    anomaly_heatmap: Optional[str] = None
    frequency_band_contributions: Optional[Dict[str, float]] = None
    tier: Optional[Dict[str, Any]] = None


class BatchPredictionResponse(BaseModel):
    results: List[SinglePrediction]
    summary: Dict[str, Any]
    tier: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    models_loaded: List[str] = Field(default_factory=list, description="Currently loaded model variants")
    timestamp: str


class UsageStats(BaseModel):
    tier: str
    requests_this_minute: int
    rate_limit: int


class ErrorResponse(BaseModel):
    detail: str
    status_code: int

# MFFT Deployment Guide

This document describes the complete deployment architecture, configuration, and operation of the **Multi-Frequency Fusion Transformer (MFFT)** AI image detection system.

---

## 1. System Architecture

```
                    ┌──────────────────────────────────────────┐
                    │               End Users                  │
                    │       (Browser / Desktop / Mobile)       │
                    └─────────────────────┬────────────────────┘
                                          │
                           HTTP / HTTPS   ▼
                    ┌──────────────────────────────────────────┐
                    │        Nginx Reverse Proxy (:80/:443)    │
                    │  - SSL termination, static routing       │
                    │  - Rate-limit headers, max upload 50MB   │
                    └──────────────┬──────────────────┬────────┘
                                   │                  │
                Path: /            ▼                  ▼ Path: /api/* or /predict
       ┌───────────────────────────────┐   ┌───────────────────────────────┐
       │   Web Interface (Next.js 14)  │   │   Inference API (FastAPI)     │
       │   - Port 3000                 │   │   - Port 8000                 │
       │   - TypeScript, Tailwind CSS  │──►│   - PyTorch 2.6+ MFFT Engine  │
       │   - Recharts visualizer       │   │   - Multi-variant loading     │
       │   - Client-side rewrites      │   │   - Explainability generator  │
       └───────────────────────────────┘   └──────────────┬────────────────┘
                                                          │
                                                          ▼
                                           ┌───────────────────────────────┐
                                           │   Verified Production Models  │
                                           │   - MFFT-Tiny (372K params)   │
                                           │   - MFFT-Base (1.62M params)  │
                                           │   - MFFT-Large (6.30M params) │
                                           └───────────────────────────────┘
```

The system comprises three decoupled tiers:

1. **Inference API (`api/`)**: A high-throughput FastAPI application that hosts the PyTorch MFFT models. It extracts frequency features across radial Fourier bands (low, mid, high), predicts whether an image is authentic or synthetic, isolates high-frequency spatial anomaly heatmaps, and calculates per-band contribution vectors.
2. **Web Interface (`web/`)**: A modern, responsive Next.js 14 application with drag-and-drop file ingestion, real-time client-side preview, dynamic model switching (`tiny`, `base`, `large`), probability bars, Recharts frequency-contribution visualization, and anomaly heatmap overlays.
3. **Nginx & Docker Compose (`compose.yaml`, `nginx.conf`)**: Production containerization with unified port forwarding, volume mounts for read-only production checkpoints, healthchecks, and internal service networking.

---

## 2. Production Models & Checkpoints

The deployment system serves models trained for 20 epochs on the 120,000-image canonical balanced dataset (84,000 train / 18,000 val / 18,000 test; SHA-256 `0043da4814d19e80...`):

| Model Variant | Parameters | File Size | Test Accuracy | Test AUC-ROC | Target Profile | Checkpoint Path |
|---|---|---|---|---|---|---|
| **MFFT-Tiny** | 372,834 | 4.37 MB | 98.59% | 99.88% | Mobile, edge, low-latency API | `model/checkpoints/production/tiny.pt` |
| **MFFT-Base** | 1,622,690 | 18.68 MB | 98.58% | 99.89% | Production default (balanced) | `model/checkpoints/production/base.pt` |
| **MFFT-Large** | 6,303,554 | 72.25 MB | 98.11% | 99.68% | Maximum capacity & forensic audits | `model/checkpoints/production/large.pt` |

To download or refresh the production models directly from Hugging Face:
```bash
python scripts/download_production_models.py
```

---

## 3. Local Development Setup

To run both services natively for development or debugging:

### Step 1: Start the Inference API
```bash
# In repository root
pip install -r api/requirements.txt

# Start FastAPI server with live reloading
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
The API server will automatically locate and load `tiny.pt`, `base.pt`, and `large.pt` from `model/checkpoints/production/`. Verify at `http://127.0.0.1:8000/`.

### Step 2: Start the Web Interface
```bash
cd web
npm install
npm run dev
```
Open `http://localhost:3000` in your browser. Next.js proxies all `/api/*` calls directly to `http://localhost:8000`.

---

## 4. Production Deployment with Docker Compose

Docker Compose provisions the complete stack (FastAPI + Next.js + Nginx) with automated startup sequencing, GPU acceleration, and health checks.

### Prerequisites
- Docker Engine 24.0+ & Docker Compose v2+
- (Optional for GPU) NVIDIA Container Toolkit (`nvidia-ctk`)

### Step 1: Launch Stack
```bash
# Build and run containers in detached mode
docker compose up -d --build
```

### Step 2: Verify Service Status
```bash
docker compose ps
```
Expected output:
```text
NAME                                IMAGE                              COMMAND                  SERVICE   STATUS              PORTS
ai-image-detection-research-api-1   ai-image-detection-research-api    "uvicorn api.main:ap…"   api       Up (healthy)        0.0.0.0:8000->8000/tcp
ai-image-detection-research-web-1   ai-image-detection-research-web    "node server.js"         web       Up                  0.0.0.0:3000->3000/tcp
ai-image-detection-research-nginx-1 nginx:alpine                       "/docker-entrypoint.…"   nginx     Up                  0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp
```

### Step 3: Access
- Web Interface: `http://localhost/` or `http://localhost:3000`
- API Health Check: `http://localhost/api/` or `http://localhost:8000/`
- Interactive Swagger Docs: `http://localhost:8000/docs`

---

## 5. Hugging Face Spaces Deployment

The inference backend can be published to Hugging Face Spaces with a single automated script.

### Pre-flight Setup
Ensure your Hugging Face token is exported:
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Dry Run (Verify Assembly)
```bash
python scripts/deploy_hf_space.py --dry-run
```
This stages the API server, models, requirements, and standalone Dockerfile without pushing.

### Deploy
```bash
python scripts/deploy_hf_space.py --token $HF_TOKEN --space-id MohsinElis/mfft-api
```
Once deployed, the space will be live at `https://mohsinelis-mfft-api.hf.space`. You can configure the frontend to talk to this endpoint by setting `NEXT_PUBLIC_API_URL=https://mohsinelis-mfft-api.hf.space` in `web/.env.local`.

---

## 6. API Reference & Verification

All prediction requests accept an optional `?model=tiny|base|large` parameter (defaults to `base`).

### 1. Health & Registry
```bash
# Health Check
curl -s http://localhost:8000/

# List Models
curl -s http://localhost:8000/models
```

Response:
```json
{
  "default": "base",
  "models": [
    { "id": "tiny", "loaded": true, "params": "372K", "description": "Fastest — edge & mobile profile" },
    { "id": "base", "loaded": true, "params": "1.62M", "description": "Balanced accuracy and speed (98.58% acc)" },
    { "id": "large", "loaded": true, "params": "6.30M", "description": "Highest capacity profile (98.11% acc)" }
  ]
}
```

### 2. Single Image Prediction
```bash
curl -X POST "http://localhost:8000/predict?model=base" \
  -H "Authorization: Bearer free" \
  -F "file=@test_sample.jpg"
```

Response:
```json
{
  "prediction": "ai_generated",
  "confidence": 0.9854,
  "real_probability": 0.0146,
  "ai_probability": 0.9854,
  "processing_time_ms": 38.2,
  "anomaly_heatmap": "data:image/png;base64,iVBORw0KGgoAAA...",
  "frequency_analysis": {
    "band_contributions": {
      "band_0": 0.182,
      "band_1": 0.315,
      "band_2": 0.503
    }
  },
  "tier": {
    "rpm": 30,
    "batch_size": 1,
    "report": true
  },
  "model_used": "base"
}
```

### 3. Batch Image Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch?model=large" \
  -H "Authorization: Bearer pro" \
  -F "files=@image1.png" \
  -F "files=@image2.jpg"
```

---

## 7. Operational Best Practices & Scaling

- **Device Auto-Detection**: The API automatically utilizes an NVIDIA GPU if CUDA is available, or gracefully falls back to multicore CPU inference.
- **Worker Scaling**: For high-concurrency production deployments behind Nginx, increase Uvicorn workers in `Dockerfile.api`:
  ```bash
  CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
  ```
- **Caching**: The Next.js frontend uses static optimization with client-side state caching in `localStorage` for user model preferences.
- **Security**: In production, configure SSL certificates in `nginx.conf` and update `server_name` to your registered domain.

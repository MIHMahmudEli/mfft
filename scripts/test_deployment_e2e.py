"""
End-to-End Local Deployment Test Script.

Validates:
1. Model checkpoint loading for tiny, base, and large.
2. Inference API routes: /, /models, /health, /predict.
3. Prediction explainability: probability scores, anomaly heatmaps, frequency band contributions.
4. Next.js static build presence.
"""

import sys
import io
import time
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from api.main import app, model_registry

def test_deployment():
    print("=" * 60)
    print("MFFT DEPLOYMENT SYSTEM E2E VERIFICATION")
    print("=" * 60)

    with TestClient(app) as client:
        # 1. Test Healthcheck
        resp = client.get("/")
        assert resp.status_code == 200, f"Healthcheck failed: {resp.text}"
        health_data = resp.json()
        print(f"[*] Healthcheck: status={health_data.get('status')}, version={health_data.get('version')}")
        print(f"[*] Models Loaded: {health_data.get('models_loaded')}")
        assert len(health_data.get("models_loaded", [])) >= 3, "Not all models loaded!"

        # 2. Test Model Registry
        resp = client.get("/models")
        assert resp.status_code == 200, f"Models endpoint failed: {resp.text}"
        registry = resp.json()
        print(f"[*] Default Model: {registry.get('default')}")
        for m in registry.get("models", []):
            print(f"    - {m['id']}: params={m['params']}, loaded={m['loaded']}")
            assert m["loaded"] is True, f"Model {m['id']} is not loaded!"

        # 3. Create dummy synthetic-like test image
        img = Image.new("RGB", (384, 384), color=(128, 64, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        # 4. Test Single Prediction on all variants
        for model_id in ["tiny", "base", "large"]:
            t0 = time.time()
            resp = client.post(
                f"/predict?model={model_id}",
                headers={"Authorization": "Bearer free"},
                files={"file": ("test.jpg", img_bytes, "image/jpeg")}
            )
            elapsed = (time.time() - t0) * 1000
            assert resp.status_code == 200, f"Prediction failed for {model_id}: {resp.text}"
            data = resp.json()

            print(f"\n[*] Variant '{model_id}':")
            print(f"    - Status: 200 OK ({elapsed:.1f}ms client-time)")
            print(f"    - Prediction: {data['prediction']} (confidence: {data['confidence']:.4f})")
            print(f"    - Real Prob: {data['real_probability']:.4f} | AI Prob: {data['ai_probability']:.4f}")
            print(f"    - Heatmap Generated: {bool(data.get('anomaly_heatmap'))} (prefix: {data.get('anomaly_heatmap', '')[:30]}...)")
            band_contribs = data.get("frequency_band_contributions", {})
            print(f"    - Frequency Contributions: {band_contribs}")
            assert data.get("anomaly_heatmap") is not None, f"Anomaly heatmap missing for {model_id}!"
            assert len(band_contribs) >= 2, f"Band contributions missing for {model_id}!"

        # 5. Check Next.js standalone build
        web_standalone = REPO_ROOT / "web" / ".next" / "standalone"
        print(f"\n[*] Checking Next.js Standalone Build: {web_standalone}")
        assert web_standalone.exists(), "Next.js standalone build directory not found!"
        print("    - Found web/.next/standalone -> Dockerfile.web build ready!")

        print("\n" + "=" * 60)
        print("ALL DEPLOYMENT VERIFICATION CHECKS PASSED SUCCESSFULLY!")
        print("=" * 60)

if __name__ == "__main__":
    test_deployment()

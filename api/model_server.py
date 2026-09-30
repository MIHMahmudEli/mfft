import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "model" / "src"))
from model import MFFTWithExplainability, build_mfft


class ModelServer:
    """
    Wraps the MFFT model for production inference.
    Handles model loading, preprocessing, prediction, and warmup.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, variant: str = "base",
                 image_size: int = 384):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.variant = variant
        self.image_size = image_size
        print(f"[ModelServer] Device: {self.device} | Variant: {variant} | Size: {image_size}")

        self.model = build_mfft(variant)
        self.model = self.model.to(self.device)
        self.is_loaded = False

        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)
            self.is_loaded = True
            print(f"[ModelServer] Model loaded from {checkpoint_path}")
        else:
            print(f"[ModelServer] No checkpoint found at {checkpoint_path}")
            print(f"[ModelServer] Running with untrained weights (random initialization)")

        self.warmup()

    def load_checkpoint(self, path: str):
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            state = torch.load(path, map_location=self.device, weights_only=False)

        raw_state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state

        # Remap keys between cluster training pipeline and local model.py naming conventions
        cleaned_state = {}
        for k, v in raw_state.items():
            nk = k.replace(".stages.", ".blocks.").replace("fga.gate.", "freq_guided_attn.freq_gate.")
            if "num_batches_tracked" in nk:
                continue
            cleaned_state[nk] = v

        self.model.load_state_dict(cleaned_state, strict=False)
        self.is_loaded = True

    def warmup(self, n_warmup: int = 3):
        dummy = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
        self.model.eval()
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self.model(dummy)
        print(f"[ModelServer] Warmup complete ({n_warmup} iterations)")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        start = time.time()

        input_tensor = self._preprocess(image)
        logits, heatmaps = self.model(input_tensor, return_heatmap=True)

        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

        bands = self.model.decomposer(input_tensor)
        freq_magnitudes = [band.abs().mean().item() for band in bands]

        if len(freq_magnitudes) == 4:
            band_contribs = {
                "low_frequency": freq_magnitudes[0],
                "low_mid_frequency": freq_magnitudes[1],
                "mid_high_frequency": freq_magnitudes[2],
                "high_frequency": freq_magnitudes[3],
            }
        else:
            band_contribs = {
                "low_frequency": freq_magnitudes[0],
                "mid_frequency": freq_magnitudes[1] if len(freq_magnitudes) > 1 else 0.0,
                "high_frequency": freq_magnitudes[2] if len(freq_magnitudes) > 2 else 0.0,
            }

        elapsed = (time.time() - start) * 1000

        result = {
            "prediction": pred,
            "real_prob": probs[0, 0].item(),
            "ai_prob": probs[0, 1].item(),
            "confidence": probs.max(dim=-1).values.item(),
            "heatmaps": heatmaps,
            "frequency_band_contributions": band_contribs,
            "processing_time_ms": elapsed,
        }
        return result

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.LANCZOS)
        img_array = np.array(image, dtype=np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = ((img_array - mean) / std).astype(np.float32)

        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

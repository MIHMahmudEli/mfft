import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


BAND_CONFIGS = {
    3: [(0.0, 0.15), (0.15, 0.45), (0.45, 1.0)],
    4: [(0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)],
}

class FrequencyDecomposition(nn.Module):
    """Radial band-pass decomposition via FFT.

    soft_masks=False reproduces the pilot behavior (binary masks).
    soft_masks=True replaces the hard cutoff with a sigmoid transition of
    width `soft_tau * r_max`, removing the ringing artifacts that hard
    masks imprint on the band images.

    Masks are cached per (H, W, device) instead of being rebuilt for every
    band of every forward pass.
    """
    def __init__(self, num_bands: int = 3, soft_masks: bool = False,
                 soft_tau: float = 0.02):
        super().__init__()
        self.bands = BAND_CONFIGS.get(num_bands, BAND_CONFIGS[3])
        self.num_bands = len(self.bands)
        self.soft_masks = soft_masks
        self.soft_tau = soft_tau
        self._mask_cache = {}

    def _get_masks(self, H: int, W: int, device) -> torch.Tensor:
        key = (H, W, str(device), self.soft_masks)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached

        ny, nx = H // 2, W // 2
        r_max = min(ny, nx)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        dist = torch.sqrt((y_grid - ny).float() ** 2 + (x_grid - nx).float() ** 2)

        masks = []
        for lo, hi in self.bands:
            r_low, r_high = r_max * lo, r_max * hi
            if self.soft_masks:
                tau = max(self.soft_tau * r_max, 1e-3)
                lower = torch.sigmoid((dist - r_low) / tau) if lo > 0 \
                    else torch.ones_like(dist)
                upper = torch.sigmoid((r_high - dist) / tau) if hi < 1.0 \
                    else torch.ones_like(dist)
                masks.append(lower * upper)
            else:
                masks.append(((dist >= int(r_low)) & (dist < int(r_high))).float())
        stacked = torch.stack(masks).unsqueeze(1).unsqueeze(1)  # (B_bands,1,1,H,W)
        self._mask_cache[key] = stacked
        return stacked

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, C, H, W = x.shape
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_shifted = torch.fft.fftshift(x_fft)
        masks = self._get_masks(H, W, x.device)

        outputs = []
        for i in range(self.num_bands):
            filtered = x_shifted * masks[i]
            band = torch.fft.ifft2(torch.fft.ifftshift(filtered), norm='ortho').real
            outputs.append(band)
        return outputs


def npr_residual(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Neighboring-pixel-relation residual (after Tan et al., CVPR 2024):
    the difference between the image and its down-up resampled version,
    isolating the local interpolation traces that generator upsampling
    layers imprint. Returns a tensor shaped like x."""
    down = F.interpolate(x, scale_factor=1.0 / factor,
                         mode='bilinear', align_corners=False)
    up = F.interpolate(down, size=x.shape[2:],
                       mode='bilinear', align_corners=False)
    return x - up


class FrequencyFeatureExtractor(nn.Module):
    """
    CNN backbone for each frequency band.
    Uses depthwise separable convolutions for efficiency.
    """
    def __init__(self, in_channels: int = 3, feat_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            self._make_block(32, 64, 3, 2),
            self._make_block(64, 128, 3, 2),
            self._make_block(128, 256, 3, 2),
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def _make_block(self, cin: int, cout: int, k: int, s: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(cin, cin, k, stride=s, padding=k//2, groups=cin, bias=False),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class PretrainedFrequencyFeatureExtractor(nn.Module):
    """
    Drop-in replacement for FrequencyFeatureExtractor backed by an
    ImageNet-pretrained MobileNetV3-Small feature trunk. Used for the
    matched-pretraining MFFT variant that decouples architecture from
    initialization in comparisons against pretrained baselines.

    Band images are 3-channel (per-RGB-channel FFT filtering), so the
    pretrained stem is used unchanged.
    """
    def __init__(self, in_channels: int = 3, feat_dim: int = 256,
                 weights: str = "DEFAULT"):
        super().__init__()
        from torchvision.models import mobilenet_v3_small
        try:
            backbone = mobilenet_v3_small(weights=weights)
        except Exception as e:  # offline or weights unavailable
            import warnings
            warnings.warn(f"Pretrained weights unavailable ({e}); "
                          "falling back to random init.")
            backbone = mobilenet_v3_small(weights=None)
        if in_channels != 3:
            # inflate the stem conv: keep RGB filters, tile them (scaled)
            # over the extra channels so pretrained features are preserved
            old = backbone.features[0][0]
            new = nn.Conv2d(in_channels, old.out_channels,
                            old.kernel_size, old.stride, old.padding,
                            bias=old.bias is not None)
            with torch.no_grad():
                reps = -(-in_channels // 3)  # ceil division
                w = old.weight.repeat(1, reps, 1, 1)[:, :in_channels]
                new.weight.copy_(w * (3.0 / in_channels))
            backbone.features[0][0] = new
        self.features = backbone.features  # output: (B, 576, H/32, W/32)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion across frequency bands.
    Each band attends to all others to produce a fused representation.
    """
    def __init__(self, dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FrequencyGuidedAttention(nn.Module):
    """
    Novel module: uses frequency information to guide spatial attention.
    High-freq regions (edges, textures) get higher attention weights.
    """
    def __init__(self, dim: int = 256):
        super().__init__()
        self.freq_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, freq_weights: torch.Tensor) -> torch.Tensor:
        gate = self.freq_gate(x)
        return x * gate * freq_weights


class MFFT(nn.Module):
    """
    Multi-Frequency Fusion Transformer (MFFT)
    ===========================================
    A novel architecture for AI-generated image detection.

    Key innovations:
    1. Frequency decomposition into Low/Mid/High bands via DCT
    2. Per-band CNN feature extraction
    3. Cross-attention fusion across frequency bands
    4. Frequency-guided spatial attention
    5. Multi-scale feature aggregation

    Input:  (B, 3, H, W)  RGB image
    Output: (B, 2)         logits [real, ai_generated]

    Ablation config (dict):
        spatial_only : bool  — skip frequency decomposition, use raw image
        skip_bands   : list  — band names to exclude: ["low", "mid", "high"]
        fusion_mode  : str   — "attention" | "concat" | "avg" | "max"
        use_fga      : bool  — enable/disable FrequencyGuidedAttention
    """
    def __init__(
        self,
        in_channels: int = 3,
        feat_dim: int = 256,
        num_bands: int = 3,
        num_heads: int = 8,
        num_classes: int = 2,
        ablation: Optional[dict] = None,
        pretrained_extractors: bool = False,
        use_npr: bool = False,
        soft_masks: bool = False,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.feat_dim = feat_dim
        self.ablation = ablation or {}
        self.pretrained_extractors = pretrained_extractors
        self.use_npr = use_npr

        self.decomposer = FrequencyDecomposition(
            num_bands=num_bands, soft_masks=soft_masks)
        extractor_cls = (PretrainedFrequencyFeatureExtractor
                         if pretrained_extractors else FrequencyFeatureExtractor)
        # with NPR enabled, the highest band additionally receives the
        # 3-channel neighboring-pixel-relation residual
        self.extractors = nn.ModuleList([
            extractor_cls(
                in_channels + (3 if (use_npr and i == num_bands - 1) else 0),
                feat_dim)
            for i in range(num_bands)
        ])

        if self.ablation.get("fusion_mode", "attention") == "attention":
            self.fusion = CrossAttentionFusion(feat_dim, num_heads)
        else:
            self.fusion = None

        self.use_fga = self.ablation.get("use_fga", True)
        if self.use_fga:
            self.freq_guided_attn = FrequencyGuidedAttention(feat_dim)
        else:
            self.freq_guided_attn = None

        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim * num_bands),
            nn.Linear(feat_dim * num_bands, feat_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 2, num_classes),
        )

        # Ablation configs whose fused representation is narrower than the
        # classifier input (avg/max fusion, skipped bands, spatial-only)
        # get a learned projection instead of the old zero-padding, which
        # wasted classifier capacity on constant inputs.
        combined_dim = self._fused_dim(feat_dim, num_bands)
        if combined_dim != feat_dim * num_bands:
            self.input_proj = nn.Linear(combined_dim, feat_dim * num_bands)
        else:
            self.input_proj = nn.Identity()

        self._init_weights()

    def _fused_dim(self, feat_dim: int, num_bands: int) -> int:
        """Width of the flattened fused representation under the current
        ablation config (mirrors the branching in forward())."""
        if self.ablation.get("spatial_only", False):
            return feat_dim
        num_active = num_bands - len(self.ablation.get("skip_bands", []))
        mode = self.ablation.get("fusion_mode", "attention")
        if mode in ("avg", "max") and num_active > 1:
            return feat_dim
        return num_active * feat_dim  # attention / concat / single band

    def _init_weights(self):
        # pretrained extractor trunks must keep their ImageNet weights
        skip = set()
        if self.pretrained_extractors:
            for ext in self.extractors:
                for m in ext.features.modules():
                    skip.add(id(m))
        for m in self.modules():
            if id(m) in skip:
                continue
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, return_heatmap: bool = False):
        spatial_only = self.ablation.get("spatial_only", False)
        skip_bands = self.ablation.get("skip_bands", [])
        fusion_mode = self.ablation.get("fusion_mode", "attention")
        if self.num_bands == 4:
            band_names = ["low", "low_mid", "mid_high", "high"]
        else:
            band_names = ["low", "mid", "high"]

        if spatial_only:
            bands = [x]
            extractors = self.extractors[:1]
            extractor_inputs = bands
        else:
            all_bands = self.decomposer(x)
            keep_indices = [i for i, name in enumerate(band_names) if name not in skip_bands]
            bands = [all_bands[i] for i in keep_indices]
            extractors = [self.extractors[i] for i in keep_indices]
            extractor_inputs = []
            for i, band in zip(keep_indices, bands):
                if self.use_npr and i == self.num_bands - 1:
                    band = torch.cat([band, npr_residual(x)], dim=1)
                extractor_inputs.append(band)

        features = []
        for band, extractor in zip(extractor_inputs, extractors):
            feat = extractor(band)
            features.append(feat)

        B = x.shape[0]
        num_active = len(features)

        if num_active == 1 or fusion_mode == "concat":
            fused = torch.cat(features, dim=-1)
            N = num_active
            D = fused.shape[-1] // N
            fused = fused.unsqueeze(1)
        elif fusion_mode == "avg":
            fused = torch.stack(features, dim=0).mean(dim=0)
            N = 1
            D = fused.shape[-1]
            fused = fused.unsqueeze(1)
        elif fusion_mode == "max":
            fused = torch.stack(features, dim=0).max(dim=0).values
            N = 1
            D = fused.shape[-1]
            fused = fused.unsqueeze(1)
        elif fusion_mode == "attention" and self.fusion is not None and num_active > 1:
            feat_stack = torch.stack(features, dim=1)
            fused = self.fusion(feat_stack)
            N = fused.shape[1]
            D = fused.shape[2]
        else:
            fused = torch.cat(features, dim=-1)
            N = num_active
            D = fused.shape[-1] // N
            fused = fused.unsqueeze(1)

        # FGA operates on per-band tokens, so it only applies when the fused
        # representation kept one token per band (attention path). Pooled
        # representations (avg/max/concat) have no band axis to weight.
        fga_applicable = (
            self.use_fga and self.freq_guided_attn is not None
            and not spatial_only
            and N == num_active and D == fused.shape[-1]
            and fused.shape[-1] == self.feat_dim
        )
        if fga_applicable:
            all_bands_for_fga = bands
            freq_magnitudes = torch.stack([
                torch.abs(band).mean(dim=(1, 2, 3))
                for band in all_bands_for_fga
            ], dim=1)
            freq_weights = F.softmax(freq_magnitudes, dim=1).unsqueeze(-1)
            guided = self.freq_guided_attn(fused, freq_weights)
        else:
            guided = fused

        combined = guided.reshape(B, N * D)
        combined = self.input_proj(combined)
        logits = self.classifier(combined)

        if return_heatmap:
            heatmaps = []
            for band in bands:
                h = torch.abs(band).mean(dim=1, keepdim=True)
                h = F.interpolate(h, size=x.shape[2:], mode='bilinear', align_corners=False)
                heatmaps.append(h)
            return logits, torch.cat(heatmaps, dim=1)
        return logits


class MFFTWithExplainability(MFFT):
    """
    Extension of MFFT that also produces:
    - Confidence score (0-1)
    - Per-region anomaly heatmap
    - Per-frequency-band contribution scores
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x / 255.0 - self.mean) / self.std
        return x

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        x = self.preprocess(x)
        self.eval()
        logits, heatmaps = self.forward(x, return_heatmap=True)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        return {
            "prediction": preds,
            "real_prob": probs[:, 0],
            "ai_prob": probs[:, 1],
            "confidence": probs.max(dim=-1).values,
            "heatmaps": heatmaps,
            "logits": logits,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_mfft(variant: str = "base", ablation: Optional[dict] = None,
               pretrained_extractors: bool = False,
               use_npr: bool = False, soft_masks: bool = False) -> MFFT:
    configs = {
        "tiny":  {"feat_dim": 128, "num_heads": 4, "num_bands": 3},
        "base":  {"feat_dim": 384, "num_heads": 6, "num_bands": 3},
        "large": {"feat_dim": 768, "num_heads": 12, "num_bands": 4},
    }
    cfg = configs.get(variant, configs["base"])
    model = MFFT(**cfg, ablation=ablation,
                 pretrained_extractors=pretrained_extractors,
                 use_npr=use_npr, soft_masks=soft_masks)
    return model

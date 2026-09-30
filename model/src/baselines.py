"""
Standard baseline models for ablation comparison.
Includes custom (SimpleCNN, LightViT) and torchvision models (ResNet, EfficientNet, ViT, Swin).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M


def _resize_pos_embed(pos_embed, new_seq_len):
    """Interpolate 2D positional embeddings to a different grid size."""
    old_seq_len = pos_embed.shape[1]
    if old_seq_len == new_seq_len:
        return pos_embed
    dim = pos_embed.shape[-1]
    cls_token = pos_embed[:, 0:1, :]
    pos_tokens = pos_embed[:, 1:, :]
    old_h = old_w = int(math.isqrt(old_seq_len - 1))
    new_h = new_w = int(math.isqrt(new_seq_len - 1))
    pos_tokens = pos_tokens.reshape(1, old_h, old_w, dim).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(pos_tokens, size=(new_h, new_w), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_h * new_w, dim)
    return torch.cat([cls_token, pos_tokens], dim=1)


class SimpleCNN(nn.Module):
    """Lightweight CNN baseline for AI image detection."""
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        if x.shape[1] != self.pos_embed.shape[1]:
            pos_embed = _resize_pos_embed(self.pos_embed, x.shape[1])
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class LightViT(nn.Module):
    """Lightweight Vision Transformer baseline."""
    def __init__(self, img_size=384, patch_size=16, in_chans=3,
                 embed_dim=256, depth=6, num_heads=8, num_classes=2):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def _wrap_torchvision(model_fn, weights_cls, num_classes=2):
    model = model_fn(weights=weights_cls)
    in_features = model.classifier[1].in_features if hasattr(model, 'classifier') else \
                  model.fc.in_features if hasattr(model, 'fc') else \
                  model.head.in_features
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        model.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(in_features, num_classes))
    elif hasattr(model, 'fc'):
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'head'):
        model.head = nn.Linear(in_features, num_classes)
    return model


def resnet18(num_classes=2):
    return _wrap_torchvision(M.resnet18, M.ResNet18_Weights.IMAGENET1K_V1, num_classes)


def resnet50(num_classes=2):
    return _wrap_torchvision(M.resnet50, M.ResNet50_Weights.IMAGENET1K_V2, num_classes)


def efficientnet_b0(num_classes=2):
    return _wrap_torchvision(M.efficientnet_b0, M.EfficientNet_B0_Weights.IMAGENET1K_V1, num_classes)


def vit_b_16(img_size=384, num_classes=2):
    model = M.vit_b_16(weights=M.ViT_B_16_Weights.IMAGENET1K_V1)
    model.image_size = img_size
    if hasattr(model, 'heads'):
        in_feat = model.heads.head.in_features
        model.heads = nn.Linear(in_feat, num_classes)
    else:
        in_feat = model.head.in_features
        model.head = nn.Linear(in_feat, num_classes)
    patch_size = model.patch_size
    n_patches = (img_size // patch_size) ** 2
    with torch.no_grad():
        model.encoder.pos_embedding = nn.Parameter(
            _resize_pos_embed(model.encoder.pos_embedding, n_patches + 1))
    return model


def swin_t(num_classes=2):
    model = M.swin_t(weights=M.Swin_T_Weights.IMAGENET1K_V1)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


class CLIPBaseline(nn.Module):
    """CLIP ViT-B/32 with a classification head for fine-tuning."""
    def __init__(self, img_size=384, num_classes=2):
        super().__init__()
        import open_clip
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_model = self.clip_model.visual
        in_features = self.clip_model.output_dim
        self.clip_model.output_dim = None
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, num_classes),
        )
        patch_size = 32
        n_patches = (img_size // patch_size) ** 2
        pe = self.clip_model.positional_embedding
        with torch.no_grad():
            self.clip_model.positional_embedding = nn.Parameter(
                _resize_pos_embed(pe.unsqueeze(0), n_patches + 1).squeeze(0))

    def forward(self, x):
        features = self.clip_model(x)
        return self.head(features)


class FreqDetect(nn.Module):
    """
    Frequency-domain baseline (Frank et al. 2020).
    Computes radial FFT magnitude profiles and classifies via MLP.
    """
    def __init__(self, img_size=384, num_radial_bins=16, num_classes=2):
        super().__init__()
        self.img_size = img_size
        self.num_radial_bins = num_radial_bins
        ny, nx = img_size // 2, img_size // 2
        y_grid, x_grid = torch.meshgrid(
            torch.arange(img_size), torch.arange(img_size), indexing='ij')
        dist = torch.sqrt((y_grid - ny)**2 + (x_grid - nx)**2).float()
        bin_width = ny / num_radial_bins
        self.register_buffer('radial_mask', (dist / bin_width).long().clamp(0, num_radial_bins - 1))

        self.classifier = nn.Sequential(
            nn.Linear(num_radial_bins * 3, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        B = x.shape[0]
        fft = torch.fft.fft2(x, norm='ortho')
        shifted = torch.fft.fftshift(fft)
        mag = torch.abs(shifted)
        radial_feats = []
        for c in range(x.shape[1]):
            channel = mag[:, c]
            pooled = torch.zeros(B, self.num_radial_bins, device=x.device, dtype=channel.dtype)
            pooled.scatter_add_(1, self.radial_mask.unsqueeze(0).expand(B, -1, -1).reshape(B, -1),
                                channel.reshape(B, -1))
            counts = (self.radial_mask.unsqueeze(0).expand(B, -1, -1).reshape(B, -1) >= 0).float().sum(dim=1, keepdim=True)
            pooled = pooled / counts.clamp(min=1)
            radial_feats.append(pooled)
        feats = torch.cat(radial_feats, dim=-1)
        return self.classifier(feats)


class DeiTSmall(nn.Module):
    """
    DeiT-Small architecture (Touvron et al. 2021).
    Uses class + distillation tokens, 12 transformer blocks, embed_dim=384.
    """
    def __init__(self, img_size=384, patch_size=16, in_chans=3,
                 embed_dim=384, depth=12, num_heads=6, num_classes=2, dropout=0.1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        if x.shape[1] != self.pos_embed.shape[1]:
            pos_embed = _resize_pos_embed(self.pos_embed, x.shape[1])
        else:
            pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return (self.head(x[:, 0]) + self.head_dist(x[:, 1])) / 2


def deit_small(img_size=384, num_classes=2):
    """Build DeiT-Small."""
    return DeiTSmall(img_size=img_size, num_classes=num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    models = {
        'SimpleCNN': SimpleCNN(),
        'LightViT': LightViT(),
        'ResNet18': resnet18(),
        'ResNet50': resnet50(),
        'EfficientNet-B0': efficientnet_b0(),
        'ViT-B/16': vit_b_16(),
        'Swin-T': swin_t(),
        'CLIP': CLIPBaseline(img_size=384),
    }
    x = torch.randn(2, 3, 384, 384)
    for name, model in models.items():
        print(f'{name:20s} params={count_parameters(model):>10,}  out={model(x).shape}')

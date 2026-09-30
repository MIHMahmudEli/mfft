"""
Train baseline models (SimpleCNN, LightViT) for ablation comparison.
Run: python -m src.train_baselines
"""
import os, sys, json, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'model'))
os.chdir(PROJECT_ROOT)

from src.dataset import AIDetectionDataset, ImageTransform
from src.config import Config
from src.baselines import (
    SimpleCNN, LightViT, count_parameters,
    resnet18, resnet50, efficientnet_b0, vit_b_16, swin_t,
    CLIPBaseline,
)

cfg = Config()
cfg.dataset.undersample = True
cfg.dataset.val_split = 0.15
cfg.dataset.test_split = 0.10

full_dataset = AIDetectionDataset(
    root_dir=cfg.dataset.root_dir,
    metadata_paths=cfg.dataset.metadata_paths,
    transform=None,
    is_train=True,
    size=cfg.training.image_size,
    undersample=False,
)
labels = [s[1] for s in full_dataset.samples]
indices = list(range(len(full_dataset)))

train_idx, temp_idx = train_test_split(
    indices, test_size=cfg.dataset.val_split + cfg.dataset.test_split,
    stratify=labels, random_state=42)
temp_labels = [labels[i] for i in temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx, test_size=cfg.dataset.test_split / (cfg.dataset.val_split + cfg.dataset.test_split),
    stratify=temp_labels, random_state=42)

train_dataset = AIDetectionDataset(
    root_dir=cfg.dataset.root_dir, metadata_paths=[],
    transform=ImageTransform(size=cfg.training.image_size, augment=True),
    is_train=True, size=cfg.training.image_size, undersample=False)
train_dataset.samples = [full_dataset.samples[i] for i in train_idx]
if cfg.dataset.undersample:
    train_dataset._undersample()

val_dataset = AIDetectionDataset(
    root_dir=cfg.dataset.root_dir, metadata_paths=[],
    transform=ImageTransform(size=cfg.training.image_size, augment=False),
    is_train=False, size=cfg.training.image_size, undersample=False)
val_dataset.samples = [full_dataset.samples[i] for i in val_idx]

test_dataset = AIDetectionDataset(
    root_dir=cfg.dataset.root_dir, metadata_paths=[],
    transform=ImageTransform(size=cfg.training.image_size, augment=False),
    is_train=False, size=cfg.training.image_size, undersample=False)
test_dataset.samples = [full_dataset.samples[i] for i in test_idx]

train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0, pin_memory=False)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
NUM_EPOCHS = 10

baselines = {
    'SimpleCNN': lambda: SimpleCNN(),
    'LightViT': lambda: LightViT(depth=4, num_heads=4, embed_dim=192),
    'ResNet-18': lambda: resnet18(),
    'ResNet-50': lambda: resnet50(),
    'EfficientNet-B0': lambda: efficientnet_b0(),
    'ViT-B/16': lambda: vit_b_16(img_size=cfg.training.image_size),
    'Swin-T': lambda: swin_t(),
    'CLIP': lambda: CLIPBaseline(),
}
results = {}

for name, model_cls in baselines.items():
    print(f'\n{"="*60}\nTraining {name}...\n{"="*60}')

    model = model_cls().to(device)
    n_params = count_parameters(model)
    print(f'Parameters: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_acc = 0
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        train_acc = correct / total * 100
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total * 100
        print(f'  Epoch {epoch+1}: train={train_acc:.2f}%, val={val_acc:.2f}%')
        if val_acc > best_acc:
            best_acc = val_acc

    elapsed = time.time() - start_time
    test_correct = test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=-1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    test_acc = test_correct / test_total * 100

    results[name] = {
        'params': n_params,
        'val_acc': round(best_acc, 2),
        'test_acc': round(test_acc, 2),
        'training_time_min': round(elapsed / 60, 1),
    }
    print(f'  Best val: {best_acc:.2f}%, Test: {test_acc:.2f}%, Time: {elapsed/60:.1f} min')

out_path = PROJECT_ROOT / 'paper' / 'tables' / 'ablation_results.json'
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to {out_path}')
print(json.dumps(results, indent=2))

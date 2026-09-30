import io
import json
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import pandas as pd
from PIL import Image
import random
import numpy as np
import warnings
from typing import Optional, Callable, Dict, List, Tuple
from torchvision import transforms


class RandomJPEGCompression:
    """Re-encode the image as JPEG at a random quality level.

    Detectors that never see compression during training collapse on
    real-world (recompressed) images, so this augmentation is applied
    before tensor conversion. Operates on PIL images.
    """
    def __init__(self, quality_range: Tuple[int, int] = (30, 95), p: float = 0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        quality = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class ImageTransform:
    def __init__(self, size: int = 384, augment: bool = True,
                 jpeg_aug: bool = True, blur_aug: bool = True):
        self.size = size
        self.augment = augment

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if augment:
            aug_ops = [
                transforms.Resize(size + 16),
                transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10, fill=128),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
            ]
            if jpeg_aug:
                aug_ops.append(RandomJPEGCompression(quality_range=(30, 95), p=0.5))
            if blur_aug:
                aug_ops.append(transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2))
            aug_ops += [
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
                self.normalize,
            ]
            self.transform = transforms.Compose(aug_ops)
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size + 16),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                self.normalize,
            ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img.convert("RGB"))


def map_kaggle_path(source: str, fn: str) -> str:
    """Map a manifest filename and source to its mounted path under /kaggle/input/."""
    fn = fn.replace('\\', '/')
    if '/kaggle/input/' in fn:
        idx = fn.find('/kaggle/input/')
        return fn[idx:]
    if fn.startswith('kaggle/input/'):
        return '/' + fn
    fn = fn.strip('/')
    if source == 'places365':
        p = fn[len('Places365/'):] if fn.startswith('Places365/') else fn
        return f'/kaggle/input/places365/{p}'
    elif source == 'faceforensics':
        p = fn[len('FaceForensics/'):] if fn.startswith('FaceForensics/') else fn
        return f'/kaggle/input/faceforensics/{p}'
    elif source in ('dfdc', 'dfdc_real'):
        p = fn
        for prefix in ['DFDC/Dataset/', 'DFDC/']:
            if p.startswith(prefix):
                p = p[len(prefix):]
                break
        return f'/kaggle/input/dfdc-faces-of-the-train-sample/train/{p}' if not p.startswith('train/') else f'/kaggle/input/dfdc-faces-of-the-train-sample/{p}'
    elif source == 'pexels_unsplash':
        p = fn[len('real/'):] if fn.startswith('real/') else fn
        return f'/kaggle/input/mfft-real/Mfft_real/{p}'
    elif source == 'stable_diffusion':
        return f'/kaggle/input/stable-diffusion/{fn}'
    elif source == 'open_images_v7':
        return f'/kaggle/input/open-images-v7-dataset/{fn}'
    elif source == 'ntire2026':
        return f'/kaggle/input/ntire2026/{fn}'
    elif source == 'midjourney':
        return f'/kaggle/input/midjourney/{fn}'
    elif source == 'glide':
        return f'/kaggle/input/genimage-ai/{fn}'
    elif source == 'imagenet':
        p = fn if fn.startswith('genimage_ai/') else f'genimage_ai/{fn}'
        return f'/kaggle/input/genimage-ai/{p}'
    return f'/kaggle/input/{source}/{fn}'


class AIDetectionDataset(Dataset):
    """
    Dataset for AI-generated image detection.
    Supports multiple metadata formats (v1, v2, v3).

    Labels:
        0 = real
        1 = ai_generated
    """
    def __init__(
        self,
        root_dir: str,
        metadata_paths: List[str],
        transform: Optional[Callable] = None,
        is_train: bool = True,
        size: int = 384,
        undersample: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform or ImageTransform(size=size, augment=is_train)
        self.samples = []
        self.metadata_paths = metadata_paths
        self._corrupted_files = []
        self._skipped_zero_byte = 0
        self.size = size

        self._load_all_metadata()

        if self._skipped_zero_byte > 0:
            print(f"  Warning: skipped {self._skipped_zero_byte} zero-byte/corrupted files")
        if self._corrupted_files:
            print(f"  Warning: {len(self._corrupted_files)} files previously flagged as corrupted")
            for f in self._corrupted_files[:5]:
                print(f"    - {f}")
            if len(self._corrupted_files) > 5:
                print(f"    ... and {len(self._corrupted_files) - 5} more")

        if undersample and is_train:
            self._undersample()

        print(f"Dataset loaded: {len(self.samples)} samples")
        self._print_stats()

    def _load_all_metadata(self):
        for mp in self.metadata_paths:
            p = Path(mp)
            if not p.exists():
                print(f"  Warning: {p} not found, skipping")
                continue

            # ── Try CSV first (local/DGX format with filename column) ──
            df = None
            is_json_manifest = False
            try:
                df = pd.read_csv(p, low_memory=False, dtype={'generator': str, 'md5': str})
                if 'filename' not in df.columns:
                    df = None  # CSV without filename column — try JSON
            except Exception:
                pass

            # ── Try JSON (rebuild_manifest.py format) ──
            manifest_data = None
            if df is None:
                try:
                    with open(p) as f:
                        manifest_data = json.load(f)
                    if "images" not in manifest_data:
                        manifest_data = None  # Not the expected format
                except Exception:
                    pass

            if manifest_data is not None:
                # New JSON manifest format from rebuild_manifest.py
                # Has: images = {image_id: {shard, label, label_int, split}, ...}
                df, image_dirs = self._load_kaggle_manifest(manifest_data)
                is_json_manifest = True
            elif df is not None:
                # Fast Kaggle mapping — avoids scanning millions of files over FUSE
                if Path("/kaggle/input").exists():
                    print(f"  [Kaggle] Resolving {len(df):,} images from mounted inputs...")
                    LABEL_MAP = {'real': 0, 'ai_generated': 1, 'deepfake': 2}
                    filenames = df['filename'].tolist()
                    labels = df['label'].tolist()
                    sources = df['source'].tolist() if 'source' in df.columns else [''] * len(df)
                    for fn, lbl, src in zip(filenames, labels, sources):
                        img_path = map_kaggle_path(str(src), str(fn))
                        label_int = LABEL_MAP.get(str(lbl).lower(), 1)
                        self.samples.append((img_path, label_int))
                    continue
                image_dirs = self._resolve_image_dirs(p, df)
            else:
                print(f"  Warning: cannot read {p}")
                continue

            if df is None or (not is_json_manifest and not image_dirs):
                continue

            for _, row in df.iterrows():
                if is_json_manifest:
                    img_path = Path(row['filename']) if pd.notna(row.get('filename')) else None
                else:
                    img_path = self._resolve_image_path(row, image_dirs)
                if img_path and img_path.exists():
                    size = img_path.stat().st_size
                    if size == 0:
                        self._skipped_zero_byte += 1
                        continue
                    label = self._get_label(row)
                    if label is not None:
                        self.samples.append((str(img_path), label))

    def _load_kaggle_manifest(self, manifest_data: dict):
        """
        Load images from a JSON manifest (rebuild_manifest.py format) by
        scanning Kaggle input mounts. Returns (DataFrame, image_dirs).
        Handles both regular datasets (directory scan) and artifact datasets
        (metadata.csv per generator).
        """
        import os, hashlib
        images_dict = manifest_data.get("images", {})
        if not images_dict:
            return None, []

        # Build a lookup: (shard, label) → {img_id: info}
        lookup = {}
        for img_id, info in images_dict.items():
            shard = info.get("shard", "")
            label = info.get("label", "")
            lookup.setdefault((shard, label), {})[img_id] = info

        # Scan Kaggle input mounts
        input_root = Path("/kaggle/input")
        resolved = []  # (path, label, label_int, shard)

        if input_root.exists():
            slug_map: Dict[str, Path] = {}
            for base in [input_root, input_root / "datasets"]:
                if not base.exists():
                    continue
                for owner in base.iterdir():
                    if not owner.is_dir():
                        continue
                    for child in owner.iterdir():
                        if child.is_dir():
                            slug_map[child.name] = child

            print(f"  Kaggle mounts: {list(slug_map.keys())}")

            img_exts = {'.jpg', '.jpeg', '.png', '.webp'}
            shards_needed = set(info.get("shard", "") for info in images_dict.values())

            for shard in sorted(shards_needed):
                mount = slug_map.get(shard)
                if mount is None:
                    for slug, path in slug_map.items():
                        if shard.replace("-", "") in slug.replace("-", "") or slug.replace("-", "") in shard.replace("-", ""):
                            mount = path
                            break
                if mount is None:
                    print(f"  Warning: shard '{shard}' not in mounts, skipping")
                    continue

                shard_entries = {}
                for (s, lbl), id_map in lookup.items():
                    if s == shard:
                        shard_entries[lbl] = id_map

                # Artifact sub-datasets: use metadata.csv
                if shard.startswith("artifact-"):
                    count = self._load_artifact_metadata(mount, shard, shard_entries, resolved)
                else:
                    # Regular dataset: scan directories
                    count = 0
                    print(f"  Scanning {shard}...", end=" ", flush=True)
                    for dirpath, _, filenames in os.walk(str(mount)):
                        for fname in filenames:
                            if Path(fname).suffix.lower() not in img_exts:
                                continue
                            full_path = os.path.join(dirpath, fname)
                            if os.path.getsize(full_path) == 0:
                                continue
                            for lbl, id_map in shard_entries.items():
                                img_id = hashlib.sha1(f"{fname}|{shard}|{lbl}".encode()).hexdigest()[:20]
                                if img_id in id_map:
                                    info = id_map[img_id]
                                    resolved.append((full_path, lbl, info.get("label_int", 0), shard))
                                    count += 1
                                    break
                    print(f"{count} images")

        if not resolved:
            print(f"  Warning: No images found in Kaggle mounts")
            return None, []

        df = pd.DataFrame(resolved, columns=['filename', 'label', 'label_int', 'source'])
        df['generator'] = ''
        df['split'] = 'train'
        print(f"  Loaded {len(df)} images from Kaggle mounts")
        return df, []

    def _load_artifact_metadata(self, mount_dir: Path, shard: str, shard_entries: dict, resolved: list) -> int:
        """Load images from artifact sub-dataset using its metadata.csv."""
        import pandas as pd
        csv_path = mount_dir / "metadata.csv"
        if not csv_path.exists():
            for child in mount_dir.iterdir():
                if child.is_dir() and (child / "metadata.csv").exists():
                    csv_path = child / "metadata.csv"
                    break
        if not csv_path.exists():
            print(f"  Warning: no metadata.csv for {shard}")
            return 0

        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            print(f"  Warning: cannot read {csv_path}: {e}")
            return 0

        count = 0
        for _, r in df.iterrows():
            target = int(r.get("target", 0))
            if target == 0:
                continue  # Skip real images
            img_path = str(r.get("image_path", r.get("filename", "")))
            if not img_path:
                continue
            if not os.path.isabs(img_path):
                img_path = str(mount_dir / img_path)
            if not os.path.exists(img_path):
                continue
            if os.path.getsize(img_path) == 0:
                continue
            fname = os.path.basename(img_path)
            for lbl, id_map in shard_entries.items():
                img_id = hashlib.sha1(f"{fname}|{shard}|{lbl}".encode()).hexdigest()[:20]
                if img_id in id_map:
                    info = id_map[img_id]
                    resolved.append((img_path, lbl, info.get("label_int", 0), shard))
                    count += 1
                    break
        print(f"  Scanning {shard}... {count} images")
        return count

    def _resolve_image_dirs(self, metadata_path: Path, df: pd.DataFrame) -> List[Path]:
        # find the nearest ancestor that has an images/ sibling, so manifests
        # in subfolders (e.g. dataset/metadata/logo/) resolve too
        base = metadata_path.parent.parent / "images"
        if not base.exists():
            for parent in metadata_path.resolve().parents:
                if (parent / "images").exists():
                    base = parent / "images"
                    break
        candidates = [
            base,  # new layout: filenames are relative paths from images root
            base / "real",
            base / "ai_generated",
            base / "ai_altered",
        ]
        existing = [c for c in candidates if c.exists()]
        if existing:
            return existing

        if 'filename' in df.columns:
            first_file = str(df['filename'].iloc[0])
            valid = [c for c in candidates if (c / first_file).exists()]
            if valid:
                return valid

        print(f"  Warning: cannot resolve image dir for {metadata_path}")
        return []

    def _resolve_image_path(self, row: pd.Series, image_dirs: List[Path]) -> Optional[Path]:
        if 'filename' in row and pd.notna(row['filename']):
            # manifests store Windows-style separators; normalize so the
            # same manifest resolves on Linux (DGX) too
            fn = str(row['filename']).replace('\\', '/')
            if '/' in fn or '\\' in fn:
                p = image_dirs[0] / fn
                if p.exists():
                    return p
            for image_dir in image_dirs:
                p = image_dir / fn
                if p.exists():
                    return p
            p = Path(fn)
            if p.exists():
                return p

        if 'image_id' in row and pd.notna(row['image_id']):
            for image_dir in image_dirs:
                for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    p = image_dir / f"{row['image_id']}{ext}"
                    if p.exists():
                        return p

        return None

    def _get_label(self, row: pd.Series) -> Optional[int]:
        if 'image_type' in row:
            val = str(row['image_type']).lower()
            if val in ('real', 'real_image'):
                return 0
            if val in ('ai_generated', 'ai', 'ai-generated', 'ai_generation'):
                return 1
            if val in ('ai_altered', 'altered', 'deepfake'):
                return 1

        if 'label' in row:
            val = str(row['label']).lower()
            if val in ('0', 'real'):
                return 0
            if val in ('1', 'ai', 'fake', 'ai_generated', 'deepfake'):
                return 1

        return None

    def _undersample(self):
        labels = np.array([s[1] for s in self.samples])
        n_real = (labels == 0).sum()
        n_ai = (labels == 1).sum()
        target = min(n_real, n_ai)

        real_indices = [i for i, l in enumerate(labels) if l == 0]
        ai_indices = [i for i, l in enumerate(labels) if l == 1]

        if len(real_indices) > target:
            real_indices = random.sample(real_indices, target)
        if len(ai_indices) > target:
            ai_indices = random.sample(ai_indices, target)

        kept = set(real_indices + ai_indices)
        self.samples = [s for i, s in enumerate(self.samples) if i in kept]

    def _print_stats(self):
        labels = [s[1] for s in self.samples]
        n_real = labels.count(0)
        n_ai = labels.count(1)
        print(f"  Real: {n_real}, AI: {n_ai}, Total: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
            return tensor, label
        except Exception as e:
            rel = Path(path).relative_to(self.root_dir) if path.startswith(str(self.root_dir)) else path
            warnings.warn(f"Corrupted image #{idx}: {rel} ({e})")

            # Retry with the next valid sample
            for offset in range(1, min(100, len(self.samples))):
                retry_idx = (idx + offset) % len(self.samples)
                retry_path, retry_label = self.samples[retry_idx]
                try:
                    img = Image.open(retry_path).convert("RGB")
                    tensor = self.transform(img)
                    if offset not in self._corrupted_files:
                        self._corrupted_files.append(retry_path)
                    return tensor, retry_label
                except Exception:
                    continue

            blank = torch.zeros(3, self.size, self.size)
            return blank, label


def create_dataloaders(
    root_dir: str,
    metadata_paths: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    size: int = 384,
    val_split: float = 0.15,
    undersample: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates stratified train/val dataloaders from metadata files.
    """
    full_dataset = AIDetectionDataset(
        root_dir=root_dir,
        metadata_paths=metadata_paths,
        transform=None,
        is_train=True,
        size=size,
        undersample=False,
    )

    labels = [s[1] for s in full_dataset.samples]

    indices = list(range(len(full_dataset)))
    from sklearn.model_selection import train_test_split

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=labels,
        random_state=42,
    )

    train_dataset = AIDetectionDataset(
        root_dir=root_dir,
        metadata_paths=[],  # don't reload
        transform=ImageTransform(size=size, augment=True),
        is_train=True,
        size=size,
        undersample=False,
    )
    train_dataset.samples = [full_dataset.samples[i] for i in train_idx]
    if undersample:
        train_dataset._undersample()

    val_dataset = AIDetectionDataset(
        root_dir=root_dir,
        metadata_paths=[],
        transform=ImageTransform(size=size, augment=False),
        is_train=False,
        size=size,
        undersample=False,
    )
    val_dataset.samples = [full_dataset.samples[i] for i in val_idx]

    print(f"\nTrain: {len(train_dataset)} samples")
    print(f"Val:   {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_weighted_sampler(dataset: AIDetectionDataset) -> WeightedRandomSampler:
    labels = [s[1] for s in dataset.samples]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    sample_weights = [weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def create_split_dataloaders(
    root_dir: str,
    metadata_paths: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    size: int = 384,
    val_split: float = 0.10,
    test_split: float = 0.10,
    seed: int = 42,
    use_weighted_sampler: bool = True,
    split_index_path: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Stratified train/val/test dataloaders with persisted split indices.

    If `split_index_path` exists, the saved indices are reused verbatim so
    every model (MFFT variants, baselines, ablations) evaluates on the
    identical test set. Otherwise the split is created, then saved there.

    Class imbalance is handled with a WeightedRandomSampler on the train
    loader (keeps all real images, oversamples the minority class) instead
    of undersampling, unless `use_weighted_sampler=False`.

    `max_samples` draws a seeded, class-balanced subset before splitting
    (used by the notebooks' smoke-verification mode).
    """
    from sklearn.model_selection import train_test_split

    full_dataset = AIDetectionDataset(
        root_dir=root_dir,
        metadata_paths=metadata_paths,
        transform=None,
        is_train=True,
        size=size,
        undersample=False,
    )

    if max_samples is not None and len(full_dataset.samples) > max_samples:
        rng = random.Random(seed)
        by_class: Dict[int, list] = {0: [], 1: []}
        for s in full_dataset.samples:
            by_class[s[1]].append(s)
        per_class = max_samples // 2
        subset = []
        for lbl, items in by_class.items():
            rng.shuffle(items)
            subset.extend(items[:per_class])
        rng.shuffle(subset)
        full_dataset.samples = subset
        print(f"max_samples: reduced to {len(subset)} balanced samples")

    if len(full_dataset.samples) == 0:
        raise RuntimeError(
            f"No samples resolved from {metadata_paths}. Check that the "
            "manifest's filenames exist under the dataset images directory."
        )

    labels = [s[1] for s in full_dataset.samples]
    indices = list(range(len(full_dataset)))

    if split_index_path and Path(split_index_path).exists():
        with open(split_index_path) as f:
            saved = json.load(f)
        if saved.get("n_samples") != len(full_dataset):
            raise ValueError(
                f"Saved split at {split_index_path} was built for "
                f"{saved.get('n_samples')} samples but dataset has "
                f"{len(full_dataset)}. Delete the file to regenerate."
            )
        train_idx = saved["train"]
        val_idx = saved["val"]
        test_idx = saved["test"]
        print(f"Reusing saved split from {split_index_path}")
    else:
        holdout = val_split + test_split
        train_idx, rest_idx = train_test_split(
            indices, test_size=holdout, stratify=labels, random_state=seed,
        )
        rest_labels = [labels[i] for i in rest_idx]
        val_idx, test_idx = train_test_split(
            rest_idx,
            test_size=test_split / holdout,
            stratify=rest_labels,
            random_state=seed,
        )
        if split_index_path:
            Path(split_index_path).parent.mkdir(parents=True, exist_ok=True)
            with open(split_index_path, "w") as f:
                json.dump({
                    "seed": seed,
                    "n_samples": len(full_dataset),
                    "val_split": val_split,
                    "test_split": test_split,
                    "train": list(map(int, train_idx)),
                    "val": list(map(int, val_idx)),
                    "test": list(map(int, test_idx)),
                }, f)
            print(f"Saved split indices to {split_index_path}")

    def _subset(idx_list, augment):
        ds = AIDetectionDataset(
            root_dir=root_dir, metadata_paths=[],
            transform=ImageTransform(size=size, augment=augment),
            is_train=augment, size=size, undersample=False,
        )
        ds.samples = [full_dataset.samples[i] for i in idx_list]
        return ds

    train_dataset = _subset(train_idx, augment=True)
    val_dataset = _subset(val_idx, augment=False)
    test_dataset = _subset(test_idx, augment=False)

    print(f"\nTrain: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    sampler = get_weighted_sampler(train_dataset) if use_weighted_sampler else None
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader

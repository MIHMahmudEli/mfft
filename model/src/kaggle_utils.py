"""
Kaggle Utilities
================
Shared helpers for all Kaggle notebooks:
  - Environment detection (Kaggle vs local vs DGX)
  - Dataset path resolution across mounted inputs
  - HuggingFace token discovery + upload (checkpoints, results, models)
  - Full-state checkpoint save/resume (survives session crashes)
  - Working directory setup

Usage in any notebook cell:
    from src.kaggle_utils import KaggleEnv
    env = KaggleEnv(project_root_search=True)
    # env.is_kaggle, env.working_dir, env.figures_dir, etc.
"""

import os
import sys
import json
import hashlib
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple, List, Any


class KaggleEnv:
    """
    Detects the runtime environment and resolves all paths.

    Attributes:
        is_kaggle (bool): True if running on Kaggle
        is_gpu (bool): True if CUDA available
        smoke_test (bool): True if no GPU (quick verification mode)
        project_root (Path): Root of the ai-image-detection-research repo
        working_dir (Path): Kaggle working dir or project root
        images_dir (Path): Resolved path to dataset/images
        checkpoints_dir (Path): model/checkpoints/
        results_dir (Path): paper/result/<mode>/
        figures_dir (Path): paper/result/<mode>/fig/
        tables_dir (Path): paper/result/<mode>/table/
        hf_token (str): HuggingFace token or empty
        hf_checkpoint_repo (str): HF repo for checkpoints
        hf_results_repo (str): HF repo for results/paper outputs
        hf_model_repo (str): HF repo for deployment-ready models
    """

    def __init__(
        self,
        project_root_search: bool = True,
        variant: str = "base",
        mode: str = "full_scale",
    ):
        import torch

        # ── Environment detection ──
        print("  [1/5] Detecting environment...")
        self.is_kaggle = Path("/kaggle/input").exists()
        self.is_gpu = torch.cuda.is_available()
        self.smoke_test = not self.is_gpu

        # ── Project root ──
        print("  [2/5] Finding project root...")
        if project_root_search:
            self.project_root = self._find_project_root()
        else:
            self.project_root = Path.cwd().resolve()

        # ── Working directory ──
        if self.is_kaggle:
            self.working_dir = Path("/kaggle/working")
        else:
            self.working_dir = self.project_root

        os.chdir(self.working_dir)
        if str(self.project_root / "model") not in sys.path:
            sys.path.insert(0, str(self.project_root / "model"))

        # ── Image data resolution ──
        print("  [3/5] Resolving image paths...")
        self.images_dir = self._resolve_images_dir()

        # ── Checkpoints, results, figures ──
        print("  [4/5] Setting up output directories...")
        self.checkpoints_dir = self.project_root / "model" / "checkpoints"
        mode_dir = "verify" if self.smoke_test else mode
        self.results_dir = self.project_root / "paper" / "result" / mode_dir
        self.figures_dir = self.results_dir / "fig"
        self.tables_dir = self.results_dir / "table"

        # ── HuggingFace config ──
        print("  [5/5] Loading HuggingFace config...")
        self.hf_token = self._find_hf_token()
        self.hf_checkpoint_repo = os.environ.get(
            "HF_CHECKPOINT_REPO", "MohsinElis/mfft-checkpoints"
        )
        self.hf_results_repo = os.environ.get(
            "HF_RESULTS_REPO", "MohsinElis/mfft-results"
        )
        self.hf_model_repo = os.environ.get(
            "HF_MODEL_REPO", "MohsinElis/mfft-model"
        )
        self.hf_manifest_repo = os.environ.get(
            "HF_MANIFEST_REPO", "MohsinElis/mfft-master-manifest"
        )

        # ── Summary ──
        print(f"KaggleEnv initialized:")
        print(f"  is_kaggle={self.is_kaggle}, is_gpu={self.is_gpu}, smoke={self.smoke_test}")
        print(f"  project_root={self.project_root}")
        print(f"  working_dir={self.working_dir}")
        print(f"  images_dir={self.images_dir}")
        print(f"  hf_token={'***' if self.hf_token else 'NOT FOUND'}")

    def _find_project_root(self) -> Path:
        """Walk up from CWD to find AGENTS.md or .git."""
        p = Path.cwd().resolve()
        for _ in range(10):
            if (p / "AGENTS.md").exists() or (p / ".git").exists():
                return p
            parent = p.parent
            if parent == p:
                break
            p = parent
        # Fallback: search all Kaggle input mounts for AGENTS.md or model/
        if Path("/kaggle/input").exists():
            for candidate in Path("/kaggle/input").iterdir():
                if candidate.is_dir() and (
                    (candidate / "AGENTS.md").exists()
                    or (candidate / ".git").exists()
                    or (candidate / "model").is_dir()
                ):
                    return candidate
        # Last resort
        for candidate in [
            Path("/kaggle/working"),
            Path.cwd().resolve(),
        ]:
            if (candidate / "model").exists() or (candidate / "AGENTS.md").exists():
                return candidate
        return Path.cwd().resolve()

    def _resolve_images_dir(self) -> Path:
        """Find dataset/images across Kaggle mounts or local."""
        # Direct paths first (fast)
        candidates = [
            self.project_root / "dataset" / "images",
            Path("/kaggle/working/ai-image-detection-research/dataset/images"),
            Path("/kaggle/working/dataset/images"),
        ]
        # Kaggle: check top-level input mounts only (no recursive scan)
        if self.is_kaggle:
            input_root = Path("/kaggle/input")
            if input_root.exists():
                for mount in input_root.iterdir():
                    if mount.is_dir():
                        img_dir = mount / "dataset" / "images"
                        if img_dir.exists():
                            candidates.insert(0, img_dir)
                        elif (mount / "images").is_dir():
                            candidates.insert(0, mount / "images")
        for c in candidates:
            if c.exists():
                return c
        return self.project_root / "dataset" / "images"

    def _find_hf_token(self) -> str:
        """Auto-discover HF token from Kaggle Secrets, env, or .env."""
        # 1. Kaggle Secret
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
        # 2. Environment variable
        token = os.environ.get("HF_TOKEN")
        if token:
            return token
        # 3. .env file
        for p in [
            Path("/kaggle/working/.env"),
            self.project_root / ".env",
            Path.home() / ".env",
        ]:
            if p.exists():
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("hf="):
                            return line.split("=", 1)[1]
                        if line.startswith("HF_TOKEN="):
                            return line.split("=", 1)[1]
        return ""

    # ═══════════════════════════════════════════════════════════
    # Checkpoint save/resume (full state)
    # ═══════════════════════════════════════════════════════════

    def save_checkpoint(
        self,
        path: Path,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch: int,
        best_acc: float,
        history: dict,
        extra: Optional[dict] = None,
    ) -> Path:
        """
        Save full training state for resumption after crash/session limit.
        Returns the path saved to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_acc": best_acc,
            "history": history,
            "saved_at": datetime.now().isoformat(),
        }
        if extra:
            state["extra"] = extra

        torch.save(state, path)
        size_mb = path.stat().st_size / 1e6
        print(f"  Checkpoint saved: {path.name} ({size_mb:.1f} MB) @ epoch {epoch}")
        return path

    def load_checkpoint(
        self,
        path: Path,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        device="cpu",
    ) -> Optional[dict]:
        """
        Load full training state. Returns extra info dict or None if not found.
        If optimizer/scheduler/scaler are provided, their states are restored.
        """
        path = Path(path)
        if not path.exists():
            return None

        try:
            state = torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            print(f"  WARNING: Could not load checkpoint {path.name}: {e}")
            return None

        model.load_state_dict(state["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])

        epoch = state.get("epoch", 0)
        best_acc = state.get("best_acc", 0.0)
        history = state.get("history", {})
        extra = state.get("extra", {})

        print(f"  Resumed from {path.name}: epoch={epoch}, best_acc={best_acc:.2f}%")
        return {
            "epoch": epoch,
            "best_acc": best_acc,
            "history": history,
            "extra": extra,
        }

    def find_resume_checkpoint(self, ckpt_dir: Path) -> Optional[Path]:
        """Find the latest checkpoint in a directory for resumption."""
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.exists():
            return None
        candidates = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]
        # Fallback to best.pt
        best = ckpt_dir / "best.pt"
        return best if best.exists() else None

    # ═══════════════════════════════════════════════════════════
    # HuggingFace download (manifest + checkpoints)
    # ═══════════════════════════════════════════════════════════

    def _get_hf_api(self):
        """Get or create HfApi instance."""
        try:
            from huggingface_hub import HfApi
        except ImportError:
            os.system("pip install huggingface_hub -q")
            from huggingface_hub import HfApi
        return HfApi(token=self.hf_token) if self.hf_token else None

    def download_manifest(self, local_path: Path) -> bool:
        """
        Download train_manifest.csv from HF if not present locally.
        Returns True if manifest is available (local or downloaded).
        """
        local_path = Path(local_path)
        if local_path.exists() and local_path.stat().st_size > 1000:
            print(f"  Manifest exists locally: {local_path.name}")
            return True

        if not self.hf_token:
            print("  WARNING: No HF token — cannot download manifest")
            return False

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            os.system("pip install huggingface_hub -q")
            from huggingface_hub import hf_hub_download

        try:
            print(f"  Downloading manifest from {self.hf_manifest_repo}...")
            path = hf_hub_download(
                repo_id=self.hf_manifest_repo,
                filename="train_manifest.csv",
                repo_type="model",
                token=self.hf_token,
            )
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, local_path)
            print(f"  Manifest downloaded: {local_path} ({local_path.stat().st_size / 1e6:.1f} MB)")
            return True
        except Exception as e:
            print(f"  WARNING: Could not download manifest from HF: {e}")
            return False

    def upload_manifest(self, local_path: Path) -> bool:
        """Upload train_manifest.csv to HF for use by other notebooks."""
        return self.upload_to_hf(
            local_path,
            self.hf_manifest_repo,
            "train_manifest.csv",
            commit_message=f"Upload manifest at {datetime.now().isoformat()}",
        )

    def rebuild_manifest_from_kaggle(self, local_path: Path) -> bool:
        """
        Run rebuild_manifest.py to create a manifest from Kaggle-mounted datasets.
        Returns True if manifest was created successfully.
        """
        local_path = Path(local_path)
        if local_path.exists() and local_path.stat().st_size > 1000:
            print(f"  Manifest already exists: {local_path}")
            return True

        rebuild_script = self.project_root / "rebuild_manifest.py"
        if not rebuild_script.exists():
            print(f"  WARNING: {rebuild_script} not found")
            return False

        print("  Running rebuild_manifest.py to create manifest from Kaggle mounts...")
        import subprocess
        try:
            result = subprocess.run(
                [sys.executable, str(rebuild_script), "--yes", "--dry-run",
                 "--input-root", "/kaggle/input"],
                capture_output=True, text=True, timeout=600,
                cwd=str(self.project_root),
            )
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            if result.returncode != 0:
                print(f"  ERROR: rebuild_manifest.py failed:\n{result.stderr[-1000:]}")
                return False
        except subprocess.TimeoutExpired:
            print("  ERROR: rebuild_manifest.py timed out (10 min)")
            return False

        # The script saves to split_manifest_rebuilt.json in CWD
        rebuilt = self.working_dir / "split_manifest_rebuilt.json"
        if rebuilt.exists():
            # Convert JSON manifest to CSV format for the dataset code
            return self._convert_json_manifest_to_csv(rebuilt, local_path)
        return False

    def _convert_json_manifest_to_csv(self, json_path: Path, csv_path: Path) -> bool:
        """Convert rebuild_manifest.py JSON output to CSV format for dataset.py."""
        import hashlib, os
        with open(json_path) as f:
            manifest = json.load(f)

        images_dict = manifest.get("images", {})
        if not images_dict:
            print("  WARNING: Empty manifest")
            return False

        # Scan Kaggle mounts to resolve image paths
        input_root = Path("/kaggle/input")
        slug_map: Dict[str, Path] = {}
        if input_root.exists():
            for base in [input_root, input_root / "datasets"]:
                if not base.exists():
                    continue
                for owner in base.iterdir():
                    if not owner.is_dir():
                        continue
                    for child in owner.iterdir():
                        if child.is_dir():
                            slug_map[child.name] = child

        # Build CSV rows
        rows = []
        img_exts = {'.jpg', '.jpeg', '.png', '.webp'}
        shards = set(info.get("shard", "") for info in images_dict.values())

        for shard in sorted(shards):
            mount = slug_map.get(shard)
            if mount is None:
                for slug, path in slug_map.items():
                    if shard.replace("-", "") in slug.replace("-", "") or slug.replace("-", "") in shard.replace("-", ""):
                        mount = path
                        break
            if mount is None:
                print(f"  Warning: shard '{shard}' not in mounts")
                continue

            # Get entries for this shard
            shard_entries = {k: v for k, v in images_dict.items() if v.get("shard") == shard}

            # Artifact sub-datasets: use metadata.csv
            if shard.startswith("artifact-"):
                count = self._scan_artifact_for_csv(mount, shard, shard_entries, rows)
            else:
                print(f"  Scanning {shard}...", end=" ", flush=True)
                count = 0
                for dirpath, _, filenames in os.walk(str(mount)):
                    for fname in filenames:
                        if Path(fname).suffix.lower() not in img_exts:
                            continue
                        full_path = os.path.join(dirpath, fname)
                        if os.path.getsize(full_path) == 0:
                            continue
                        for lbl in ["real", "ai_generated", "deepfake"]:
                            label_int = {"real": 0, "ai_generated": 1, "deepfake": 2}[lbl]
                            img_id = hashlib.sha1(f"{fname}|{shard}|{lbl}".encode()).hexdigest()[:20]
                            if img_id in shard_entries:
                                rows.append({
                                    'image_id': img_id,
                                    'filename': full_path,
                                    'label': lbl,
                                    'source': shard,
                                    'generator': '',
                                    'width': 0,
                                    'height': 0,
                                    'file_size_bytes': os.path.getsize(full_path),
                                    'md5': '',
                                })
                                count += 1
                                break
                print(f"{count} images")

        if not rows:
            print("  WARNING: No images resolved")
            return False

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"  Manifest CSV created: {csv_path} ({len(df)} rows)")
        return True

    def _scan_artifact_for_csv(self, mount_dir: Path, shard: str, shard_entries: dict, rows: list) -> int:
        """Scan artifact sub-dataset using metadata.csv for CSV conversion."""
        import hashlib, os, pandas as pd
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
                continue
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
                    rows.append({
                        'image_id': img_id,
                        'filename': img_path,
                        'label': lbl,
                        'source': shard,
                        'generator': '',
                        'width': 0,
                        'height': 0,
                        'file_size_bytes': os.path.getsize(img_path),
                        'md5': '',
                    })
                    count += 1
                    break
        print(f"  Scanning {shard}... {count} images")
        return count

    def download_latest_checkpoint(self, ckpt_dir: Path, model_name: str) -> Optional[Path]:
        """
        Download the latest checkpoint from HF if no local checkpoint exists.
        Returns path to local checkpoint or None.
        """
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Check local first
        local = self.find_resume_checkpoint(ckpt_dir)
        if local is not None:
            print(f"  Local checkpoint found: {local.name}")
            return local

        if not self.hf_token:
            return None

        try:
            from huggingface_hub import list_repo_files, hf_hub_download
        except ImportError:
            os.system("pip install huggingface_hub -q")
            from huggingface_hub import list_repo_files, hf_hub_download

        try:
            print(f"  No local checkpoint — checking HF ({self.hf_checkpoint_repo})...")
            files = list_repo_files(self.hf_checkpoint_repo, repo_type="model", token=self.hf_token)
            ckpt_files = sorted(
                [f for f in files if f.startswith(f"checkpoints/{model_name}/checkpoint_epoch_")],
                key=lambda x: int(x.split("_")[-1].replace(".pt", "")),
            )
            if not ckpt_files:
                # Try best.pt
                best_path = f"checkpoints/{model_name}/best.pt"
                if best_path in files:
                    ckpt_files = [best_path]

            if not ckpt_files:
                print("  No checkpoint found on HF")
                return None

            latest = ckpt_files[-1]
            print(f"  Downloading {latest} from HF...")
            path = hf_hub_download(
                repo_id=self.hf_checkpoint_repo,
                filename=latest,
                repo_type="model",
                token=self.hf_token,
            )
            local_ckpt = ckpt_dir / Path(latest).name
            shutil.copy2(path, local_ckpt)
            print(f"  Checkpoint downloaded: {local_ckpt.name}")
            return local_ckpt
        except Exception as e:
            print(f"  WARNING: Could not download checkpoint from HF: {e}")
            return None

    # ═══════════════════════════════════════════════════════════
    # HuggingFace upload
    # ═══════════════════════════════════════════════════════════

    def upload_to_hf(
        self,
        local_path: Path,
        repo_id: str,
        path_in_repo: str,
        repo_type: str = "model",
        commit_message: str = "",
    ) -> bool:
        """Upload a file or directory to HuggingFace Hub."""
        if not self.hf_token:
            print(f"  WARNING: No HF token — skipping upload to {repo_id}")
            return False

        try:
            from huggingface_hub import HfApi
        except ImportError:
            os.system("pip install huggingface_hub -q")
            from huggingface_hub import HfApi

        api = HfApi(token=self.hf_token)
        api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)

        local_path = Path(local_path)
        msg = commit_message or f"Upload {local_path.name} at {datetime.now().isoformat()}"

        if local_path.is_dir():
            api.upload_folder(
                folder_path=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=msg,
            )
        else:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=msg,
            )
        print(f"  Uploaded: {local_path.name} -> {repo_id}/{path_in_repo}")
        return True

    def upload_checkpoint(self, local_ckpt: Path, model_name: str) -> bool:
        """Upload a checkpoint file to the checkpoints HF repo."""
        return self.upload_to_hf(
            local_ckpt,
            self.hf_checkpoint_repo,
            f"checkpoints/{model_name}/{local_ckpt.name}",
        )

    def upload_results_dir(self, local_dir: Path, tag: str = "") -> bool:
        """Upload an entire results directory to the results HF repo."""
        return self.upload_to_hf(
            local_dir,
            self.hf_results_repo,
            f"results/{tag}/{local_dir.name}" if tag else f"results/{local_dir.name}",
        )

    def upload_model_for_deployment(self, local_ckpt: Path, model_name: str) -> bool:
        """Upload a model checkpoint to the deployment HF repo."""
        return self.upload_to_hf(
            local_ckpt,
            self.hf_model_repo,
            f"models/{model_name}/{local_ckpt.name}",
            commit_message=f"Deploy {model_name} checkpoint",
        )

    # ═══════════════════════════════════════════════════════════
    # Deployment export
    # ═══════════════════════════════════════════════════════════

    def export_for_deployment(
        self,
        model,
        variant: str,
        local_dir: Path,
        best_acc: float = 0.0,
        num_classes: int = 2,
        extra_config: Optional[dict] = None,
    ) -> Path:
        """
        Save a deployment-ready package: state_dict + config.json + README.md.
        Returns the directory path.
        """
        import torch

        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save state_dict (clean, no optimizer/scheduler)
        model_path = local_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        # 2. Build config
        config = {
            "variant": variant,
            "num_classes": num_classes,
            "image_size": getattr(model, "img_size", 384),
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "best_val_acc": round(best_acc, 2),
            "framework": "pytorch",
            "architecture": "MFFT (Multi-Frequency Fusion Transformer)",
            "exported_at": datetime.now().isoformat(),
        }
        if extra_config:
            config.update(extra_config)

        config_path = local_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # 3. Generate README
        readme = f"""# MFFT-{variant.title()} — AI Image Detection Model

## Architecture
Multi-Frequency Fusion Transformer (MFFT-{variant})

- **Parameters**: {config['total_params']:,}
- **Image Size**: {config['image_size']}x{config['image_size']}
- **Classes**: {num_classes} (real, ai_generated)
- **Best Val Acc**: {best_acc:.2f}%

## Usage

```python
import torch
from src.model import build_mfft

model = build_mfft("{variant}")
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()

# Inference
from src.dataset import ImageTransform
transform = ImageTransform(size={config['image_size']}, augment=False)
img = transform(pil_image).unsqueeze(0)
logits = model(img)
probs = torch.softmax(logits, dim=-1)
```

## Files
- `model.pt` — PyTorch state_dict
- `config.json` — Model configuration
- `README.md` — This file

## Exported
- Date: {config['exported_at']}
- Framework: PyTorch {torch.__version__}
"""
        readme_path = local_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme)

        size_mb = model_path.stat().st_size / 1e6
        print(f"  Deployment package: {local_dir} ({size_mb:.1f} MB)")
        return local_dir

    def upload_deployment_model(self, local_dir: Path, variant: str) -> bool:
        """Upload deployment package to the mfft-model HF repo."""
        return self.upload_to_hf(
            local_dir,
            self.hf_model_repo,
            f"models/{variant}",
            commit_message=f"Upload MFFT-{variant} deployment package",
        )

    def upload_figures_and_metrics(
        self,
        results_dir: Path,
        tag: str = "",
    ) -> bool:
        """
        Upload fig/ and table/ (or entire results_dir) to mfft-results repo.
        Preserves directory structure under results/<tag>/.
        """
        results_dir = Path(results_dir)
        if not results_dir.exists():
            print(f"  WARNING: {results_dir} does not exist — skipping upload")
            return False

        base_path = f"results/{tag}" if tag else "results"
        ok = True

        # Upload fig/ directory if it exists
        fig_dir = results_dir / "fig"
        if fig_dir.exists() and any(fig_dir.iterdir()):
            ok &= self.upload_to_hf(
                fig_dir, self.hf_results_repo, f"{base_path}/fig",
                commit_message=f"Upload {tag} figures",
            )

        # Upload table/ directory if it exists
        table_dir = results_dir / "table"
        if table_dir.exists() and any(table_dir.iterdir()):
            ok &= self.upload_to_hf(
                table_dir, self.hf_results_repo, f"{base_path}/table",
                commit_message=f"Upload {tag} tables",
            )

        # Upload standalone files (CSVs, PNGs in results_dir root)
        for f in results_dir.iterdir():
            if f.is_file() and f.suffix in (".csv", ".png", ".json"):
                ok &= self.upload_to_hf(
                    f, self.hf_results_repo, f"{base_path}/{f.name}",
                    commit_message=f"Upload {tag}/{f.name}",
                )

        return ok


def setup_notebook_env(
    variant: str = "base",
    mode: str = "full_scale",
) -> KaggleEnv:
    """
    One-call setup for notebook Cell 1.
    Returns a fully initialized KaggleEnv.
    """
    env = KaggleEnv(project_root_search=True, variant=variant, mode=mode)
    return env

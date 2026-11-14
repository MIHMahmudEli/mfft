from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class TrainingConfig:
    model_variant: str = "base"
    image_size: int = 384
    batch_size: int = 32
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    val_check_interval: int = 500
    save_top_k: int = 3

    scheduler: str = "cosine"
    optimizer: str = "adamw"

    num_workers: int = 4
    seed: int = 42

    def __post_init__(self):
        root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class DatasetConfig:
    root_dir: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    metadata_paths: List[str] = field(default_factory=lambda: [
        os.path.join(os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        ), "dataset", "metadata", "clean_metadata.csv"),
    ])
    undersample: bool = False          # superseded by weighted sampling
    use_weighted_sampler: bool = True  # oversample minority class, keep all data
    val_split: float = 0.10
    test_split: float = 0.10
    split_index_path: str = os.path.join(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    ), "dataset", "metadata", "split_indices.json")


@dataclass
class WandbConfig:
    project: str = "mfft-ai-detection"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    log_every_n_steps: int = 50
    watch_model: bool = True


@dataclass
class Config:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    output_dir: str = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        "checkpoints"
    )

    def to_dict(self) -> dict:
        return {
            "training": self.training.to_dict(),
            "dataset": self.dataset.__dict__,
            "wandb": self.wandb.__dict__,
            "output_dir": self.output_dir,
        }

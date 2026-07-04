"""Configuration objects and repository-local paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    """Default filesystem locations used by the command line tools."""

    root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    default_checkpoint: Path = (
        PROJECT_ROOT / "artifacts" / "checkpoints" / "mnist" / "final_best_epoch.pth.tar"
    )
    sample_output_dir: Path = PROJECT_ROOT / "outputs" / "samples"
    checkpoint_dir: Path = PROJECT_ROOT / "outputs" / "checkpoints"


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for the MNIST MVAE training loop."""

    epochs: int = 500
    latent_size: int = 64
    batch_size: int = 100
    lr: float = 1e-3
    beta_anneal_epochs: int = 200
    lambda_image: float = 1.0
    lambda_label: float = 50.0
    seed: int = 42
    train_size: int = 50_000
    val_size: int = 10_000
    num_workers: int = 0
    checkpoint_interval: int = 2
    lr_decay_step: int = 5
    lr_decay_gamma: float = 0.1
    min_lr: float = 2e-6


@dataclass(frozen=True)
class SamplingConfig:
    """Parameters for conditioned generation."""

    condition_label: int | None = 6
    condition_image_label: int | None = None
    num_samples: int = 64
    seed: int = 42
    device: str = "auto"


DEFAULT_PATHS = ProjectPaths()

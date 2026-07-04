"""MNIST-focused Multimodal Variational Autoencoder package."""

from mvae_mnist.config import DEFAULT_PATHS, ProjectPaths, SamplingConfig, TrainingConfig
from mvae_mnist.model import MVAE

__all__ = [
    "DEFAULT_PATHS",
    "MVAE",
    "ProjectPaths",
    "SamplingConfig",
    "TrainingConfig",
]

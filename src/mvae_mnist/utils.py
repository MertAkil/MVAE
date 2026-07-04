"""Shared utilities for checkpoints, devices, seeds, and conditioning."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import MNIST

from mvae_mnist.config import DEFAULT_PATHS
from mvae_mnist.model import MVAE


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve `auto`, `cpu`, `cuda`, or `mps` into a concrete torch device."""

    if isinstance(requested, torch.device):
        return requested

    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if requested == "mps":
        mps_available = getattr(torch.backends, "mps", None) is not None
        if not mps_available or not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("Device must be one of: auto, cpu, cuda, mps.")

    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_digit(digit: int | None) -> int | None:
    if digit is None:
        return None
    if digit < 0 or digit > 9:
        raise ValueError("MNIST digit conditions must be in the range [0, 9].")
    return digit


def save_checkpoint(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _torch_load(path: Path, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    file_path: Path | str,
    device: str | torch.device = "cpu",
) -> tuple[MVAE, optim.Optimizer, dict[str, Any]]:
    """Load an MVAE checkpoint while preserving compatibility with old weights."""

    checkpoint_path = Path(file_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    target_device = resolve_device(device)
    checkpoint = _torch_load(checkpoint_path, map_location=target_device)
    latent_dim = checkpoint.get("n_latents", checkpoint.get("latent_size", 64))

    model = MVAE(latent_dim).to(target_device)
    model.load_state_dict(checkpoint["state_dict"])

    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer, checkpoint


def fetch_label(label: int, device: torch.device) -> Tensor:
    label = validate_digit(label)
    return torch.tensor([label], dtype=torch.long, device=device)


def fetch_mnist_image(
    label: int,
    data_dir: Path = DEFAULT_PATHS.data_dir,
    device: torch.device | None = None,
    seed: int | None = None,
) -> Tensor:
    """Fetch one test-set MNIST image with the requested label."""

    label = validate_digit(label)
    dataset = MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    targets = dataset.targets
    if not torch.is_tensor(targets):
        targets = torch.as_tensor(targets)

    matches = torch.nonzero(targets == label, as_tuple=False).flatten()
    if len(matches) == 0:
        raise ValueError(f"No MNIST test images found for label {label}.")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    match_index = torch.randint(len(matches), (1,), generator=generator).item()
    image, _ = dataset[int(matches[match_index])]
    image = image.unsqueeze(0)
    return image.to(device) if device is not None else image


def condition_posterior(
    model: MVAE,
    condition_image_label: int | None = None,
    condition_label: int | None = None,
    condition_image: Tensor | None = None,
    data_dir: Path = DEFAULT_PATHS.data_dir,
    device: torch.device | None = None,
    seed: int | None = None,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Return posterior parameters for image, label, joint, or prior conditioning."""

    target_device = device or next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    condition_image_label = validate_digit(condition_image_label)
    condition_label = validate_digit(condition_label)

    image = condition_image.to(target_device) if condition_image is not None else None
    if image is None and condition_image_label is not None:
        image = fetch_mnist_image(
            label=condition_image_label,
            data_dir=data_dir,
            device=target_device,
            seed=seed,
        )

    label = fetch_label(condition_label, target_device) if condition_label is not None else None

    if image is None and label is None:
        mu = torch.zeros((1, model.z_dim), device=target_device, dtype=dtype)
        logvar = torch.zeros_like(mu)
    else:
        mu_stack, logvar_stack = model.prepare_poe(image_modal=image, label_modal=label)
        mu, logvar = model.compute_poe(mu_stack, logvar_stack)

    return mu, logvar, image, label


def check_modality_cond(
    condition_on_image: int | None,
    condition_on_text: int | None,
    model: MVAE,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Backward-compatible wrapper around `condition_posterior`."""

    mu, logvar, image, label = condition_posterior(
        model=model,
        condition_image_label=condition_on_image,
        condition_label=condition_on_text,
    )
    return mu, logvar.mul(0.5).exp(), image, label

"""Conditioned sample generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision.utils import save_image

from mvae_mnist.config import DEFAULT_PATHS
from mvae_mnist.utils import condition_posterior, load_checkpoint, resolve_device, set_seed


@dataclass(frozen=True)
class SampleOutput:
    image_path: Path
    label_path: Path
    labels: list[int]


def generate_samples(
    checkpoint_path: Path = DEFAULT_PATHS.default_checkpoint,
    output_dir: Path = DEFAULT_PATHS.sample_output_dir,
    condition_label: int | None = 6,
    condition_image_label: int | None = None,
    num_samples: int = 64,
    device: str = "auto",
    seed: int = 42,
    data_dir: Path = DEFAULT_PATHS.data_dir,
) -> SampleOutput:
    """Generate image samples and predicted digit labels from a trained checkpoint."""

    target_device = resolve_device(device)
    set_seed(seed)

    model, _, _ = load_checkpoint(checkpoint_path, device=target_device)
    model.eval()

    with torch.no_grad():
        mu, logvar, _, _ = condition_posterior(
            model=model,
            condition_image_label=condition_image_label,
            condition_label=condition_label,
            data_dir=data_dir,
            device=target_device,
            seed=seed,
        )
        std = logvar.mul(0.5).exp()
        samples = torch.randn(num_samples, model.z_dim, device=target_device)
        samples = samples * std.expand_as(samples) + mu.expand_as(samples)

        generated_images = torch.sigmoid(model.image_decoder(samples))
        label_probs = torch.softmax(model.label_decoder(samples), dim=1)
        labels = torch.argmax(label_probs, dim=1).detach().cpu().tolist()
        confidences = torch.max(label_probs, dim=1).values.detach().cpu().tolist()

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "generated_image.png"
    label_path = output_dir / "generated_label.txt"

    save_image(generated_images.view(num_samples, 1, 28, 28), image_path)
    with label_path.open("w", encoding="utf-8") as label_file:
        for index, (label, confidence) in enumerate(zip(labels, confidences, strict=True)):
            label_file.write(f"sample_{index:03d}: label={label} confidence={confidence:.4f}\n")

    return SampleOutput(image_path=image_path, label_path=label_path, labels=labels)

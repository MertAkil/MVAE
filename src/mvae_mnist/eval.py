"""Lightweight importance-sampling evaluation for the MNIST MVAE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from mvae_mnist.config import DEFAULT_PATHS
from mvae_mnist.eval_utils import (
    bernoulli_log_pdf,
    categorical_log_pdf,
    gaussian_log_pdf,
    log_mean_exp,
    unit_gaussian_log_pdf,
)
from mvae_mnist.utils import (
    condition_posterior,
    fetch_label,
    fetch_mnist_image,
    load_checkpoint,
    resolve_device,
    set_seed,
)


@dataclass(frozen=True)
class EvaluationResult:
    negative_log_joint: float
    negative_log_marginal_image: float


def log_joint_estimate(
    recon_image: Tensor,
    image: Tensor,
    recon_label: Tensor,
    label: Tensor,
    z: Tensor,
    mu: Tensor,
    logvar: Tensor,
) -> Tensor:
    """Estimate negative log p(x, y) with importance samples."""

    batch_size, n_samples, z_dim = z.size()
    input_dim = image.size(1)
    label_dim = recon_label.size(2)

    image = image.unsqueeze(1).repeat(1, n_samples, 1)
    label = label.unsqueeze(1).repeat(1, n_samples)

    z2d = z.view(batch_size * n_samples, z_dim)
    mu2d = mu.view(batch_size * n_samples, z_dim)
    logvar2d = logvar.view(batch_size * n_samples, z_dim)
    recon_image_2d = recon_image.view(batch_size * n_samples, input_dim)
    image_2d = image.view(batch_size * n_samples, input_dim)
    recon_label_2d = recon_label.view(batch_size * n_samples, label_dim)
    label_2d = label.view(batch_size * n_samples)

    log_p_x_given_z_2d = bernoulli_log_pdf(image_2d, recon_image_2d)
    log_p_y_given_z_2d = categorical_log_pdf(label_2d, recon_label_2d)
    log_q_z_given_x_2d = gaussian_log_pdf(z2d, mu2d, logvar2d)
    log_p_z_2d = unit_gaussian_log_pdf(z2d)

    log_weight_2d = (
        log_p_x_given_z_2d + log_p_y_given_z_2d + log_p_z_2d - log_q_z_given_x_2d
    )
    log_weight = log_weight_2d.view(batch_size, n_samples)
    return -torch.mean(log_mean_exp(log_weight, dim=1))


def log_marginal_estimate(
    recon_image: Tensor,
    image: Tensor,
    z: Tensor,
    mu: Tensor,
    logvar: Tensor,
) -> Tensor:
    """Estimate negative log p(x) with importance samples."""

    batch_size, n_samples, z_dim = z.size()
    input_dim = image.size(1)
    image = image.unsqueeze(1).repeat(1, n_samples, 1)

    z2d = z.view(batch_size * n_samples, z_dim)
    mu2d = mu.view(batch_size * n_samples, z_dim)
    logvar2d = logvar.view(batch_size * n_samples, z_dim)
    recon_image_2d = recon_image.view(batch_size * n_samples, input_dim)
    image_2d = image.view(batch_size * n_samples, input_dim)

    log_p_x_given_z_2d = bernoulli_log_pdf(image_2d, recon_image_2d)
    log_q_z_given_x_2d = gaussian_log_pdf(z2d, mu2d, logvar2d)
    log_p_z_2d = unit_gaussian_log_pdf(z2d)

    log_weight_2d = log_p_x_given_z_2d + log_p_z_2d - log_q_z_given_x_2d
    log_weight = log_weight_2d.view(batch_size, n_samples)
    return -torch.mean(log_mean_exp(log_weight, dim=1))


def evaluate_checkpoint(
    checkpoint_path: Path = DEFAULT_PATHS.default_checkpoint,
    condition_label: int | None = 5,
    condition_image_label: int | None = None,
    n_samples: int = 64,
    device: str = "auto",
    seed: int = 42,
    data_dir: Path = DEFAULT_PATHS.data_dir,
) -> EvaluationResult:
    """Evaluate a checkpoint under the requested conditioning setup."""

    if condition_label is None and condition_image_label is None:
        raise ValueError("Evaluation requires a condition label or condition image label.")

    target_device = resolve_device(device)
    set_seed(seed)
    model, _, _ = load_checkpoint(checkpoint_path, device=target_device)
    model.eval()

    observed_digit = condition_label if condition_label is not None else condition_image_label
    observed_image = fetch_mnist_image(
        label=observed_digit,
        data_dir=data_dir,
        device=target_device,
        seed=seed,
    )
    observed_label = fetch_label(observed_digit, target_device)

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
        samples = torch.randn(n_samples, model.z_dim, device=target_device)
        samples = samples * std.expand_as(samples) + mu.expand_as(samples)

        recon_image = model.image_decoder(samples).view(1, n_samples, 784)
        recon_label = F.log_softmax(model.label_decoder(samples), dim=1).view(1, n_samples, 10)
        image = observed_image.view(1, 784)
        z = samples.view(1, n_samples, model.z_dim)
        mu_samples = mu.expand_as(samples).view(1, n_samples, model.z_dim)
        logvar_samples = logvar.expand_as(samples).view(1, n_samples, model.z_dim)

        negative_log_joint = log_joint_estimate(
            recon_image,
            image,
            recon_label,
            observed_label,
            z,
            mu_samples,
            logvar_samples,
        )
        negative_log_marginal = log_marginal_estimate(
            recon_image,
            image,
            z,
            mu_samples,
            logvar_samples,
        )

    return EvaluationResult(
        negative_log_joint=float(negative_log_joint.detach().cpu()),
        negative_log_marginal_image=float(negative_log_marginal.detach().cpu()),
    )

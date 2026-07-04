"""Numerically stable log-probability helpers for evaluation."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

LOG2PI = float(np.log(2.0 * math.pi))


def bernoulli_log_pdf(x: Tensor, logits: Tensor) -> Tensor:
    """Log-likelihood of Bernoulli data parameterized by logits."""

    log_pdf = -F.relu(logits) + x * logits - torch.log1p(torch.exp(-logits.abs()))
    return torch.sum(log_pdf, dim=1)


def categorical_log_pdf(x: Tensor, log_probs: Tensor) -> Tensor:
    """Log-likelihood of categorical labels parameterized by log probabilities."""

    x_one_hot = F.one_hot(x.long(), num_classes=log_probs.size(1)).to(
        device=log_probs.device,
        dtype=log_probs.dtype,
    )
    return torch.sum(x_one_hot * log_probs, dim=1)


def gaussian_log_pdf(x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    """Log-likelihood of samples under a diagonal Gaussian."""

    log_pdf = -0.5 * LOG2PI - logvar / 2.0 - torch.pow(x - mu, 2) / (2.0 * torch.exp(logvar))
    return torch.sum(log_pdf, dim=1)


def unit_gaussian_log_pdf(x: Tensor) -> Tensor:
    """Log-likelihood of samples under N(0, I)."""

    log_pdf = -0.5 * LOG2PI - torch.pow(x, 2) / 2.0
    return torch.sum(log_pdf, dim=1)


def log_mean_exp(x: Tensor, dim: int = 1) -> Tensor:
    """Compute log(mean(exp(x))) along one dimension."""

    return torch.logsumexp(x, dim=dim, keepdim=True) - math.log(x.size(dim))

"""Training loop for the MNIST MVAE."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mvae_mnist.config import DEFAULT_PATHS, TrainingConfig
from mvae_mnist.data import build_mnist_dataloaders
from mvae_mnist.model import MVAE
from mvae_mnist.utils import load_checkpoint, resolve_device, save_checkpoint, set_seed


def elbo_mnist(
    gen_image: Tensor | None,
    image: Tensor | None,
    gen_label: Tensor | None,
    label: Tensor | None,
    mu: Tensor,
    logvar: Tensor,
    lambda_image: float = 1.0,
    lambda_label: float = 50.0,
    annealing_factor: float = 1.0,
) -> Tensor:
    """Negative ELBO objective used for the MNIST MVAE."""

    image_bce: Tensor | float = 0.0
    label_bce: Tensor | float = 0.0

    if gen_image is not None and image is not None:
        image_bce = F.binary_cross_entropy_with_logits(
            gen_image.view(-1, 28 * 28),
            image.view(-1, 28 * 28),
            reduction="none",
        )
        image_bce = torch.sum(image_bce, dim=1)

    if gen_label is not None and label is not None:
        label_bce = F.cross_entropy(gen_label, label, reduction="none")

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return torch.mean(lambda_image * image_bce + lambda_label * label_bce + annealing_factor * kld)


class ClampedStepScheduler:
    """Small scheduler matching the original step-down behavior without private APIs."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        step_size: int,
        gamma: float,
        min_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr

    def step(self, epoch: int) -> None:
        if (epoch + 1) % self.step_size != 0:
            return
        for group in self.optimizer.param_groups:
            group["lr"] = max(float(group["lr"]) * self.gamma, self.min_lr)


def _move_optimizer_to_device(optimizer: optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _batch_loss(
    model: MVAE,
    image: Tensor,
    label: Tensor,
    config: TrainingConfig,
    annealing_factor: float,
) -> Tensor:
    gen_image, _, mu_image, logvar_image = model(image_modal=image, label_modal=None)
    _, gen_label, mu_label, logvar_label = model(image_modal=None, label_modal=label)
    gen_image_joint, gen_label_joint, mu_joint, logvar_joint = model(
        image_modal=image,
        label_modal=label,
    )

    joint_loss = elbo_mnist(
        gen_image_joint,
        image,
        gen_label_joint,
        label,
        mu_joint,
        logvar_joint,
        config.lambda_image,
        config.lambda_label,
        annealing_factor,
    )
    image_loss = elbo_mnist(
        gen_image,
        image,
        None,
        None,
        mu_image,
        logvar_image,
        config.lambda_image,
        config.lambda_label,
        annealing_factor,
    )
    label_loss = elbo_mnist(
        None,
        None,
        gen_label,
        label,
        mu_label,
        logvar_label,
        config.lambda_image,
        config.lambda_label,
        annealing_factor,
    )
    return joint_loss + image_loss + label_loss


def train_epoch(
    model: MVAE,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    epoch: int,
    device: torch.device,
    limit_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    beta = min(1.0, epoch / config.beta_anneal_epochs)

    progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.epochs} beta={beta:.3f}")
    for batch_index, (image, label) in enumerate(progress):
        if limit_batches is not None and batch_index >= limit_batches:
            break

        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        loss = _batch_loss(model, image, label, config, annealing_factor=beta)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1
        progress.set_postfix(loss=total_loss / n_batches)

    if n_batches == 0:
        raise ValueError("Training dataloader produced no batches.")
    return total_loss / n_batches


def validate_epoch(
    model: MVAE,
    dataloader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    limit_batches: int | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_index, (image, label) in enumerate(dataloader):
            if limit_batches is not None and batch_index >= limit_batches:
                break
            image = image.to(device)
            label = label.to(device)
            loss = _batch_loss(model, image, label, config, annealing_factor=1.0)
            total_loss += float(loss.detach().cpu())
            n_batches += 1

    if n_batches == 0:
        raise ValueError("Validation dataloader produced no batches.")
    return total_loss / n_batches


def run_training(
    config: TrainingConfig | None = None,
    checkpoint_dir: Path = DEFAULT_PATHS.checkpoint_dir,
    data_dir: Path = DEFAULT_PATHS.data_dir,
    device: str = "auto",
    resume: bool = False,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
) -> dict[str, list[float] | float | int]:
    """Train the MVAE and write checkpoints to `checkpoint_dir`."""

    config = config or TrainingConfig()
    target_device = resolve_device(device)
    set_seed(config.seed)

    loaders = build_mnist_dataloaders(
        data_dir=data_dir,
        batch_size=config.batch_size,
        train_size=config.train_size,
        val_size=config.val_size,
        seed=config.seed,
        num_workers=config.num_workers,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "final_best_epoch.pth.tar"

    if resume and best_path.exists():
        model, optimizer, checkpoint = load_checkpoint(best_path, device=target_device)
        _move_optimizer_to_device(optimizer, target_device)
        train_losses = list(checkpoint.get("train_loss_list", []))
        val_losses = list(checkpoint.get("val_loss_list", []))
        start_epoch = len(train_losses)
        best_loss = float(checkpoint.get("best_loss", val_losses[-1] if val_losses else np.inf))
    else:
        model = MVAE(config.latent_size).to(target_device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        train_losses = []
        val_losses = []
        start_epoch = 0
        best_loss = np.inf

    scheduler = ClampedStepScheduler(
        optimizer=optimizer,
        step_size=config.lr_decay_step,
        gamma=config.lr_decay_gamma,
        min_lr=config.min_lr,
    )

    for epoch in range(start_epoch, config.epochs):
        train_loss = train_epoch(
            model,
            loaders.train,
            optimizer,
            config,
            epoch,
            target_device,
            limit_batches=limit_train_batches,
        )
        val_loss = validate_epoch(
            model,
            loaders.val,
            config,
            target_device,
            limit_batches=limit_val_batches,
        )
        scheduler.step(epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        state = {
            "state_dict": model.state_dict(),
            "best_loss": best_loss,
            "n_latents": config.latent_size,
            "optimizer": optimizer.state_dict(),
            "train_loss_list": train_losses,
            "val_loss_list": val_losses,
            "epoch": epoch + 1,
        }

        if is_best:
            save_checkpoint(state, best_path)
        if (epoch + 1) % config.checkpoint_interval == 0:
            save_checkpoint(state, checkpoint_dir / f"epoch_{epoch + 1}.pth.tar")

        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} best_loss={best_loss:.4f}"
        )

    return {
        "train_loss_list": train_losses,
        "val_loss_list": val_losses,
        "best_loss": float(best_loss),
        "epochs_completed": config.epochs,
    }


def with_overrides(config: TrainingConfig, **overrides: object) -> TrainingConfig:
    """Return a config copy with CLI-provided overrides."""

    return replace(config, **{key: value for key, value in overrides.items() if value is not None})

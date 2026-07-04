"""Command line interface for the MNIST MVAE project."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mvae_mnist.config import DEFAULT_PATHS, TrainingConfig
from mvae_mnist.eval import evaluate_checkpoint
from mvae_mnist.sample import generate_samples
from mvae_mnist.train import run_training


def _optional_digit(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    digit = int(value)
    if digit < 0 or digit > 9:
        raise argparse.ArgumentTypeError("digit must be between 0 and 9")
    return digit


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mvae-mnist",
        description="Train, sample, and evaluate the MNIST Multimodal VAE.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample", help="Generate conditioned samples.")
    sample_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PATHS.default_checkpoint)
    sample_parser.add_argument("--output-dir", type=Path, default=DEFAULT_PATHS.sample_output_dir)
    sample_parser.add_argument("--data-dir", type=Path, default=DEFAULT_PATHS.data_dir)
    sample_parser.add_argument("--condition-label", type=_optional_digit, default=6)
    sample_parser.add_argument("--condition-image-label", type=_optional_digit, default=None)
    sample_parser.add_argument("--num-samples", type=_positive_int, default=64)
    sample_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    sample_parser.add_argument("--seed", type=int, default=42)
    sample_parser.add_argument(
        "--prior",
        action="store_true",
        help="Ignore conditions and sample from the unit Gaussian prior.",
    )

    train_parser = subparsers.add_parser("train", help="Train or smoke-test the model.")
    train_parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_PATHS.checkpoint_dir)
    train_parser.add_argument("--data-dir", type=Path, default=DEFAULT_PATHS.data_dir)
    train_parser.add_argument("--epochs", type=_positive_int, default=TrainingConfig.epochs)
    train_parser.add_argument(
        "--latent-size",
        type=_positive_int,
        default=TrainingConfig.latent_size,
    )
    train_parser.add_argument("--batch-size", type=_positive_int, default=TrainingConfig.batch_size)
    train_parser.add_argument("--lr", type=float, default=TrainingConfig.lr)
    train_parser.add_argument("--train-size", type=_positive_int, default=TrainingConfig.train_size)
    train_parser.add_argument("--val-size", type=_positive_int, default=TrainingConfig.val_size)
    train_parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    train_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--limit-train-batches", type=_positive_int, default=None)
    train_parser.add_argument("--limit-val-batches", type=_positive_int, default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Run a lightweight checkpoint evaluation.")
    eval_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PATHS.default_checkpoint)
    eval_parser.add_argument("--data-dir", type=Path, default=DEFAULT_PATHS.data_dir)
    eval_parser.add_argument("--condition-label", type=_optional_digit, default=5)
    eval_parser.add_argument("--condition-image-label", type=_optional_digit, default=None)
    eval_parser.add_argument("--num-samples", type=_positive_int, default=64)
    eval_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    eval_parser.add_argument("--seed", type=int, default=42)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "sample":
        condition_label = None if args.prior else args.condition_label
        condition_image_label = None if args.prior else args.condition_image_label
        result = generate_samples(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            condition_label=condition_label,
            condition_image_label=condition_image_label,
            num_samples=args.num_samples,
            device=args.device,
            seed=args.seed,
            data_dir=args.data_dir,
        )
        print(f"wrote image grid: {result.image_path}")
        print(f"wrote predicted labels: {result.label_path}")
        return 0

    if args.command == "train":
        config = TrainingConfig(
            epochs=args.epochs,
            latent_size=args.latent_size,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            train_size=args.train_size,
            val_size=args.val_size,
        )
        result = run_training(
            config=config,
            checkpoint_dir=args.checkpoint_dir,
            data_dir=args.data_dir,
            device=args.device,
            resume=args.resume,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
        )
        print(f"best_loss={result['best_loss']:.4f}")
        return 0

    if args.command == "evaluate":
        result = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            condition_label=args.condition_label,
            condition_image_label=args.condition_image_label,
            n_samples=args.num_samples,
            device=args.device,
            seed=args.seed,
            data_dir=args.data_dir,
        )
        print(f"negative_log_joint={result.negative_log_joint:.4f}")
        print(f"negative_log_marginal_image={result.negative_log_marginal_image:.4f}")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

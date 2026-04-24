"""
Tiny ablation runner for the current 8-task FaceXFormer implementation.

This is not intended to reproduce paper numbers locally. It trains tiny
variants for a few batches to verify:
- dataset loading
- model forward/backward
- sampler options
- loss-weight options
- approximate attention ablation plumbing

Recommended first run:
    python scripts/ablation_study_tiny.py --variants full standard_cross_attention unbalanced_sampler uniform_loss --max-samples 8 --max-train-batches 2 --epochs 1

Later, keep this script structure but increase samples/epochs on a cloud or
cluster machine.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.optim as optim

from config import config
from datasets import create_multi_task_dataloader
from losses import MultiTaskLoss
from network.models.facexformer import FaceXFormer
from scripts.baseline_verification import evaluate_tiny
from scripts.small_run_common import (
    build_eval_datasets,
    build_train_datasets,
    load_checkpoint_if_available,
    predictions_from_outputs,
    resolve_dataset_root,
    set_seed,
    write_json,
)


DEFAULT_TRAIN_TASKS = ["segmentation", "landmark", "headpose"]
DEFAULT_VARIANTS = ["full", "standard_cross_attention", "unbalanced_sampler", "uniform_loss"]


class ZeroAttention(torch.nn.Module):
    """
    Attention replacement used for a lightweight local ablation.

    The paper compares standard cross-attention against bidirectional
    cross-attention. In this repo's current transformer, the extra direction is
    the image-to-token/face-to-task attention module. Replacing it with zeros
    approximates "standard cross-attention" for smoke testing.
    """

    def forward(self, q, k, v):
        return torch.zeros_like(q)


def apply_variant(model, variant):
    """Apply model-side changes for a named ablation variant."""

    if variant == "standard_cross_attention":
        for layer in model.face_decoder.transformer.layers:
            layer.cross_attn_image_to_token = ZeroAttention()
    elif variant in {"full", "unbalanced_sampler", "uniform_loss"}:
        pass
    else:
        raise ValueError(f"Unknown variant: {variant}")


def loss_weights_for_variant(variant):
    """Use paper-style uniform weights for the uniform-loss ablation."""

    if variant == "uniform_loss":
        return {key: 1.0 for key in config.LOSS_WEIGHTS}
    return dict(config.LOSS_WEIGHTS)


def train_tiny_variant(model, train_datasets, criterion, device, args, variant):
    """Train one tiny ablation variant for a capped number of batches."""

    use_balanced_batches = variant != "unbalanced_sampler"
    loader = create_multi_task_dataloader(
        train_datasets,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        use_upsampling=True,
        rank=0,
        world_size=1,
        use_balanced_batches=use_balanced_batches,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())

    train_rows = []
    model.train()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        batches = 0
        started = time.time()

        batch_sampler = getattr(loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)

        for batch_idx, (images, targets) in enumerate(loader, start=1):
            if batch_idx > args.max_train_batches:
                break

            images = images.to(device)
            task_ids = targets["task_id"].to(device)
            for key, value in list(targets.items()):
                if isinstance(value, torch.Tensor):
                    targets[key] = value.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp and torch.cuda.is_available()):
                outputs = model(images, targets, task_ids)
                predictions = predictions_from_outputs(outputs)
                loss, _ = criterion(predictions, targets, task_ids, compute_individual=True)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            batches += 1
            print(f"[{variant}] epoch {epoch} batch {batch_idx}/{args.max_train_batches} loss={loss.item():.4f}")

        train_rows.append(
            {
                "variant": variant,
                "epoch": epoch,
                "train_batches": batches,
                "train_loss": total_loss / max(batches, 1),
                "seconds": time.time() - started,
                "balanced_batches": use_balanced_batches,
            }
        )

    return train_rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Tiny local ablation study runner.")
    parser.add_argument("--dataset-root", default="datasets", help="Repo-local dataset root. Nested datasets/datasets is auto-detected.")
    parser.add_argument("--checkpoint", default=None, help="Optional starting checkpoint. Omit to start from ImageNet/random task heads.")
    parser.add_argument("--allow-partial-checkpoint", action="store_true", help="Load only shape-compatible checkpoint tensors.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TRAIN_TASKS, help="Training tasks for tiny ablation.")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=DEFAULT_VARIANTS)
    parser.add_argument("--max-samples", type=int, default=8, help="Max samples per task dataset.")
    parser.add_argument("--eval-samples", type=int, default=8, help="Max eval samples per task dataset.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows for easier debugging.")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision when available.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/ablation_tiny")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_root = resolve_dataset_root(args.dataset_root)
    output_dir = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Dataset root: {dataset_root}")
    print(f"Device: {device}")
    print(f"Training tasks: {', '.join(args.tasks)}")
    print(f"Variants: {', '.join(args.variants)}")

    train_datasets, train_missing = build_train_datasets(dataset_root, args.tasks, args.max_samples, args.seed)
    eval_datasets, eval_missing = build_eval_datasets(dataset_root, args.tasks, args.eval_samples, args.seed)

    for label, missing in [("train", train_missing), ("eval", eval_missing)]:
        if missing:
            print(f"\n{label} datasets skipped or unavailable:")
            for item in missing:
                print(f"  - {item}")

    if not train_datasets:
        raise RuntimeError("No training datasets could be loaded. Check --dataset-root and --tasks.")
    if not eval_datasets:
        raise RuntimeError("No evaluation datasets could be loaded. Check --dataset-root and --tasks.")

    all_train_rows = []
    all_eval_rows = []

    for variant_idx, variant in enumerate(args.variants):
        print(f"\n{'=' * 80}\nRunning variant: {variant}\n{'=' * 80}")
        set_seed(args.seed + variant_idx)

        model = FaceXFormer().to(device)
        loaded = load_checkpoint_if_available(
            model,
            args.checkpoint,
            device,
            strict=not args.allow_partial_checkpoint,
        )
        apply_variant(model, variant)

        criterion = MultiTaskLoss(loss_weights_for_variant(variant)).to(device)
        train_rows = train_tiny_variant(model, train_datasets, criterion, device, args, variant)
        eval_rows = evaluate_tiny(model, eval_datasets, criterion, device, args.batch_size, args.num_workers)

        for row in eval_rows:
            row["variant"] = variant
            row["checkpoint_loaded"] = loaded

        all_train_rows.extend(train_rows)
        all_eval_rows.extend(eval_rows)

        variant_dir = output_dir / variant
        write_json(variant_dir / "train_summary.json", train_rows)
        write_json(variant_dir / "eval_summary.json", eval_rows)

        if args.checkpoint is None:
            # Save tiny-run weights so we can inspect/resume the smoke run if needed.
            torch.save(
                {
                    "variant": variant,
                    "model_state_dict": model.state_dict(),
                    "train_rows": train_rows,
                },
                variant_dir / "tiny_checkpoint_last.pth",
            )

    write_csv(output_dir / "ablation_train_summary.csv", all_train_rows)
    write_csv(output_dir / "ablation_eval_summary.csv", all_eval_rows)
    write_json(
        output_dir / "ablation_summary.json",
        {
            "dataset_root": str(dataset_root),
            "tasks": args.tasks,
            "variants": args.variants,
            "train": all_train_rows,
            "eval": all_eval_rows,
        },
    )

    print(f"\nSaved ablation outputs under: {output_dir}")
    print("Tiny ablation summary:")
    for row in all_train_rows:
        print(f"  {row['variant']:24s} epoch={row['epoch']} train_loss={row['train_loss']:.4f}")


if __name__ == "__main__":
    main()

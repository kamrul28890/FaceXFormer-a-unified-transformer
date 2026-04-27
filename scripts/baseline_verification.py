"""
Tiny baseline verification for the current FaceXFormer repo.

This script evaluates a small fragment of each dataset using either:
1. a supplied checkpoint, for a real baseline comparison, or
2. random model weights, for a smoke test of the dataset/model/loss pipeline.

The default dataset root resolves to the copied repo dataset path. On this
machine that is usually `datasets/datasets`, with fallback to `datasets`.

Example smoke test:
    python scripts/baseline_verification.py --max-samples 4 --batch-size 2

Example with a checkpoint:
    python scripts/baseline_verification.py --checkpoint checkpoints/best_model.pth --max-samples 32
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
from torch.utils.data import DataLoader

from config import config
from losses import MultiTaskLoss
from network.models.facexformer import FaceXFormer
from scripts.small_run_common import (
    PAPER_TARGETS,
    TASK_IDS,
    build_eval_datasets,
    load_checkpoint_if_available,
    predictions_from_outputs,
    resolve_dataset_root,
    set_seed,
    write_json,
)
from train import (
    compute_accuracy,
    compute_f1_score,
    compute_mae,
    compute_nme,
    compute_recall_at_precision,
)


DEFAULT_TASKS = [
    "segmentation",
    "landmark",
    "headpose",
    "attribute",
    "age",
    "gender",
    "race",
    "visibility",
]


def metric_for_task(task_name, predictions, targets):
    """Compute the paper-style metric used by the current 8-task implementation."""

    if task_name == "segmentation":
        return compute_f1_score(predictions["seg_output"], targets["segmentation"], num_classes=config.SEGMENTATION_CLASSES)
    if task_name == "landmark":
        return compute_nme(predictions["landmark_output"], targets["landmark"])
    if task_name == "headpose":
        return compute_mae(predictions["headpose_output"], targets["headpose"], in_radians=True)
    if task_name == "attribute":
        return compute_accuracy(predictions["attribute_output"], targets["attribute"])
    if task_name == "age":
        return compute_mae(predictions["age_output"], targets["age"].float())
    if task_name == "gender":
        return compute_accuracy(predictions["gender_output"], targets["gender"])
    if task_name == "race":
        return compute_accuracy(predictions["race_output"], targets["race"])
    if task_name == "visibility":
        return compute_recall_at_precision(predictions["visibility_output"], targets["visibility"])
    raise KeyError(f"Unsupported task: {task_name}")


def evaluate_tiny(model, datasets_by_task, criterion, device, batch_size, num_workers):
    """Evaluate tiny subsets one task at a time and return report rows."""

    model.eval()
    rows = []

    with torch.no_grad():
        for task_name, datasets in datasets_by_task.items():
            for dataset in datasets:
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available(),
                )

                total_loss = 0.0
                total_metric = 0.0
                batches = 0
                samples = 0
                started = time.time()

                for images, targets in loader:
                    images = images.to(device)
                    task_ids = torch.full(
                        (images.shape[0],),
                        TASK_IDS[task_name],
                        dtype=torch.long,
                        device=device,
                    )

                    for key, value in list(targets.items()):
                        if isinstance(value, torch.Tensor):
                            targets[key] = value.to(device)
                    targets["task_id"] = task_ids

                    outputs = model(images, targets, task_ids)
                    predictions = predictions_from_outputs(outputs)
                    loss, _ = criterion(predictions, targets, task_ids, compute_individual=True)
                    metric = metric_for_task(task_name, predictions, targets)

                    total_loss += float(loss.item())
                    total_metric += float(metric)
                    batches += 1
                    samples += int(images.shape[0])

                avg_loss = total_loss / max(batches, 1)
                avg_metric = total_metric / max(batches, 1)
                target_info = PAPER_TARGETS.get(task_name, {})
                paper_target = target_info.get("target")
                gap = None if paper_target is None else avg_metric - paper_target

                rows.append(
                    {
                        "task": task_name,
                        "dataset": dataset.get_name() if hasattr(dataset, "get_name") else dataset.__class__.__name__,
                        "samples": samples,
                        "batches": batches,
                        "loss": avg_loss,
                        "metric": avg_metric,
                        "paper_metric": target_info.get("metric", ""),
                        "paper_target": paper_target,
                        "gap_metric_minus_target": gap,
                        "seconds": time.time() - started,
                    }
                )

    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "dataset",
        "samples",
        "batches",
        "loss",
        "metric",
        "paper_metric",
        "paper_target",
        "gap_metric_minus_target",
        "seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Tiny baseline verification for FaceXFormer.")
    parser.add_argument("--dataset-root", default=config.DATASET_ROOT, help=f"Path to dataset root. Defaults to {config.DATASET_ROOT}.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Omit for random-weight smoke test.")
    parser.add_argument("--allow-partial-checkpoint", action="store_true", help="Load only shape-compatible checkpoint tensors.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, choices=DEFAULT_TASKS, help="Tasks to evaluate.")
    parser.add_argument("--max-samples", type=int, default=8, help="Max samples per dataset fragment.")
    parser.add_argument("--batch-size", type=int, default=2, help="Small local eval batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows for easier debugging.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/baseline_tiny")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_root = resolve_dataset_root(args.dataset_root)
    output_dir = Path(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset root: {dataset_root}")
    print(f"Device: {device}")
    print(f"Tasks: {', '.join(args.tasks)}")

    datasets_by_task, missing = build_eval_datasets(dataset_root, args.tasks, args.max_samples, args.seed)
    if missing:
        print("\nDatasets skipped or unavailable:")
        for item in missing:
            print(f"  - {item}")
    if not datasets_by_task:
        raise RuntimeError("No evaluation datasets could be loaded. Check --dataset-root.")

    model = FaceXFormer().to(device)
    loaded = load_checkpoint_if_available(
        model,
        args.checkpoint,
        device,
        strict=not args.allow_partial_checkpoint,
    )
    if loaded:
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint supplied. Running random-weight smoke test; metrics are not meaningful.")

    criterion = MultiTaskLoss(config.LOSS_WEIGHTS).to(device)
    rows = evaluate_tiny(model, datasets_by_task, criterion, device, args.batch_size, args.num_workers)

    csv_path = output_dir / "gap_analysis_baseline_tiny.csv"
    json_path = output_dir / "gap_analysis_baseline_tiny.json"
    write_csv(csv_path, rows)
    write_json(json_path, {"checkpoint_loaded": loaded, "dataset_root": str(dataset_root), "rows": rows})

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print("\nTiny baseline summary:")
    for row in rows:
        print(
            f"  {row['task']:12s} {row['dataset'][:32]:32s} "
            f"n={row['samples']:4d} loss={row['loss']:.4f} metric={row['metric']:.4f}"
        )


if __name__ == "__main__":
    main()

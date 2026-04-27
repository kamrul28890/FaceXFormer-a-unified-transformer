"""
Current 8-task baseline verification for the FaceXFormer repo.

This script evaluates the current 8-task implementation using either:
1. a supplied checkpoint, for a real baseline comparison, or
2. random model weights, for a smoke test of the dataset/model/loss pipeline.

Use `--max-samples` to cap each dataset for a quick smoke test. Use
`--max-samples 0` to evaluate every available sample in each selected split.

The default dataset root resolves to the copied repo dataset path. On this
machine that is usually `datasets/datasets`, with fallback to `datasets`.

Example smoke test:
    python scripts/baseline_current_8task.py --max-samples 4 --batch-size 2

Example with a checkpoint:
    python scripts/baseline_current_8task.py --checkpoint checkpoints/best_model.pth --max-samples 0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
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


def normalize_metric_for_report(task_name, raw_metric, paper_target):
    """
    Keep raw script metrics, but also provide report-ready values.

    The current metric functions return mixed units:
    - segmentation/accuracy/visibility are fractions in [0, 1]
    - landmark NME is already a percentage
    - headpose MAE is computed in radians from this repo's Euler labels
    - age MAE is already in years

    Paper tables report percentages for F1/accuracy/recall, degrees for
    head-pose MAE, NME percentage for landmarks, and years for age.
    """

    if raw_metric is None:
        return None, "", None

    if task_name in {"segmentation", "attribute", "gender", "race", "visibility"}:
        normalized = raw_metric * 100.0
        unit = "percent"
    elif task_name == "headpose":
        normalized = raw_metric * (180.0 / math.pi)
        unit = "degrees"
    elif task_name == "landmark":
        normalized = raw_metric
        unit = "nme_percent"
    elif task_name == "age":
        normalized = raw_metric
        unit = "years"
    else:
        normalized = raw_metric
        unit = "raw"

    normalized_gap = None if paper_target is None else normalized - paper_target
    return normalized, unit, normalized_gap


def metric_for_task(task_name, predictions, targets):
    """Compute the paper-style metric used by the current 8-task implementation."""

    if task_name == "segmentation":
        return compute_f1_score(predictions["seg_output"], targets["segmentation"], num_classes=config.SEGMENTATION_CLASSES)
    if task_name == "landmark":
        return compute_nme(predictions["landmark_output"], targets["landmark"])
    if task_name == "headpose":
        return compute_mae(predictions["headpose_output"], targets["headpose"])
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


def evaluate_current_8task(model, datasets_by_task, criterion, device, batch_size, num_workers):
    """Evaluate selected current 8-task datasets one task at a time."""

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
                normalized_metric, normalized_unit, normalized_gap = normalize_metric_for_report(
                    task_name,
                    avg_metric,
                    paper_target,
                )

                rows.append(
                    {
                        "task": task_name,
                        "dataset": dataset.get_name() if hasattr(dataset, "get_name") else dataset.__class__.__name__,
                        "samples": samples,
                        "batches": batches,
                        "loss": avg_loss,
                        "metric": avg_metric,
                        "gap_metric_minus_target": gap,
                        "raw_metric": avg_metric,
                        "raw_gap_metric_minus_target": gap,
                        "normalized_metric": normalized_metric,
                        "normalized_metric_unit": normalized_unit,
                        "normalized_gap_metric_minus_target": normalized_gap,
                        "paper_metric": target_info.get("metric", ""),
                        "paper_target": paper_target,
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
        "gap_metric_minus_target",
        "raw_metric",
        "raw_gap_metric_minus_target",
        "normalized_metric",
        "normalized_metric_unit",
        "normalized_gap_metric_minus_target",
        "paper_metric",
        "paper_target",
        "seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path):
    if not path or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_manifest(output_dir: Path, args, checkpoint_loaded, dataset_root, rows, csv_path: Path, json_path: Path):
    """Write run metadata beside the baseline outputs."""

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    manifest = {
        "baseline_scope": "current_8task",
        "baseline_scope_note": (
            "Current repo implementation: segmentation, landmark, headpose, "
            "attribute, age, gender, race, visibility. Not the full 10-task "
            "paper baseline."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint": {
            "path": str(checkpoint_path) if checkpoint_path else None,
            "sha256": file_sha256(checkpoint_path) if checkpoint_path else None,
            "size_bytes": checkpoint_path.stat().st_size if checkpoint_path and checkpoint_path.exists() else None,
        },
        "dataset_root": str(dataset_root),
        "tasks": args.tasks,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "result_files": [str(csv_path), str(json_path), str(output_dir / "baseline_current_8task_manifest.json")],
        "num_result_rows": len(rows),
        "total_samples": sum(int(row.get("samples", 0)) for row in rows),
        "metric_columns": {
            "metric": "Original raw script metric, retained for backward compatibility.",
            "raw_metric": "Same value as metric.",
            "normalized_metric": "Report-ready metric in normalized_metric_unit.",
            "normalized_metric_unit": "percent, degrees, nme_percent, years, or raw.",
            "normalized_gap_metric_minus_target": "normalized_metric minus paper_target when paper_target is available.",
        },
        "normalization_rules": {
            "segmentation": "raw F1 fraction * 100 -> percent",
            "attribute_gender_race": "raw accuracy fraction * 100 -> percent",
            "visibility": "raw recall fraction * 100 -> percent",
            "headpose": "raw radians * 180/pi -> degrees",
            "landmark": "raw NME already percent",
            "age": "raw MAE already years",
        },
    }

    manifest_json = output_dir / "baseline_current_8task_manifest.json"
    manifest_md = output_dir / "baseline_current_8task_manifest.md"
    write_json(manifest_json, manifest)

    manifest_md.write_text(
        "# Current 8-Task Baseline Run Manifest\n\n"
        f"Generated: {manifest['generated_at']}\n\n"
        "Scope: current 8-task repo implementation. This is not the full 10-task paper baseline.\n\n"
        f"Checkpoint: `{manifest['checkpoint']['path']}`\n\n"
        f"SHA256: `{manifest['checkpoint']['sha256']}`\n\n"
        f"Dataset root: `{manifest['dataset_root']}`\n\n"
        "Result files:\n\n"
        f"- `{csv_path}`\n"
        f"- `{json_path}`\n"
        f"- `{manifest_json}`\n\n"
        "Metric columns:\n\n"
        "- `metric`: original raw script metric.\n"
        "- `raw_metric`: same as `metric`, explicit name.\n"
        "- `normalized_metric`: report-ready value.\n"
        "- `normalized_metric_unit`: unit for normalized metric.\n"
        "- `normalized_gap_metric_minus_target`: normalized metric minus paper target.\n\n"
        "Normalization rules: segmentation/attribute/gender/race/visibility fractions are multiplied by 100; "
        "headpose radians are converted to degrees; landmark NME remains percent; age MAE remains years.\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Current 8-task baseline verification for FaceXFormer.")
    parser.add_argument("--dataset-root", default="datasets", help="Repo-local dataset root. Nested datasets/datasets is auto-detected.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Omit for random-weight smoke test.")
    parser.add_argument("--allow-partial-checkpoint", action="store_true", help="Load only shape-compatible checkpoint tensors.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, choices=DEFAULT_TASKS, help="Tasks to evaluate.")
    parser.add_argument("--max-samples", type=int, default=8, help="Max samples per dataset. Use 0 for all available samples.")
    parser.add_argument("--batch-size", type=int, default=2, help="Small local eval batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows for easier debugging.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/baseline_current_8task")
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
    rows = evaluate_current_8task(model, datasets_by_task, criterion, device, args.batch_size, args.num_workers)

    csv_path = output_dir / "gap_analysis_baseline_current_8task.csv"
    json_path = output_dir / "gap_analysis_baseline_current_8task.json"
    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "baseline_scope": "current_8task",
            "baseline_scope_note": (
                "Current repo implementation: segmentation, landmark, headpose, "
                "attribute, age, gender, race, visibility. Not the full 10-task "
                "paper baseline."
            ),
            "checkpoint_loaded": loaded,
            "dataset_root": str(dataset_root),
            "rows": rows,
        },
    )
    write_manifest(output_dir, args, loaded, dataset_root, rows, csv_path, json_path)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved manifest: {output_dir / 'baseline_current_8task_manifest.json'}")
    print("\nCurrent 8-task baseline summary:")
    for row in rows:
        print(
            f"  {row['task']:12s} {row['dataset'][:32]:32s} "
            f"n={row['samples']:4d} loss={row['loss']:.4f} "
            f"raw={row['raw_metric']:.4f} normalized={row['normalized_metric']:.4f} {row['normalized_metric_unit']}"
        )


if __name__ == "__main__":
    main()

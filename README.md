# FaceXFormer Reproduction

Local reproduction scaffold for **FaceXFormer: A Unified Transformer for Multi-Task Facial Analysis**.

This repository currently contains an 8-task training/evaluation implementation plus baseline and tiny ablation scripts for local verification before running full experiments on a GPU cluster.

GitHub remote:

```text
https://github.com/preetom-saha-arko/facexformer-main
```

## Current Scope

Implemented in the current codebase:

| Task ID | Task | Dataset(s) | Output |
|---:|---|---|---|
| 0 | Face parsing | CelebAMask-HQ | 11-class mask in current repo |
| 1 | Landmark detection | 300W | 68 x 2 coordinates |
| 2 | Head pose | 300W-LP, BIWI eval | yaw, pitch, roll |
| 3 | Attributes | CelebA, LFWA eval | 40 binary labels |
| 4 | Age | UTKFace, FairFace | 8 age bins |
| 5 | Gender | UTKFace, FairFace | 2 classes |
| 6 | Race | UTKFace, FairFace | 5 classes |
| 7 | Visibility | COFW | 29 visibility scores |

Not fully wired yet:

- Expression recognition: RAF-DB/AffectNet
- Face recognition: MS1MV3/verification sets
- Paper-accurate 142 task-token architecture
- 19-class CelebAMask-HQ segmentation mapping

See [to-dos.md](to-dos.md) for the reproduction roadmap.

## Repository Layout

```text
.
|-- config.py                         # Training/model configuration
|-- datasets.py                       # Dataset classes and multi-task dataloaders
|-- losses.py                         # Multi-task loss
|-- train.py                          # DDP training entry point
|-- train_simple.py                   # Single-GPU training entry point
|-- evaluate.py                       # Checkpoint evaluation
|-- inference.py                      # Inference utilities
|-- network/
|   `-- models/
|       |-- facexformer.py            # Swin-B + FaceX decoder model
|       `-- transformer.py            # Two-way attention decoder blocks
|-- scripts/
|   |-- baseline_current_8task.py     # Current 8-task baseline/eval runner
|   |-- baseline_verification.py      # Backward-compatible wrapper
|   |-- ablation_study_tiny.py        # Tiny ablation train/eval runner
|   `-- small_run_common.py           # Shared run helpers
|-- docs/                             # Original project page assets
|-- requirements.txt                  # Working local requirements
`-- to-dos.md                         # Reproduction checklist
```

Large local-only paths are ignored by Git:

- `.venv/`
- `datasets/`
- `checkpoints/`
- `logs/`
- `runs/`
- `results/`
- model weights such as `*.pth` and `*.pt`

## Environment Setup

Recommended local setup on Windows with an NVIDIA GPU:

```powershell
cd D:\Projects\facexformer-main
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

The current verified local environment used:

```text
Python 3.10.11
Torch 2.5.1+cu121
Torchvision 0.20.1+cu121
CUDA visible on RTX 4070 Ti
```

Quick sanity check:

```powershell
@'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
'@ | python -
```

## Dataset Layout

The baseline and ablation scripts default to:

```text
datasets/
```

If a nested full copy exists, they automatically prefer:

```text
datasets/datasets/
```

Expected task folders include:

```text
datasets/
|-- CelebAMask-HQ/
|-- 300w/
|-- 300W_LP/
|-- CelebA/
|-- UTKFace/
|-- FairFace/
|-- COFW/
|-- BIWI/
|-- LFWA/
`-- RAFDB/
```

The main dataset classes still contain some legacy defaults pointing at `../facexformer-my/datasets`; the tiny scripts pass the repo-local dataset root explicitly.

## Current 8-Task Baseline Verification

Use this to verify dataset loading, model forward pass, loss, and metrics for the current 8-task implementation. Add `--max-samples` for a quick subset run, or use `--max-samples 0` for all available samples in each selected split.

Smoke test without a checkpoint:

```powershell
python scripts\baseline_current_8task.py `
  --tasks landmark `
  --max-samples 4 `
  --batch-size 1 `
  --num-workers 0
```

Broader smoke test:

```powershell
python scripts\baseline_current_8task.py `
  --tasks segmentation landmark headpose attribute `
  --max-samples 4 `
  --batch-size 1 `
  --num-workers 0
```

With a checkpoint:

```powershell
python scripts\baseline_current_8task.py `
  --checkpoint checkpoints\best_model.pth `
  --allow-partial-checkpoint `
  --tasks segmentation landmark headpose attribute age gender race visibility `
  --max-samples 16 `
  --batch-size 2 `
  --num-workers 0
```

Outputs:

```text
results/baseline_current_8task/gap_analysis_baseline_current_8task.csv
results/baseline_current_8task/gap_analysis_baseline_current_8task.json
results/baseline_current_8task/baseline_current_8task_manifest.json
results/baseline_current_8task/baseline_current_8task_manifest.md
```

Without a checkpoint, metrics are not meaningful; that mode is only a pipeline smoke test.

## Tiny Ablation Study

Use this to verify training/evaluation plumbing for small ablation variants before cloud or cluster runs.

First tiny run:

```powershell
python scripts\ablation_study_tiny.py `
  --variants full `
  --tasks segmentation landmark headpose `
  --max-samples 4 `
  --eval-samples 4 `
  --epochs 1 `
  --max-train-batches 1 `
  --batch-size 1 `
  --num-workers 0
```

Compare local variants:

```powershell
python scripts\ablation_study_tiny.py `
  --variants full standard_cross_attention unbalanced_sampler uniform_loss `
  --tasks segmentation landmark headpose `
  --max-samples 8 `
  --eval-samples 8 `
  --epochs 1 `
  --max-train-batches 3 `
  --batch-size 1 `
  --amp `
  --num-workers 0
```

Outputs:

```text
results/ablation_tiny/ablation_train_summary.csv
results/ablation_tiny/ablation_eval_summary.csv
results/ablation_tiny/ablation_summary.json
results/ablation_tiny/<variant>/tiny_checkpoint_last.pth
```

Current variants:

| Variant | Meaning |
|---|---|
| `full` | Current model path |
| `standard_cross_attention` | Disables the extra face-to-task attention direction for a lightweight local approximation |
| `unbalanced_sampler` | Uses the non-balanced sampler path |
| `uniform_loss` | Sets all current task loss weights to 1.0 |

## Full Training

Single GPU:

```powershell
python train_simple.py
```

DDP/multi-GPU:

```powershell
torchrun --nproc_per_node=2 train.py
```

For full paper-scale reproduction, use a cluster. The paper trains with total batch size 384 over 8 GPUs for 12 epochs. This local repo is best used for debugging, reduced ablations, and smoke tests.

## Evaluation

Evaluate a saved checkpoint:

```powershell
python evaluate.py --checkpoint checkpoints\best_model.pth
```

## Development Notes

- Use `.venv` locally, but do not commit it.
- Do not commit datasets, checkpoints, results, logs, or downloaded pretrained weights.
- Keep tiny local outputs under `results/`; this path is ignored.
- Prefer tiny scripts first before launching expensive training.
- The repo currently tracks the 8-task implementation. The 10-task paper reproduction work is planned in [to-dos.md](to-dos.md).

## Known Dependency Notes

The original dependency list included several conda/Linux-specific or legacy packages that do not install cleanly on this Windows CUDA setup:

- `mkl-fft`
- `mkl-random`
- `mkl-service`
- `triton`

`mxnet==1.6.0` is only relevant for possible legacy face-recognition data conversion workflows and has outdated dependency metadata. It is not required for the current 8-task baseline or tiny ablation runners.

## License and Attribution

This project builds on the FaceXFormer architecture and released resources. See [LICENSE](LICENSE) for repository license details. Cite the original FaceXFormer paper when using this reproduction work.

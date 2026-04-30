# FaceXFormer Reproduction Project Overview

## Purpose

This repository contains a reproduction-oriented implementation of FaceXFormer, a unified transformer for multi-task facial analysis. The work began from the public FaceXFormer inference/checkpoint release and reconstructs the missing training and evaluation infrastructure needed for independent reproduction.

The final reproduction scope covers eight tasks:

- Face parsing / segmentation
- Landmark detection
- Head pose estimation
- Attribute prediction
- Age estimation
- Gender classification
- Race classification
- Landmark visibility prediction

Expression recognition and face recognition are paper tasks, but they are outside the completed eight-task reproduction scope in this repository.

## Main Deliverables

- Complete training/evaluation code for the eight-task reproduction.
- Dataset adapters, preprocessing, augmentation, loss functions, and multi-task sampling logic.
- Baseline checkpoint verification scripts and result manifests.
- Preliminary eight-task training results with known bug/suspect rows clearly flagged.
- A polished LaTeX report package in `report_final/`.
- Generated report assets in `report_assets/` and `report_final/assets/`.
- Source paper, proposal, final presentation, and report-writing notes preserved for traceability.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `network/models/` | FaceXFormer model and transformer decoder implementation. |
| `datasets.py` | Dataset adapters, preprocessing, augmentation, and multi-task data handling. |
| `losses.py` | Task-specific losses for segmentation, landmarks, pose, attributes, age, gender, race, and visibility. |
| `train.py` | Distributed and single-node training entry point with balanced multi-task sampling. |
| `evaluate.py` | Evaluation entry point and metric computation. |
| `scripts/baseline_current_8task.py` | Current eight-task baseline verification pipeline. |
| `scripts/baseline_verification.py` | Helper script for checkpoint sanity checks. |
| `scripts/generate_report_assets.py` | Regenerates report figures and visual assets. |
| `results/` | Baseline runs, metric checks, manifests, and small diagnostic outputs. |
| `checkpoints/` | Reproduced / downloaded model checkpoints tracked through Git LFS. |
| `report_assets/` | Standalone generated figures used while composing the report. |
| `report_draft/` | Section-by-section Markdown drafts. |
| `report_final/` | Final LaTeX report package and rendered PDF. |
| `docs/reproduction/` | Project-level documentation for the delivered reproduction. |

## Current Scientific Status

The released checkpoint is broadly validated for segmentation and CelebA attribute prediction after metric normalization. The reproduced eight-task training run is functional, but several rows are not final paper-comparable results:

- Landmark, head pose, and age rows are affected by an identified training/evaluation issue.
- Some gender, race, and visibility rows are marked suspect because they exceed expected/paper values and may reflect protocol or metric differences.
- Baseline and training reports distinguish paper-comparable rows from diagnostic rows.

The main conclusion is that the public FaceXFormer checkpoint supports partial verification, but full independent reproduction requires the complete training pipeline, data protocols, loss weights, and metric normalization rules.

## What Is Not Included

The repository intentionally does not include local Python environments or datasets:

- `.venv/`, `env/`, `venv/`, and related environment folders are excluded.
- `datasets/` is excluded because the raw datasets are too large and are governed by their own licenses/access rules.

Large checkpoint artifacts are tracked with Git LFS rather than normal Git blobs.

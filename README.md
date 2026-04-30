# FaceXFormer-main Training Setup

## Reproduction Documentation Map

This repository now includes a complete reproduction package for the FaceXFormer eight-task reproduction work.

Start here:

- [Project overview](docs/reproduction/PROJECT_OVERVIEW.md)
- [Implementation details](docs/reproduction/IMPLEMENTATION_DETAILS.md)
- [Artifact manifest](docs/reproduction/ARTIFACT_MANIFEST.md)
- [Final report package](report_final/README.md)
- Final compiled PDF: `report_final/main.pdf`

Scope note: local environments and raw datasets are intentionally not tracked. Large checkpoint artifacts are tracked with Git LFS. The final report intentionally omits ablation studies because validated ablation experiments were not completed.

Complete training infrastructure for FaceXFormer-main with multi-GPU support and upsampling strategy.

> **Original Paper**: [_FaceXFormer_ : A Unified Transformer for Facial Analysis](https://kartik-3004.github.io/facexformer/) (ICCV 2025)
> [Project Page](https://kartik-3004.github.io/facexformer/) | [Paper (arXiv)](https://arxiv.org/abs/2403.12960v3) | [Hugging Face Model](https://huggingface.co/kartiknarayan/facexformer)

## Overview

This implementation adapts the original facexformer-main architecture with comprehensive training capabilities:

- ✅ Multi-task co-training on 8 tasks (NO expression recognition)
- ✅ Multi-GPU distributed data parallel (DDP) training
- ✅ Single-GPU training support
- ✅ Upsampling strategy for balanced task representation
- ✅ Automatic GPU memory configuration
- ✅ Task-specific loss weighting
- ✅ Checkpoint saving and resuming

## Installation

> **Note**: A plain `pip install -r requirements.txt` is **not** sufficient. PyTorch requires a CUDA-specific wheel that must be installed via conda before the rest of the dependencies (see Step 3).

### Recommended: Fresh conda environment (new server)

This is the safest approach on a new server. It gives you a clean, isolated environment with the correct CUDA-enabled PyTorch.

> The bundled `environment_facex.yml` pins **CUDA 11.7 + PyTorch 2.0.1**. This works on **any server with CUDA driver ≥ 450**, including servers showing CUDA 12.x in `nvidia-smi` — CUDA drivers are backward-compatible. PyTorch also bundles its own CUDA 11.7 runtime internally, so no system-level CUDA 11.7 toolkit is required.
>
> The manual steps below are only needed if you want a newer PyTorch version (e.g. to use features introduced after 2.0.1).

#### Step 1 — Check your server's CUDA driver version

```bash
nvidia-smi          # top-right corner shows "CUDA Version: X.Y"
nvcc --version      # shows the toolkit version (may differ from driver)
```

The driver version determines the maximum CUDA toolkit you can use. Common pairings:

| Driver version | Max CUDA toolkit |
|---|---|
| ≥ 570 | CUDA 12.8 |
| ≥ 525 | CUDA 12.1 |
| ≥ 520 | CUDA 11.8 |
| ≥ 450 | CUDA 11.7 |

> **Note**: A newer driver is always compatible with older CUDA runtimes. A server showing `CUDA 12.8` in `nvidia-smi` can run CUDA 11.7, 11.8, 12.1, or 12.8 PyTorch builds.

#### Step 2 — Create the conda environment

```bash
conda create -n facexformer-main python=3.10 -y
conda activate facexformer-main
```

#### Step 3 — Install CUDA-enabled PyTorch

Pick the command that matches your driver (from Step 1):

```bash
# CUDA 12.1 (driver ≥ 525) — recommended for modern servers
# Pins pytorch==2.4.0 and mkl<2025 to avoid MKL 2025 iJIT_NotifyEvent symbol error
conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 "mkl<2025" \
    -c pytorch -c nvidia -c defaults -y

# CUDA 11.8 (driver ≥ 520)
conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=11.8 "mkl<2025" \
    -c pytorch -c nvidia -c defaults -y

# CUDA 11.7 (driver ≥ 450) — matches the bundled environment_facex.yml
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 "mkl<2025" \
    -c pytorch -c nvidia -c defaults -y
```

> **Why pin versions?** Unpinned `conda install pytorch` currently resolves to PyTorch 2.5.1 whose conda package cache entry has a known corruption issue (missing `__pycache__` files). Pinning `pytorch==2.4.0` sidesteps this. The `"mkl<2025"` pin prevents MKL 2025+ from being installed, which removed `libittnotify.so` and causes `iJIT_NotifyEvent: undefined symbol` errors at import time.

> If your cluster uses modules (e.g. `module load cuda/12.1`), load the CUDA module **before** running this step.

#### Step 4 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

> **Important**: `torch` and `torchvision` are **commented out** in `requirements.txt`. This is intentional — pip and conda share the same `site-packages` directory, so an uncommented `torch>=2.0.0` line would cause pip to uninstall conda's CUDA-enabled torch and replace it with a CPU-only pip wheel. Do not uncomment those lines.

#### Step 5 — Verify the installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
python test_setup.py
```

Both `torch.cuda.is_available()` should return `True` and all `test_setup.py` checks should pass before starting training.

---

### Alternative: Reproduce exact environment from `environment_facex.yml`

Use this if you want an exact replica of the original environment (CUDA 11.7, PyTorch 2.0.1, Python 3.10). Works on any server with **CUDA driver ≥ 450** — including modern servers showing CUDA 12.x in `nvidia-smi`:

```bash
conda env create -f environment_facex.yml
conda activate facexformer-main
pip install -r requirements.txt
```

---

### Notes on specific packages

- **`mkl-fft`, `mkl-random`, `mkl-service`** — Intel MKL wrappers. These install cleanly on x86 servers but will fail on ARM/POWER nodes (e.g. GH200). If you are on a non-x86 node, remove those three lines from `requirements.txt` before Step 4 — PyTorch bundles its own math libraries.
- **`typing-extensions`** — pinned to `>=4.8.0` in `requirements.txt`. PyTorch 2.4.0 imports `typing_extensions.deprecated` which was added in 4.4.0; the original repo pinned `==4.3.0` which causes `ImportError: cannot import name 'deprecated'` at torch import time.
- **`triton`** — commented out in `requirements.txt`; conda's pytorch package already installs the correct triton version. Installing the pip version would downgrade it and break torch.
- **`mxnet`** — historical dependency; not imported by any code in this repo. Commented out in `requirements.txt`.
- **`facenet-pytorch`** — only required for `inference.py` (MTCNN face detection). Not needed for training.

---

## Setup Completion Summary

### ✅ Completed Tasks

**Dataset Configuration:**
- ✅ All dataset paths configured to point to `config.DATASET_ROOT` everywhere in the codebase
- ✅ CelebA cache files cleared for regeneration
- ✅ Dataset loading progress messages added
- ✅ Config print spam fixed with worker detection
- ✅ Proper train/test partition implemented with 3 test-only datasets (300VW, BIWI, LFWA)
- ✅ 300VW uses correct test sequences (categories 1,2,3)

**Training Infrastructure:**
- ✅ Multi-GPU DDP training with DistributedDataParallel (batch job and standalone torchrun modes)
- ✅ Balanced task distribution in each batch for each GPU
- ✅ BalancedMultiTaskBatchSampler implemented for exact task balance
- ✅ UpsampledMultiTaskDataset balances all datasets to 122,450 samples
- ✅ Multi-task collate function with tensor conversion and dummy values
- ✅ Loss function handles all task combinations without KeyErrors
- ✅ All task labels have key existence checks
- ✅ Gradient flow issues resolved in loss computation
- ✅ Visibility shape fixed to [29] values instead of [1] average

**Metrics & Evaluation:**
- ✅ NME calculation properly normalized by inter-ocular distance (IOD)
- ✅ Age MAE converted from logits to continuous years using bin centers
- ✅ Headpose MAE correctly handles radians to degrees conversion with `in_radians=True`
- ✅ Visibility occlusion metric class targets inverted to match paper precision/recall
- ✅ Unified metric computation applied across all validation and scripts

**Configuration:**
- ✅ Config auto-configuration based on GPU memory (70GB+: 96 batch, etc.)
- ✅ Manual override section for custom batch sizes
- ✅ Unnecessary config variables removed while preserving multi-GPU settings
- ✅ DIST_BACKEND='nccl' for optimal multi-GPU performance

### Key Features Implemented

**Multi-GPU Training:**
- PyTorch DDP with NCCL backend for optimal performance
- BalancedMultiTaskBatchSampler ensures exact task balance within each batch across all GPUs
- Random assignment of extra samples prevents bias
- Epoch-wise shuffling for data diversity

**Dataset Upsampling Strategy:**
- All datasets upsampled to 122,450 samples for stable training
- Maintains original data distribution while ensuring sufficient samples per task
- Prevents training instability from small datasets

**Loss Function:**
- Multi-task loss with proper gradient handling
- Key existence checks for all 8 tasks
- Task-specific weighting and proper loss accumulation
- Handles missing labels gracefully

**Data Loading:**
- Custom collate function for mixed-task batches
- Tensor conversion for all data types
- Dummy tensors match actual model output shapes
- Visibility prediction returns 29 individual scores

### Architecture Details

- **Model**: FaceXFormer with Swin-Base backbone
- **Tasks**: 8 tasks (segmentation, landmark, headpose, attribute, age, gender, race, visibility)
- **Task Tokens**: 18 total (7 single + 11 mask tokens)
- **Parameters**: ~92M
- **Training**: Multi-GPU DDP with balanced batch sampling

### Testing & Verification

**Setup Verification:**
```bash
# Test dataset loading
python test_setup.py

# Test single-GPU training (dry run)
python train.py --dry-run

# Test multi-GPU training (if available)
torchrun --nproc_per_node=2 train.py --dry-run
```

**Expected Behavior:**
- All datasets load successfully from `../facexformer-my/datasets/`
- Balanced batches contain equal representation of all tasks
- Loss computation works without KeyErrors
- Multi-GPU setup initializes correctly

### Next Steps

1. **Verify Setup**: Run `python test_setup.py` to confirm all components work
2. **Single-GPU Training**: Start with single GPU to verify training loop
3. **Multi-GPU Scaling**: Scale to multiple GPUs once single-GPU works
4. **Monitor Training**: Watch for balanced task loss convergence
5. **Checkpoint Management**: Use automatic checkpoint saving/resuming

## Excluded Task

**Facial Expression Recognition** has been completely removed from this implementation as requested:
- ❌ RAF-DB dataset
- ❌ AffectNet dataset
- ❌ Expression loss function
- ❌ Expression task tokens

## Supported Tasks

| Task ID | Task Name | Datasets | Output Dimension |
|---------|-----------|----------|------------------|
| 0 | Face Parsing | CelebAMask-HQ | 11 classes |
| 1 | Landmark Detection | 300W | 136 (68×2) |
| 2 | Head Pose Estimation | 300W-LP | 3 (Euler angles) |
| 3 | Attributes Prediction | CelebA | 40 binary attributes |
| 4 | Age Estimation | UTKFace, FairFace | 8 age groups |
| 5 | Gender Classification | UTKFace, FairFace | 2 classes |
| 6 | Race Classification | UTKFace, FairFace | 5 classes |
| 7 | Visibility Prediction | COFW | 29 visibility scores |

## Architecture

FaceXFormer-main uses:
- **Backbone**: Swin Transformer-Base (pretrained on ImageNet)
- **Decoder**: TwoWayTransformer (SAM-style, 2 blocks)
- **Task Tokens**: 18 total (7 single tokens + 11 mask tokens)
  - 1 landmark token → MLP → 136 coords
  - 1 pose token → MLP → 3 angles
  - 1 attribute token → MLP → 40 attributes
  - 1 visibility token → MLP → 29 scores
  - 1 age token → MLP → 8 groups
  - 1 gender token → MLP → 2 classes
  - 1 race token → MLP → 5 classes
  - 11 mask tokens → hypernetwork → 11-class segmentation

**Total Parameters**: ~92M (91,985,943)

## Dataset Setup

### Dataset Location

**IMPORTANT**: This implementation uses datasets from the `facexformer-my/datasets` folder. All dataset paths are configured to point to `../facexformer-my/datasets/` by default.

### Required Datasets

The following datasets should already be available in `../facexformer-my/datasets/`:

1. **CelebAMask-HQ** (Face Parsing)
   - Location: `../facexformer-my/datasets/CelebAMask-HQ/`
   - Train: 28,000 | Test: 2,000

2. **300W** (Landmarks)
   - Location: `../facexformer-my/datasets/300w/`
   - Train: ~3,148 | Test: ~689

3. **300W-LP** (Head Pose)
   - Location: `../facexformer-my/datasets/300W_LP/`
   - Train: 122,450

4. **CelebA** (Attributes)
   - Location: `../facexformer-my/datasets/CelebA/`
   - Requires: `img_align_celeba/`, `list_attr_celeba.csv`, `list_eval_partition.csv`
   - Train: 162,770 | Test: 19,867

5. **UTKFace** (Age/Gender/Race)
   - Location: `../facexformer-my/datasets/UTKFace/`
   - Total: ~23,372 (split 85/15)

6. **FairFace** (Age/Gender/Race)
   - Location: `../facexformer-my/datasets/FairFace/extracted/`
   - Train: 76,125 | Val: 10,875

7. **COFW** (Visibility)
   - Location: `../facexformer-my/datasets/COFW/`
   - Train: 1,345 | Test: 507

### Expected Directory Structure

```
facexformer-my/datasets/
├── CelebAMask-HQ/
│   ├── CelebA-HQ-img/
│   └── CelebAMask-HQ-mask-anno/
├── 300w/
│   └── 300W/
│       ├── 01_Indoor/
│       └── 02_Outdoor/
├── 300W_LP/
│   ├── AFW/
│   ├── AFW_Flip/
│   ├── HELEN/
│   └── ...
├── CelebA/
│   ├── img_align_celeba/
│   ├── list_attr_celeba.csv
│   └── list_eval_partition.csv
├── UTKFace/
│   └── [age]_[gender]_[race]_[date].jpg
├── FairFace/
│   └── extracted/
│       ├── train/
│       ├── val/
│       ├── train_labels.csv
│       └── val_labels.csv
└── COFW/
    ├── COFW_train.mat
    └── COFW_test.mat
```

## Training

### Single GPU Training

For training on a single GPU:

```bash
python train_simple.py
```

### Multi-GPU Training (no job scheduler)

If your server has multiple GPUs but no SLURM or PBS queue, use `torchrun --standalone`. It launches all GPU processes itself with no external scheduler needed.

#### Basic usage

```bash
# All 8 GPUs on the machine
torchrun --standalone --nproc_per_node=8 train.py

# Specific number of GPUs (e.g. 4)
torchrun --standalone --nproc_per_node=4 train.py

# Select specific GPU devices (e.g. GPUs 0,1,2,3 out of 8)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py
```

`--standalone` tells torchrun to act as its own rendezvous server — no `MASTER_ADDR` or `MASTER_PORT` setup required.

#### Keep the job running after SSH disconnect

Without a job scheduler you must protect the process from terminal disconnection. Use either:

**tmux (recommended):**
```bash
tmux new -s train          # create a named session
torchrun --standalone --nproc_per_node=8 train.py
# Ctrl+B, then D to detach — job keeps running
tmux attach -t train       # reattach later to check progress
```

**screen:**
```bash
screen -S train
torchrun --standalone --nproc_per_node=8 train.py
# Ctrl+A, then D to detach
screen -r train            # reattach
```

**nohup (no terminal multiplexer needed):**
```bash
nohup torchrun --standalone --nproc_per_node=8 train.py \
    > logs/train.log 2>&1 &
echo $!                    # prints the PID — save it to kill later if needed
tail -f logs/train.log     # follow the log in real time
```

#### Ablation study (no job scheduler)

```bash
# Single variant, 8 GPUs, foreground (inside tmux/screen)
torchrun --standalone --nproc_per_node=8 ablation_study.py --variant full

# All five variants back-to-back (each waits for the previous to finish)
for variant in full standard_cross_attention unbalanced_sampler uniform_loss no_augmentation; do
    echo "=== Running variant: $variant ==="
    torchrun --standalone --nproc_per_node=8 ablation_study.py --variant $variant
done

# Run all four in parallel on different GPU sets (2 GPUs each on an 8-GPU server)
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    ablation_study.py --variant full &
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 \
    ablation_study.py --variant standard_cross_attention &
CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 \
    ablation_study.py --variant unbalanced_sampler &
CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 \
    ablation_study.py --variant uniform_loss &
wait
```

> When running multiple jobs in parallel on different GPU sets, each needs a distinct `MASTER_PORT` to avoid port conflicts:
> ```bash
> CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --master_port=29500 --nproc_per_node=2 ablation_study.py --variant full &
> CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --master_port=29501 --nproc_per_node=2 ablation_study.py --variant standard_cross_attention &
> ```

### Resume Training

```bash
# Single GPU
python train_simple.py --resume ./checkpoints/checkpoint_epoch_10.pth

# Multi-GPU
torchrun --standalone --nproc_per_node=8 train.py --resume ./checkpoints/checkpoint_epoch_10.pth
```

## Multi-Task Co-Training Strategy

### Upsampling for Stable Multi-Dataset Training

This implementation uses an **upsampling strategy** to ensure stable training across tasks with varying dataset sizes. The `UpsampledMultiTaskDataset` class balances all datasets by:

1. **Identifying the largest dataset** (300W-LP with 122,450 samples)
2. **Upsampling smaller datasets** to match the largest:
   - COFW: 1,345 → 122,450 (91× upsampling)
   - 300W: 3,148 → 122,450 (39× upsampling)
   - CelebAMask-HQ: 28,000 → 122,450 (4× upsampling)
3. **Shuffling each epoch** to prevent overfitting on repeated samples
4. **Ensuring balanced task representation** in each batch

This approach guarantees:
- ✅ All tasks receive equal training attention
- ✅ No task dominates the training process
- ✅ Stable gradient updates across all tasks
- ✅ Effective multi-GPU training with DDP

### Batch Composition

Each batch contains samples from multiple tasks:
- Task IDs are used to filter outputs during loss computation
- Only relevant predictions are used for each sample
- Efficient GPU utilization across all tasks

## Data Augmentation

Task-specific augmentation pipelines are implemented in [datasets.py](datasets.py). All augmentations are applied **only during training**; evaluation uses clean resized images. ImageNet normalization (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) is applied to all tasks.

### Landmark Detection (300W)

ArcFace face alignment is applied first: 5 keypoints are derived from the 68-point annotations and used to estimate a partial affine transform that aligns the face to a canonical template (scaled from the 112×112 ArcFace template to the target resolution). Geometric augmentations are then applied jointly to the image and landmarks so they remain consistent.

| Augmentation | Parameters | Probability |
|---|---|---|
| ArcFace alignment | 5-point template (always applied) | 100% |
| Rotation | ±18° around image centre | 100% (sampled) |
| Scaling | 0.9–1.1× around centre | 100% (sampled) |
| Translation | ±5% of image size | 100% (sampled) |
| Horizontal flip | Landmark indices reordered via `_FLIP_INDICES_68` | 50% |
| Grayscale | Convert to gray and back to RGB | 20% |
| Gaussian blur | Radius 0.5–2.0 px | 30% |
| Random occlusion | Random-colour rectangle, 10–40% of image size | 40% |
| Gamma correction | γ ∈ [0.7, 1.3] | 20% |

### Head Pose Estimation (300W-LP)

| Augmentation | Parameters | Probability |
|---|---|---|
| Random resized crop | 80–100% of image, re-scaled to target size | 100% (sampled) |
| Grayscale | Convert to gray and back to RGB | 10% |
| Gaussian blur | Radius 0.5–1.5 px | 10% |
| Gamma correction | γ ∈ [0.7, 1.3] | 10% |

### Attribute Prediction (CelebA)

| Augmentation | Parameters | Probability |
|---|---|---|
| Rotation | ±18° | 100% (sampled) |
| Scaling | 0.9–1.1× around centre | 100% (sampled) |
| Translation | ±1% of image size | 100% (sampled) |
| Horizontal flip | Mirror image | 50% |
| Grayscale | Convert to gray and back to RGB | 10% |
| Gaussian blur | Radius 0.5–1.5 px | 10% |
| Gamma correction | γ ∈ [0.7, 1.3] | 20% |

### Age / Gender / Race (UTKFace, FairFace)

| Augmentation | Parameters | Probability |
|---|---|---|
| Rotation | ±18° | 100% (sampled) |
| Scaling | 0.9–1.1× around centre | 100% (sampled) |
| Translation | ±1% of image size | 100% (sampled) |
| Horizontal flip | Mirror image | 50% |
| Grayscale | Convert to gray and back to RGB | 10% |
| Gaussian blur | Radius 0.5–1.5 px | 10% |
| Gamma correction | γ ∈ [0.7, 1.3] | 10% |

### Visibility Prediction (COFW)

Horizontal flip reorders the 29 visibility labels according to the symmetric `_COFW_FLIP_IDX` mapping so that left/right landmark labels remain correct after mirroring.

| Augmentation | Parameters | Probability |
|---|---|---|
| Horizontal flip | Visibility labels reordered via `_COFW_FLIP_IDX` | 50% |
| Grayscale | Convert to gray and back to RGB | 10% |
| Gaussian blur | Radius 0.5–1.5 px | 10% |
| Gamma correction | γ ∈ [0.7, 1.3] | 10% |

### Face Parsing (CelebAMask-HQ)

No random augmentations are applied. Images are resized to the target resolution and normalized with ImageNet statistics.

### Summary

| Task | Rotation | Scale | Translate | Flip | Grayscale | Blur | Occlusion | Gamma | ArcFace Align |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Landmark Detection | ✅ ±18° | ✅ ±10% | ✅ ±5% | ✅ 50% | ✅ 20% | ✅ 30% | ✅ 40% | ✅ 20% | ✅ |
| Head Pose | ❌ | ❌ | ✅ crop | ❌ | ✅ 10% | ✅ 10% | ❌ | ✅ 10% | ❌ |
| Attributes | ✅ ±18° | ✅ ±10% | ✅ ±1% | ✅ 50% | ✅ 10% | ✅ 10% | ❌ | ✅ 20% | ❌ |
| Age/Gender/Race | ✅ ±18° | ✅ ±10% | ✅ ±1% | ✅ 50% | ✅ 10% | ✅ 10% | ❌ | ✅ 10% | ❌ |
| Visibility | ❌ | ❌ | ❌ | ✅ 50% | ✅ 10% | ✅ 10% | ❌ | ✅ 10% | ❌ |
| Face Parsing | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

## Configuration

Edit [config.py](config.py) to adjust training settings:

```python
# Batch size (per GPU)
BATCH_SIZE = 96  # Auto-configured based on GPU memory

# Learning rate (scaled based on batch size)
LEARNING_RATE = 2e-4

# Training epochs
NUM_EPOCHS = 12

# Learning rate decay
LR_DECAY_EPOCHS = [6, 10]  # Decay at epochs 6 and 10
LR_DECAY_FACTOR = 0.1      # Decay by 10x

# Loss weights
LOSS_WEIGHTS = {
    'seg': 1.0,    # Segmentation
    'ind': 1.0,    # Landmarks
    'hpe': 1.0,    # Head pose
    'attr': 1.0,   # Attributes
    'a': 1.0,      # Age
    'g/r': 1.0,    # Gender/Race
    'vis': 1.0     # Visibility
}
```

## GPU Memory Auto-Configuration

The training script automatically adjusts batch size based on available GPU memory:

| GPU Memory | Batch Size | Learning Rate | Gradient Checkpointing |
|------------|------------|---------------|------------------------|
| ≥70 GB     | 96         | 2.5e-5        | No                     |
| ≥40 GB     | 64         | 1.67e-5       | No                     |
| ≥20 GB     | 32         | 8.33e-6       | No                     |
| ≥10 GB     | 16         | 4.17e-6       | Yes                    |
| ≥6 GB      | 4          | 1.04e-6       | Yes                    |
| <6 GB      | 2          | 5.2e-7        | Yes                    |

## File Structure

```
facexformer-main/
├── README.md                    # This file
├── config.py                    # Configuration settings
├── datasets.py                  # Dataset implementations with upsampling
├── losses.py                    # Multi-task loss functions
├── train.py                     # Multi-GPU training script (DDP)
├── train_simple.py              # Single-GPU training script
├── ablation_study.py            # Full-dataset ablation study (multi-GPU, DDP)
├── submit_job.slurm             # SLURM job script for main training
├── submit_ablation.slurm        # SLURM job script for full ablation study
├── test_setup.py                # Verification script
├── scripts/
│   ├── small_run_common.py      # Shared utilities for tiny smoke-test runs
│   ├── baseline_current_8task.py # Current 8-task baseline evaluation
│   ├── baseline_verification.py # Backward-compatible wrapper
│   └── ablation_study_tiny.py  # Tiny ablation smoke-test (multi-GPU, DDP)
├── network/
│   └── models/
│       ├── facexformer.py       # Main model
│       └── transformer.py      # TwoWayTransformer decoder
├── checkpoints/                 # Saved model checkpoints
└── logs/                        # Training logs
```

## Monitoring Training

### Checkpoints

Checkpoints are saved to `./checkpoints/`:
- `best_model.pth`: Model with lowest validation loss
- `checkpoint_epoch_N.pth`: Periodic checkpoints (every 2 epochs)

Each checkpoint contains:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'scheduler_state_dict': OrderedDict,
    'loss': float
}
```

### Loss Tracking

Training output includes:
- Overall loss per epoch
- Individual task losses (when available)
- Learning rate
- Progress bar with current batch loss
- Training time per epoch

## Troubleshooting

### pip replaced conda torch (ImportError or wrong version)

Symptom: after `pip install -r requirements.txt`, `torch.__version__` shows `2.1.2` instead of `2.4.0`, or you get `ImportError: cannot import name 'deprecated' from 'typing_extensions'`.

Cause: pip and conda share the same `site-packages`. Any uncommented `torch` line in `requirements.txt` (or a transitive dependency that pulls torch) will uninstall the conda torch and replace it with a pip wheel. Once pip removes the files, conda's package metadata becomes stale — it still thinks torch is installed, so `conda install pytorch==2.4.0` will say "already installed" without fixing anything.

Fix:
```bash
# 1. Remove the pip-installed torch
pip uninstall -y torch torchvision triton

# 2. Force-reinstall via conda (--force-reinstall bypasses the stale metadata)
conda install --force-reinstall pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 "mkl<2025" \
    -c pytorch -c nvidia -c defaults -y

# 3. Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
# Expected: 2.4.0 True 12.1
```

> Always confirm `torch` and `torchvision` remain commented out in `requirements.txt` before re-running `pip install -r requirements.txt`.

---

### Dataset Not Found

If you encounter dataset errors, verify the paths:

```python
# Test dataset loading
python -c "from datasets import CelebAMaskHQDataset; print('CelebAMask-HQ OK')"
python -c "from datasets import W300Dataset; print('300W OK')"
python -c "from datasets import W300LPDataset; print('300W-LP OK')"
```

If datasets are not found, ensure:
1. The `facexformer-my/datasets/` folder exists relative to `facexformer-main/`
2. All datasets are properly extracted
3. Dataset paths in [datasets.py](datasets.py) are correct

### Out of Memory (OOM)

1. **Reduce batch size** in [config.py](config.py):
   ```python
   BATCH_SIZE = 4  # Reduce from default
   ```

2. **Enable gradient checkpointing**:
   ```python
   GRADIENT_CHECKPOINTING = True
   ```

3. **Use single-GPU training** instead of multi-GPU:
   ```bash
   python train_simple.py
   ```

### Multi-GPU Issues

1. **Check NCCL backend** is available:
   ```python
   import torch.distributed as dist
   print(dist.is_nccl_available())  # Should be True
   ```

2. **Set environment variables** for debugging:
   ```bash
   # Windows (PowerShell)
   $env:NCCL_DEBUG="INFO"
   $env:TORCH_DISTRIBUTED_DEBUG="DETAIL"

   # Linux
   export NCCL_DEBUG=INFO
   export TORCH_DISTRIBUTED_DEBUG=DETAIL
   ```

3. **Ensure all GPUs are visible**:
   ```python
   import torch
   print(f"Available GPUs: {torch.cuda.device_count()}")
   ```

### Loss Function Architecture Compatibility

**Important**: The loss function is designed to work with the model's internal task filtering:

- The model outputs **filtered tensors** containing only samples relevant to each task
- The loss function extracts corresponding labels using `task_ids`
- Do NOT modify the model architecture without understanding this interaction

If you see index errors during training, ensure:
1. The model architecture has not been modified
2. Task IDs in [config.py](config.py) match the model's expectations
3. The loss function in [losses.py](losses.py) has not been altered

## Differences from facexformer-my

| Aspect | facexformer-main | facexformer-my |
|--------|------------------|----------------|
| Decoder | TwoWayTransformer (SAM-style) | FaceXDecoder (custom) |
| Task Tokens | 18 fixed embeddings | 103 learnable parameters |
| Unified Head | ❌ Not present | ✅ Present |
| Landmark Head | Simple MLP | Hourglass network |
| Head Pose Output | 3D (Euler angles) | 9D (rotation matrix) |
| Expression Task | ❌ Excluded | ✅ Included |
| Dataset Upsampling | ✅ Implemented | ✅ Implemented |

## Testing the Setup

Run the verification script to ensure everything is configured correctly:

```bash
python test_setup.py
```

This will test:
1. ✅ Configuration loading
2. ✅ Model initialization
3. ✅ Forward pass
4. ✅ Backward pass
5. ✅ Dataset loading
6. ✅ Loss computation

All tests should pass before starting training.

## scripts/ Folder

The `scripts/` directory contains three utilities for **pipeline verification and smoke testing** before committing to a full training run. They are not part of the main training pipeline.

### `scripts/small_run_common.py` — Shared utilities

Provides helpers used by both other scripts:

| Symbol | Purpose |
|---|---|
| `TASK_IDS` | Maps each of the 8 task names to their integer ID |
| `PAPER_TARGETS` | Published benchmark targets (e.g. segmentation F1 → 92.01) for gap analysis |
| `NamedSubset` | Wraps a `Dataset` and randomly draws a fixed-size fragment |
| `build_train_datasets` | Constructs tiny train dataset fragments for all 8 tasks |
| `build_eval_datasets` | Constructs tiny eval dataset fragments for all 8 tasks |
| `load_checkpoint_if_available` | Loads a `.pth` checkpoint with DDP prefix stripping; supports partial/shape-compatible loading |
| `predictions_from_outputs` | Unpacks the 8-tuple returned by `FaceXFormer.forward()` into a named dict |
| `resolve_dataset_root` | Resolves the dataset root, handling the `../facexformer-my/datasets` default |

### `scripts/baseline_current_8task.py` - Current 8-task baseline check

Evaluates selected dataset splits and prints per-task metrics alongside paper targets. Two modes:

- **Smoke test** (no checkpoint): loads random weights, verifies the data → model → loss → metric chain does not crash.
- **Baseline comparison** (with `--checkpoint`): loads trained weights and reports normalized metrics plus paper gaps where the dataset/protocol matches.

`scripts/baseline_verification.py` is kept as a backward-compatible wrapper for older commands.

```bash
# Smoke test (no checkpoint required)
python scripts/baseline_current_8task.py --max-samples 4 --batch-size 2

# Baseline comparison with a checkpoint
python scripts/baseline_current_8task.py --checkpoint checkpoints/best_model.pth --max-samples 0
```

Output is saved to `results/baseline_current_8task/`.

### `scripts/ablation_study_tiny.py` — Tiny ablation smoke-test

Trains and evaluates the same 4 ablation variants as the full study, but on tiny dataset fragments for a handful of batches. Used to verify the ablation machinery end-to-end before running `ablation_study.py` on the full dataset. Supports multi-GPU DDP via `torchrun` (same setup as `train.py`).

```bash
# Smoke test all variants (single GPU, 2 batches, 1 epoch)
python scripts/ablation_study_tiny.py \
    --variants full standard_cross_attention unbalanced_sampler uniform_loss no_augmentation \
    --max-samples 8 --max-train-batches 2 --epochs 1

# Larger tiny run on the cluster (submit via submit_ablation.slurm with small settings)
python scripts/ablation_study_tiny.py \
    --variants standard_cross_attention \
    --max-samples 64 --max-train-batches 50 --epochs 5 --amp
```

Output is saved to `results/ablation_tiny/<variant>/`.

---

## Ablation Study

The ablation study isolates the contribution of three design choices in FaceXFormer-main:

### Variants

| Variant | What changes | Purpose |
|---|---|---|
| `full` | Nothing — unmodified model | Baseline for comparison |
| `standard_cross_attention` | `cross_attn_image_to_token` replaced with `ZeroAttention` (zero output) in every transformer layer | Tests the value of bidirectional cross-attention vs. standard one-directional cross-attention |
| `unbalanced_sampler` | `use_balanced_batches=False` in the dataloader | Tests the value of the balanced-batch sampler |
| `uniform_loss` | All task loss weights set to 1.0 instead of `config.LOSS_WEIGHTS` | Tests the value of the paper's tuned per-task loss weighting |
| `no_augmentation` | Data augmentation disabled (`is_train=False`) | Tests the value of data augmentation in the training pipeline |

All variants use the same datasets, augmentation pipeline, optimizer, and learning rate schedule as the main training run.

### Full ablation study (`ablation_study.py`)

Runs a single variant on the **complete dataset** with the same distributed training setup as `train.py`. Submit one SLURM job per variant.

```bash
# Submit all five variants as separate jobs
ABLATION_VARIANT=full                     sbatch submit_ablation.slurm
ABLATION_VARIANT=standard_cross_attention sbatch submit_ablation.slurm
ABLATION_VARIANT=unbalanced_sampler       sbatch submit_ablation.slurm
ABLATION_VARIANT=uniform_loss             sbatch submit_ablation.slurm
ABLATION_VARIANT=no_augmentation          sbatch submit_ablation.slurm
```

The `--variant` argument can also be passed directly when running without SLURM:

```bash
torchrun --standalone --nproc_per_node=4 ablation_study.py --variant standard_cross_attention
```

Checkpoints are saved to `results/ablation_full/<variant>/checkpoints/` and final test results to `results/ablation_full/<variant>/test_results.json`.

### SLURM configuration (`submit_ablation.slurm`)

Mirrors `submit_job.slurm` exactly (4 nodes, 1 GPU per node, `mpirun` + `torchrun`, NCCL backend). Uses `MASTER_PORT=29502` to avoid conflicts with the main training job (29500) and the tiny ablation script (29501).

### Recommended workflow

1. **Verify pipeline first** — run a smoke test with the tiny script:
   ```bash
   python scripts/ablation_study_tiny.py --variants full --max-samples 8 --max-train-batches 2 --epochs 1
   ```
2. **Run full ablation** — once smoke test passes, submit the SLURM jobs.
3. **Compare results** — compare `test_results.json` across the four variant output directories.

## Citation

If you use this code, please cite the original FaceXFormer paper:

```bibtex
@article{narayan2024facexformer,
  title={FaceXFormer: A Unified Transformer for Facial Analysis},
  author={Narayan, Kartik and VS, Vibashan and Chellappa, Rama and Patel, Vishal M},
  journal={arXiv preprint arXiv:2403.12960},
  year={2024}
}
```

## License

This implementation follows the license of the original facexformer-main repository.

## Support

For issues or questions:
1. Check this README
2. Review [config.py](config.py) settings
3. Run [test_setup.py](test_setup.py) to verify the setup
4. Ensure datasets are accessible at `../facexformer-my/datasets/`
5. Check GPU memory and adjust batch size accordingly

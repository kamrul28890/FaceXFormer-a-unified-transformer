# FaceXFormer-main Training Setup

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

## Setup Completion Summary

### ✅ Completed Tasks

**Dataset Configuration:**
- ✅ All dataset paths configured to point to `../facexformer-my/datasets/`
- ✅ CelebA cache files cleared for regeneration
- ✅ Dataset loading progress messages added
- ✅ Config print spam fixed with worker detection
- ✅ Proper train/test partition implemented with 3 test-only datasets (300VW, BIWI, LFWA)
- ✅ 300VW uses correct test sequences (categories 1,2,3)

**Training Infrastructure:**
- ✅ Multi-GPU DDP training with DistributedDataParallel
- ✅ Balanced task distribution in each batch for each GPU
- ✅ BalancedMultiTaskBatchSampler implemented for exact task balance
- ✅ UpsampledMultiTaskDataset balances all datasets to 122,450 samples
- ✅ Multi-task collate function with tensor conversion and dummy values
- ✅ Loss function handles all task combinations without KeyErrors
- ✅ All task labels have key existence checks
- ✅ Gradient flow issues resolved in loss computation
- ✅ Visibility shape fixed to [29] values instead of [1] average

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

This script provides:
- Simple, straightforward training loop
- No distributed training complexity
- Progress tracking with tqdm
- Automatic checkpointing

### Multi-GPU Training

#### Windows

```bash
# Use all available GPUs
launch_train.bat

# Specify number of GPUs (e.g., 4 GPUs)
launch_train.bat 4

# Or use torchrun directly
python -m torch.distributed.run --standalone --nproc_per_node=4 train.py
```

#### Linux

```bash
# Use all available GPUs
bash launch_train.sh

# Specify number of GPUs
NUM_GPUS=4 bash launch_train.sh

# Or use torchrun directly
torchrun --standalone --nproc_per_node=4 train.py
```

### Resume Training

```bash
# Single GPU
python train_simple.py --resume ./checkpoints/checkpoint_epoch_10.pth

# Multi-GPU
python -m torch.distributed.run --standalone --nproc_per_node=4 train.py --resume ./checkpoints/checkpoint_epoch_10.pth
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
├── README.md              # This file
├── config.py              # Configuration settings
├── datasets.py            # Dataset implementations with upsampling
├── losses.py              # Multi-task loss functions
├── train.py               # Multi-GPU training script (DDP)
├── train_simple.py        # Single-GPU training script
├── launch_train.sh        # Launch script (Linux)
├── launch_train.bat       # Launch script (Windows)
├── test_setup.py          # Verification script
├── network/
│   └── models/
│       ├── facexformer.py    # Main model
│       └── transformer.py    # TwoWayTransformer decoder
├── checkpoints/           # Saved model checkpoints
└── logs/                  # Training logs
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

## Age Classification Details

Age is predicted as an 8-class classification task with the following bucket structure:

### Age Buckets

| Class Index | Age Range | Representative Value |
|-------------|-----------|---------------------|
| 0 | 0-9 years | 4.5 |
| 1 | 10-19 years | 14.5 |
| 2 | 20-29 years | 24.5 |
| 3 | 30-39 years | 34.5 |
| 4 | 40-49 years | 44.5 |
| 5 | 50-59 years | 54.5 |
| 6 | 60-69 years | 64.5 |
| 7 | 70+ years | 75.0 |

### Loss Function

Age loss combines classification and regression:
- **Classification Loss**: Cross-entropy on 8 classes
- **Regression Loss**: L1 loss using representative values (normalized to [0,1])
- **Final Loss**: L_a = mean(CE + normalized_L1)

### Dataset Handling

- **UTKFace**: Continuous ages (0-116) are automatically binned into the 8 buckets
- **FairFace**: String labels ('0-2', '10-19', etc.) are mapped to bucket indices
- **Conversion**: The `age_to_bucket()` helper in [datasets.py](datasets.py) handles the bucketing logic

### Evaluation Metrics

- **MAE (Mean Absolute Error)**: Computed by converting predicted class probabilities to continuous age using weighted average with representative values
- **Accuracy**: Percentage of correctly classified age buckets

## Loss Functions

The loss function follows the paper formulation (excluding expression and face recognition):

### Task-Specific Losses

1. **Face Parsing (L_seg)**:
   - Combination: mean(Dice Loss + Cross Entropy)
   - Applied to: CelebAMask-HQ (11 classes)

2. **Landmark Detection (L_ind)**:
   - **STAR Loss**: Smooth L1-like formulation
   - More robust to outliers than L2
   - Formula: STAR(d) = 0.5 * d² if d < 1, else d - 0.5
   - Applied to: 300W (68 landmarks)

3. **Head Pose Estimation (L_hpe)**:
   - **Geodesic Loss** on SO(3) rotation group
   - Converts Euler angles to rotation matrices
   - Computes: d(R₁, R₂) = arccos((trace(R₁ᵀR₂) - 1) / 2)
   - More geometrically meaningful than L1 on Euler angles
   - Applied to: 300W-LP (yaw, pitch, roll)

4. **Attributes (L_attr)**:
   - Binary Cross Entropy with Logits
   - Applied to: CelebA (40 binary attributes)

5. **Age (L_a)**:
   - Combination: mean(L1 + Cross Entropy)
   - L1: Normalized regression on representative values
   - CE: 8-class classification
   - Applied to: UTKFace, FairFace

6. **Gender/Race (L_g, L_r)**:
   - Cross Entropy Loss
   - Gender: 2 classes (Male/Female)
   - Race: 5 classes (White, Black, Asian, Indian, Others)
   - Applied to: UTKFace, FairFace

7. **Visibility (L_vis)**:
   - Binary Cross Entropy with Logits
   - Element-wise filtering for valid landmarks
   - Applied to: COFW (29 landmark visibility scores)

### Total Loss

L_total = w_seg × L_seg + w_ind × L_ind + w_hpe × L_hpe + w_attr × L_attr + w_a × L_a + w_g × L_g + w_r × L_r + w_vis × L_vis

Loss weights are configured in [config.py](config.py).

## License

This implementation follows the license of the original facexformer-main repository.

## Support

For issues or questions:
1. Check this README
2. Review [config.py](config.py) settings
3. Run [test_setup.py](test_setup.py) to verify the setup
4. Ensure datasets are accessible at `../facexformer-my/datasets/`
5. Check GPU memory and adjust batch size accordingly

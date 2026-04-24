# Server Run Instructions

These instructions are for running the FaceXFormer reproduction scaffold on a server or cluster with the full datasets.

Target repository:

```text
https://github.com/kamrul28890/FaceXFormer-a-unified-transformer
```

## 1. Clone the Repository

```bash
git clone https://github.com/kamrul28890/FaceXFormer-a-unified-transformer.git
cd FaceXFormer-a-unified-transformer
```

## 2. Create the Environment

Use Python 3.10. A virtualenv works for a single machine. On a cluster, a conda/mamba environment is also fine.

Virtualenv:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Conda alternative:

```bash
conda create -n facexformer python=3.10 -y
conda activate facexformer
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

## 3. Place the Datasets

Datasets should live under:

```text
datasets/
```

The tiny-run scripts also auto-detect a nested full copy:

```text
datasets/datasets/
```

Expected folders for the current 8-task implementation:

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
`-- LFWA/
```

Optional/future 10-task folders:

```text
datasets/
|-- RAFDB/
|-- AffectNet+/
`-- MS1MV3/
```

Important: datasets are intentionally ignored by Git. Do not commit dataset files.

## 4. First Smoke Tests

Run a tiny baseline verification first. This checks dataset loading, model creation, forward pass, loss, and metrics.

```bash
python scripts/baseline_verification.py \
  --tasks landmark \
  --max-samples 4 \
  --batch-size 1 \
  --num-workers 0
```

Then test a slightly wider subset:

```bash
python scripts/baseline_verification.py \
  --tasks segmentation landmark headpose attribute \
  --max-samples 4 \
  --batch-size 1 \
  --num-workers 0
```

Outputs:

```text
results/baseline_tiny/gap_analysis_baseline_tiny.csv
results/baseline_tiny/gap_analysis_baseline_tiny.json
```

Without a checkpoint, these metrics are only smoke-test metrics. They are not scientifically meaningful.

## 5. Tiny Ablation Smoke Test

Run one tiny training/evaluation variant:

```bash
python scripts/ablation_study_tiny.py \
  --variants full \
  --tasks segmentation landmark headpose \
  --max-samples 4 \
  --eval-samples 4 \
  --epochs 1 \
  --max-train-batches 1 \
  --batch-size 1 \
  --num-workers 0
```

Then compare several local ablation variants:

```bash
python scripts/ablation_study_tiny.py \
  --variants full standard_cross_attention unbalanced_sampler uniform_loss \
  --tasks segmentation landmark headpose \
  --max-samples 8 \
  --eval-samples 8 \
  --epochs 1 \
  --max-train-batches 3 \
  --batch-size 1 \
  --amp \
  --num-workers 0
```

Outputs:

```text
results/ablation_tiny/ablation_train_summary.csv
results/ablation_tiny/ablation_eval_summary.csv
results/ablation_tiny/ablation_summary.json
results/ablation_tiny/<variant>/tiny_checkpoint_last.pth
```

## 6. Full Baseline Verification

If a checkpoint is available:

```bash
python scripts/baseline_verification.py \
  --checkpoint checkpoints/best_model.pth \
  --allow-partial-checkpoint \
  --tasks segmentation landmark headpose attribute age gender race visibility \
  --max-samples 0 \
  --batch-size 16 \
  --num-workers 8
```

Use `--max-samples 0` to mean "use the whole available dataset subset".

Adjust `--batch-size` based on GPU memory.

## 7. Full Training

Single-GPU training:

```bash
python train_simple.py
```

Multi-GPU DDP training:

```bash
torchrun --nproc_per_node=8 train.py
```

The paper trains with:

```text
8 GPUs
48 images per GPU
384 total batch size
12 epochs
AdamW
LR 1e-4
weight decay 1e-5
LR drops at epochs 6 and 10
```

The current repo config may need batch-size adjustment for your GPU memory.

## 8. Suggested Cluster Script

Example SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=facexformer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /path/to/FaceXFormer-a-unified-transformer
source .venv/bin/activate

mkdir -p logs checkpoints results

torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  train.py
```

If NCCL networking fails on the cluster, try adding:

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
```

## 9. Evaluation

Evaluate a checkpoint:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

## 10. What to Send Back

Please send back:

```text
results/baseline_tiny/*.csv
results/ablation_tiny/*.csv
logs/*.out
logs/*.err
checkpoint metadata: epoch, train loss, validation result
```

Do not send raw datasets or full checkpoints through Git.

## 11. Known Current Limitations

The current repo is an 8-task scaffold. The full paper target still needs:

- 142 task-token architecture
- 19-class segmentation mapping
- expression task
- face-recognition task
- paper-accurate landmark head
- full paper-grade ablations

Use `to-dos.md` as the implementation roadmap.

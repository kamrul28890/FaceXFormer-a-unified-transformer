"""
Training script for FaceXFormer-main with multi-GPU distributed data parallel support.
Adapted from facexformer-my, excluding expression recognition.

Features:
- Multi-GPU training using DistributedDataParallel
- Upsampled multi-task co-training
- Task-specific loss weighting
- Automatic GPU memory configuration
- Checkpoint saving and loading

Usage (single GPU):
    python train.py

Usage (standalone multi-GPU via torchrun):
    torchrun --nproc_per_node=4 train.py

Usage via SLURM (see submit_job.slurm):
    sbatch submit_job.slurm
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse
from pathlib import Path
import warnings
import json
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
import numpy as np

from scripts.small_run_common import PAPER_TARGETS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Suppress DDP gradient stride mismatch warning (performance warning only, not an error)
warnings.filterwarnings('ignore', message='Grad strides do not match bucket view strides')

from config import config
from network.models.facexformer import FaceXFormer
from losses import MultiTaskLoss
from datasets import (
    CelebAMaskHQDataset, W300Dataset, W300LPDataset,
    CelebADataset, UTKFaceDataset, FairFaceDataset, COFWDataset,
    W300VWDataset, BIWIDataset, LFWADataset,
    MultiLabelDatasetWrapper, create_multi_task_dataloader
)


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed mode")
        return 0, 1, 0
    
    # Set device to the specific GPU assigned to this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Initialize process group with explicit device_id to avoid NCCL warnings
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        device_id=device  # Explicitly specify device for NCCL communication
    )
    dist.barrier()
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    device: torch.device,
    rank: int = 0
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    task_losses = {}
    
    import time
    batch_times = []
    
    if rank == 0:
        print(f"Starting epoch {epoch} training with {len(dataloader)} batches...")
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]', ncols=100, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    else:
        pbar = dataloader
    
    for batch_idx, (images, targets) in enumerate(pbar):
        batch_start = time.time()
        
        if rank == 0 and batch_idx == 0:
            print(f"[rank{rank}] Loading first batch...")
        images = images.to(device)
        task_ids = targets['task_id'].to(device)
        
        if rank == 0 and batch_idx == 0:
            print(f"[rank{rank}] First batch loaded, shape: {images.shape}, moving targets to device...")
        
        # Move all targets to device
        for key in targets:
            targets[key] = targets[key].to(device)
        
        if rank == 0 and batch_idx == 0:
            print(f"[rank{rank}] Starting forward pass...")
        
        # Forward pass
        landmark_out, headpose_out, attribute_out, visibility_out, \
        age_out, gender_out, race_out, seg_out = model(images, targets, task_ids)
        
        if rank == 0 and batch_idx == 0:
            print(f"[rank{rank}] Forward pass complete, computing loss...")
        
        # Prepare predictions dict
        predictions = {
            'landmark_output': landmark_out,
            'headpose_output': headpose_out,
            'attribute_output': attribute_out,
            'visibility_output': visibility_out,
            'age_output': age_out,
            'gender_output': gender_out,
            'race_output': race_out,
            'seg_output': seg_out
        }
        
        # Compute loss
        loss, individual_losses = criterion(
            predictions,
            targets,
            task_ids,
            compute_individual=True
        )
        
        if rank == 0 and batch_idx == 0:
            print(f"[rank{rank}] Loss computed: {loss.item():.4f}, starting backward pass...")
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent explosion - this is the key!
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        optimizer.zero_grad()
        
        if rank == 0 and batch_idx == 0:
            print(f"[rank{rank}] Backward pass complete, clipping gradients...")
        
        if rank == 0 and batch_idx == 0:
            print(f"[rank{rank}] First iteration complete!")
        
        # Accumulate losses
        total_loss += loss.item()
        for task_name, task_loss in individual_losses.items():
            if task_name not in task_losses:
                task_losses[task_name] = 0.0
            task_losses[task_name] += task_loss.item()
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Update progress bar
        if rank == 0:
            avg_batch_time = sum(batch_times[-10:]) / len(batch_times[-10:])  # Last 10 batches
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'batch_time': f'{batch_time:.2f}s',
                'avg_time': f'{avg_batch_time:.2f}s'
            })
            
            # Print every 10 batches for visibility
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}: loss={loss.item():.4f}, time={batch_time:.2f}s")
    
    num_batches = len(dataloader)
    if dist.is_available() and dist.is_initialized():
        loss_tensor = torch.tensor([total_loss, num_batches], dtype=torch.float64, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = loss_tensor[0].item()
        num_batches = max(int(loss_tensor[1].item()), 1)

        for task_name in config.LOSS_WEIGHTS.keys():
            task_tensor = torch.tensor(task_losses.get(task_name, 0.0), dtype=torch.float64, device=device)
            dist.all_reduce(task_tensor, op=dist.ReduceOp.SUM)
            task_losses[task_name] = task_tensor.item()

    # Average losses
    avg_loss = total_loss / max(num_batches, 1)
    avg_task_losses = {k: v / max(num_batches, 1) for k, v in task_losses.items()}
    
    return avg_loss, avg_task_losses


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    device: torch.device,
    rank: int = 0
):
    """Evaluate the model on test set."""
    model.eval()
    total_loss = 0.0
    task_losses = {}
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Test]')
        else:
            pbar = dataloader
        
        for images, targets in pbar:
            images = images.to(device)
            task_ids = targets['task_id'].to(device)
            
            # Move all targets to device
            for key in targets:
                targets[key] = targets[key].to(device)
            
            # Forward pass
            landmark_out, headpose_out, attribute_out, visibility_out, \
            age_out, gender_out, race_out, seg_out = model(images, targets, task_ids)
            
            # Prepare predictions dict
            predictions = {
                'landmark_output': landmark_out,
                'headpose_output': headpose_out,
                'attribute_output': attribute_out,
                'visibility_output': visibility_out,
                'age_output': age_out,
                'gender_output': gender_out,
                'race_output': race_out,
                'seg_output': seg_out
            }
            
            # Compute loss
            loss, individual_losses = criterion(
                predictions,
                targets,
                task_ids,
                compute_individual=True
            )
            
            # Accumulate losses
            total_loss += loss.item()
            for task_name, task_loss in individual_losses.items():
                if task_name not in task_losses:
                    task_losses[task_name] = 0.0
                task_losses[task_name] += task_loss.item()
            
            if rank == 0:
                pbar.set_postfix({'loss': loss.item()})
    
    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_task_losses = {k: v / len(dataloader) for k, v in task_losses.items()}
    
    return avg_loss, avg_task_losses


def compute_nme(predictions, ground_truth, image_size=224):
    """
    Compute Normalized Mean Error for landmark prediction.
    Normalizes by inter-ocular distance (IOD) — the standard for 300W.

    Args:
        predictions: [B, 136] tensor (68 landmarks * 2), normalized [0,1]
        ground_truth: [B, 136] tensor, normalized [0,1]
        image_size: used to convert normalized coords to pixels for IOD

    Returns:
        float: NME value (percentage)
    """
    predictions = predictions.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()

    def to_unit_interval(values):
        # Current FaceXFormer landmark heads are trained in centered
        # coordinates. Older utilities used [0, 1], so accept both here.
        if np.nanmin(values) < 0.0:
            values = (values + 1.0) / 2.0
        return values

    # Reshape to [B, 68, 2] — scale to pixel space for meaningful distances
    pred_pts = to_unit_interval(predictions).reshape(-1, 68, 2) * image_size
    gt_pts = to_unit_interval(ground_truth).reshape(-1, 68, 2) * image_size

    # Per-landmark Euclidean distance: [B, 68]
    distances = np.sqrt(np.sum((pred_pts - gt_pts) ** 2, axis=2))

    # Inter-ocular distance: outer eye corners (indices 36 and 45 for 68-point)
    iod = np.sqrt(np.sum((gt_pts[:, 45, :] - gt_pts[:, 36, :]) ** 2, axis=1))  # [B]
    iod = np.maximum(iod, 1e-6)  # avoid division by zero

    # NME per sample, then average
    nme_per_sample = np.mean(distances, axis=1) / iod  # [B]
    nme = np.mean(nme_per_sample) * 100  # percentage

    return nme


def compute_age_from_logits(age_logits):
    """
    Convert age classification logits to continuous age predictions.
    
    Args:
        age_logits: [B, 8] tensor of logits for 8 age buckets
    
    Returns:
        [B] tensor of predicted ages
    """
    # Age bucket centers (from authors): 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
    age_bins = torch.tensor([4.5, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 75.0], 
                           device=age_logits.device, dtype=torch.float32)
    
    # Weighted average of bin centers using softmax probabilities
    age_probs = F.softmax(age_logits, dim=1)  # [B, 8]
    predicted_ages = (age_probs * age_bins).sum(dim=1)  # [B]
    
    return predicted_ages


def compute_mae(predictions, ground_truth, in_radians=False):
    """
    Compute Mean Absolute Error.

    Args:
        predictions: tensor - can be [B], [B, 1], or [B, 8] (age logits)
        ground_truth: tensor - [B] class indices (0-7) for age, or continuous values
        in_radians: if True, convert the result from radians to degrees

    Returns:
        float: MAE value (in years for age, in degrees for head pose, otherwise native units)
    """
    age_bins = torch.tensor([4.5, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 75.0],
                            dtype=torch.float32)

    # Age logits [B, 8]: convert both prediction AND ground truth to years
    if len(predictions.shape) == 2 and predictions.shape[1] == 8:
        predictions = compute_age_from_logits(predictions)
        # ground_truth is class index (0-7); map to bin-centre years
        gt_idx = ground_truth.long().clamp(0, 7).cpu()
        ground_truth = age_bins[gt_idx].to(predictions.device)
    elif len(predictions.shape) > 1:
        predictions = predictions.squeeze()

    if len(ground_truth.shape) > 1:
        ground_truth = ground_truth.squeeze()

    predictions = predictions.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()

    mae = np.mean(np.abs(predictions - ground_truth))
    if in_radians:
        mae = np.degrees(mae)
    return mae


def compute_f1_score(predictions, ground_truth, num_classes=11):
    """
    Compute F1-score for segmentation (macro average across classes).
    
    Args:
        predictions: [B, C, H, W] tensor (logits)
        ground_truth: [B, H, W] tensor (class indices)
        num_classes: number of segmentation classes
    
    Returns:
        float: macro F1-score
    """
    # Get predicted classes
    pred_classes = torch.argmax(predictions, dim=1)  # [B, H, W]
    
    # Flatten
    pred_flat = pred_classes.cpu().numpy().flatten()
    gt_flat = ground_truth.cpu().numpy().flatten()
    
    # Compute macro F1 (average across classes)
    f1 = f1_score(gt_flat, pred_flat, average='macro', labels=range(num_classes), zero_division=0)
    
    return f1


def compute_accuracy(predictions, ground_truth):
    """
    Compute accuracy for classification tasks.
    
    Args:
        predictions: [B, C] tensor (logits) or [B, num_attributes] for multi-label
        ground_truth: [B] tensor (class indices) or [B, num_attributes] for multi-label
    
    Returns:
        float: accuracy
    """
    # Check if it's multi-label (ground_truth is 2D)
    if len(ground_truth.shape) > 1 and ground_truth.shape[1] > 1:  # Multi-label (attributes)
        pred_classes = (torch.sigmoid(predictions) > 0.5).float()
        # For multi-label, compute average accuracy across attributes
        correct = (pred_classes == ground_truth).float().mean()
        return correct.item()
    else:  # Multi-class or binary classification (ground_truth is 1D class indices)
        if predictions.shape[1] > 1:  # Multi-class (gender, race, age, etc.)
            pred_classes = torch.argmax(predictions, dim=1)
        else:  # Binary classification (single logit)
            pred_classes = (torch.sigmoid(predictions) > 0.5).long().squeeze()
    
    pred_classes = pred_classes.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    
    acc = accuracy_score(ground_truth, pred_classes)
    return acc


def compute_recall_at_precision(predictions, ground_truth, target_precision=0.8):
    """
    Compute recall at a given precision threshold for occlusion detection.
    Labels use 1=visible, 0=occluded; the metric measures how well the model
    detects the occluded (minority, harder) class, matching the paper's metric.
    
    Args:
        predictions: [B, 29] tensor (visibility scores; high = predicted visible)
        ground_truth: [B, 29] tensor (visibility labels: 1=visible, 0=occluded)
        target_precision: desired precision threshold (default 0.8)
    
    Returns:
        float: recall of occluded landmarks at target precision
    """
    predictions = torch.sigmoid(predictions).cpu().numpy().flatten()
    ground_truth = ground_truth.cpu().numpy().flatten()
    
    # Ensure ground_truth is binary and flip to positive=occluded
    # Labels: 1=visible → invert so 1=occluded for occlusion-detection PR curve
    ground_truth_occ = 1 - (ground_truth > 0.5).astype(int)
    # Model output is high for visible → invert score for occlusion detection
    predictions_occ = 1.0 - predictions
    
    # Compute precision-recall curve for occluded class
    precisions, recalls, thresholds = precision_recall_curve(ground_truth_occ, predictions_occ)
    
    # Find recall at target precision
    idx = np.where(precisions >= target_precision)[0]
    if len(idx) > 0:
        recall_at_prec = recalls[idx[0]]
    else:
        recall_at_prec = 0.0
    
    return recall_at_prec


def normalize_metric_for_report(task_name, raw_metric, paper_target=None):
    """Convert raw metric helper outputs into paper-table units."""
    if raw_metric is None:
        return None, "", None

    if task_name in {"segmentation", "attribute", "gender", "race", "visibility"}:
        normalized = raw_metric * 100.0
        unit = "percent"
    elif task_name == "landmark":
        normalized = raw_metric
        unit = "nme_percent"
    elif task_name == "headpose":
        normalized = raw_metric
        unit = "degrees"
    elif task_name == "age":
        normalized = raw_metric
        unit = "years"
    else:
        normalized = raw_metric
        unit = "raw"

    gap = None if paper_target is None else normalized - paper_target
    return normalized, unit, gap


def paper_target_for_dataset(task_name, dataset_name):
    """Return the paper target only for rows that match the paper protocol."""
    target_info = PAPER_TARGETS.get(task_name, {})
    paper_dataset = target_info.get("dataset")
    if not paper_dataset:
        return {}

    if task_name == "landmark" and any(part in dataset_name.lower() for part in ["common", "challenging"]):
        return {}

    dataset_aliases = {
        "CelebAMask-HQ": ["CelebAMaskHQ", "CelebAMask-HQ"],
        "300W": ["W300Dataset", "300W"],
        "BIWI": ["BIWIDataset", "BIWI"],
        "CelebA": ["CelebADataset", "CelebA"],
        "UTKFace": ["UTKFaceDataset", "UTKFace"],
        "COFW": ["COFWDataset", "COFW"],
    }
    aliases = dataset_aliases.get(paper_dataset, [paper_dataset])
    if any(alias in dataset_name for alias in aliases):
        return target_info
    return {}


def metric_for_task(task_name, predictions, targets):
    """Compute the same raw metric used by the current 8-task baseline."""
    if task_name == 'segmentation':
        return compute_f1_score(predictions['seg_output'], targets['segmentation'], num_classes=config.SEGMENTATION_CLASSES)
    if task_name == 'landmark':
        return compute_nme(predictions['landmark_output'], targets['landmark'])
    if task_name == 'headpose':
        return compute_mae(predictions['headpose_output'], targets['headpose'], in_radians=True)
    if task_name == 'attribute':
        return compute_accuracy(predictions['attribute_output'], targets['attribute'])
    if task_name == 'age':
        return compute_mae(predictions['age_output'], targets['age'].float())
    if task_name == 'gender':
        return compute_accuracy(predictions['gender_output'], targets['gender'])
    if task_name == 'race':
        return compute_accuracy(predictions['race_output'], targets['race'])
    if task_name == 'visibility':
        return compute_recall_at_precision(predictions['visibility_output'], targets['visibility'])
    raise KeyError(f"Unsupported task: {task_name}")


def segmentation_f1_from_confusion(confusion):
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    denom = (2.0 * tp) + fp + fn
    f1_per_class = np.divide(
        2.0 * tp,
        denom,
        out=np.zeros_like(tp, dtype=np.float64),
        where=denom > 0,
    )
    return float(f1_per_class.mean())


def validate_per_dataset(
    model: nn.Module,
    test_datasets: Dict[str, list],
    criterion: nn.Module,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1
):
    """
    Evaluate the model on test set with per-dataset breakdown and task-specific metrics.
    
    Returns:
        dict: Nested dictionary with structure:
            {
                'task_name': {
                    'dataset_name': {
                        'loss': loss_value,
                        'metric': metric_value  # F1, NME, MAE, Accuracy, or Recall
                    },
                    'overall': {
                        'loss': average_loss,
                        'metric': average_metric
                    }
                },
                'total': overall_loss
            }
    """
    model.eval()
    eval_model = model.module if isinstance(model, DDP) else model
    results = {}
    
    if rank == 0:
        print(f"\n{'='*80}")
        print("FINAL TEST SET EVALUATION")
        print(f"{'='*80}\n")
    
    total_loss = 0.0
    total_samples = 0
    
    # Task ID mapping
    task_id_map = {
        'segmentation': 0,
        'landmark': 1,
        'headpose': 2,
        'attribute': 3,
        'age': 4,
        'gender': 5,
        'race': 6,
        'visibility': 7
    }
    
    # Metric names for each task
    metric_names = {
        'segmentation': 'F1-Score',
        'landmark': 'NME (%)',
        'headpose': 'MAE (degrees)',
        'attribute': 'Accuracy',
        'age': 'MAE (years)',
        'gender': 'Accuracy',
        'race': 'Accuracy',
        'visibility': 'Recall@P80'
    }
    
    # Evaluate each task separately
    for task_name, datasets in test_datasets.items():
        if rank == 0:
            print(f"\n{'─'*80}")
            print(f"📊 TASK: {task_name.upper()} (Metric: {metric_names[task_name]})")
            print(f"{'─'*80}")
        
        results[task_name] = {}
        task_total_loss = 0.0
        task_total_metric = 0.0
        task_total_samples = 0
        
        # Evaluate each dataset for this task
        for dataset_idx, dataset in enumerate(datasets):
            # Get dataset name - check if it's a wrapper with custom name method
            if hasattr(dataset, 'get_name'):
                dataset_name = dataset.get_name()
            else:
                dataset_name = dataset.__class__.__name__
            
            # Create dataloader for single dataset. For evaluation, use exact
            # strided rank subsets instead of DistributedSampler so samples are
            # not padded/duplicated to make replicas even.
            if world_size > 1:
                local_indices = list(range(rank, len(dataset), world_size))
                local_dataset = Subset(dataset, local_indices)
                dataloader = torch.utils.data.DataLoader(
                    local_dataset,
                    batch_size=config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=config.NUM_WORKERS,
                    pin_memory=True
                )
            else:
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=config.NUM_WORKERS,
                    pin_memory=True
                )
            
            dataset_loss = 0.0
            dataset_metric = 0.0
            num_batches = 0
            dataset_samples = 0
            metric_predictions = []
            metric_targets = []
            segmentation_confusion = None
            first_batch_checked = False  # Flag for debug output
            
            with torch.no_grad():
                if rank == 0:
                    pbar = tqdm(dataloader, desc=f'  📁 {dataset_name}', leave=False, ncols=100)
                else:
                    pbar = dataloader
                
                for images, targets in pbar:
                    images = images.to(device)
                    
                    # Get task ID for this task
                    task_id = task_id_map[task_name]
                    batch_size = images.shape[0]
                    task_ids = torch.full((batch_size,), task_id, dtype=torch.long, device=device)
                    
                    # Move targets to device
                    for key in targets:
                        if isinstance(targets[key], torch.Tensor):
                            targets[key] = targets[key].to(device)
                    
                    # Add task_id to targets
                    targets['task_id'] = task_ids
                    
                    # Forward pass
                    landmark_out, headpose_out, attribute_out, visibility_out, \
                    age_out, gender_out, race_out, seg_out = eval_model(images, targets, task_ids)
                    
                    # Prepare predictions dict
                    predictions = {
                        'landmark_output': landmark_out,
                        'headpose_output': headpose_out,
                        'attribute_output': attribute_out,
                        'visibility_output': visibility_out,
                        'age_output': age_out,
                        'gender_output': gender_out,
                        'race_output': race_out,
                        'seg_output': seg_out
                    }
                    
                    # Compute loss
                    loss, _ = criterion(
                        predictions,
                        targets,
                        task_ids,
                        compute_individual=True
                    )
                    
                    dataset_loss += loss.item()
                    num_batches += 1
                    dataset_samples += batch_size
                    
                    # Compute task-specific metric
                    if task_name == 'segmentation':
                        if 'segmentation' in targets:
                            pred_classes = torch.argmax(seg_out, dim=1).detach().cpu()
                            gt_classes = targets['segmentation'].detach().cpu()
                            num_classes = config.SEGMENTATION_CLASSES
                            flat = gt_classes.reshape(-1) * num_classes + pred_classes.reshape(-1)
                            counts = torch.bincount(flat, minlength=num_classes * num_classes)
                            batch_confusion = counts.reshape(num_classes, num_classes).numpy()
                            if segmentation_confusion is None:
                                segmentation_confusion = batch_confusion
                            else:
                                segmentation_confusion += batch_confusion
                        elif rank == 0 and not first_batch_checked:
                            print(f"      ⚠️ WARNING: 'segmentation' not in targets. Available keys: {list(targets.keys())}")
                            first_batch_checked = True
                    
                    elif task_name == 'landmark':
                        if 'landmark' in targets:
                            metric = metric_for_task(task_name, predictions, targets)
                            dataset_metric += float(metric) * batch_size
                        elif rank == 0 and not first_batch_checked:
                            print(f"      ⚠️ WARNING: 'landmark' not in targets. Available keys: {list(targets.keys())}")
                            first_batch_checked = True
                    
                    elif task_name == 'headpose':
                        if 'headpose' in targets:
                            metric = metric_for_task(task_name, predictions, targets)
                            dataset_metric += float(metric) * batch_size
                    
                    elif task_name == 'attribute':
                        if 'attribute' in targets:
                            metric = metric_for_task(task_name, predictions, targets)
                            dataset_metric += float(metric) * batch_size
                        elif rank == 0 and not first_batch_checked:
                            print(f"      ⚠️ WARNING: 'attribute' not in targets. Available keys: {list(targets.keys())}")
                            first_batch_checked = True
                    
                    elif task_name == 'age':
                        if 'age' in targets:
                            metric = metric_for_task(task_name, predictions, targets)
                            dataset_metric += float(metric) * batch_size
                    
                    elif task_name == 'gender':
                        if 'gender' in targets:
                            metric = metric_for_task(task_name, predictions, targets)
                            dataset_metric += float(metric) * batch_size
                    
                    elif task_name == 'race':
                        if 'race' in targets:
                            metric = metric_for_task(task_name, predictions, targets)
                            dataset_metric += float(metric) * batch_size
                    
                    elif task_name == 'visibility':
                        if 'visibility' in targets:
                            metric_predictions.append(visibility_out.detach().cpu())
                            metric_targets.append(targets['visibility'].detach().cpu())
                    
                    if rank == 0:
                        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Synchronize across GPUs
            if world_size > 1:
                # Gather losses from all ranks
                dataset_loss_tensor = torch.tensor(dataset_loss, device=device)
                dataset_metric_tensor = torch.tensor(dataset_metric, device=device)
                num_batches_tensor = torch.tensor(num_batches, device=device)
                dataset_samples_tensor = torch.tensor(dataset_samples, device=device)
                
                torch.distributed.all_reduce(dataset_loss_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(dataset_metric_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(num_batches_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(dataset_samples_tensor, op=torch.distributed.ReduceOp.SUM)
                
                dataset_loss = dataset_loss_tensor.item()
                dataset_metric = dataset_metric_tensor.item()
                num_batches = int(num_batches_tensor.item())
                dataset_samples = int(dataset_samples_tensor.item())

                if task_name == 'segmentation':
                    if segmentation_confusion is None:
                        segmentation_confusion = np.zeros(
                            (config.SEGMENTATION_CLASSES, config.SEGMENTATION_CLASSES),
                            dtype=np.float64,
                        )
                    confusion_tensor = torch.tensor(segmentation_confusion, dtype=torch.float64, device=device)
                    torch.distributed.all_reduce(confusion_tensor, op=torch.distributed.ReduceOp.SUM)
                    segmentation_confusion = confusion_tensor.cpu().numpy()

                if task_name == 'visibility':
                    local_predictions = torch.cat(metric_predictions, dim=0) if metric_predictions else torch.empty(0)
                    local_targets = torch.cat(metric_targets, dim=0) if metric_targets else torch.empty(0)
                    gathered_predictions = [None for _ in range(world_size)]
                    gathered_targets = [None for _ in range(world_size)]
                    torch.distributed.all_gather_object(gathered_predictions, local_predictions)
                    torch.distributed.all_gather_object(gathered_targets, local_targets)
                    metric_predictions = [item for item in gathered_predictions if item is not None and item.numel() > 0]
                    metric_targets = [item for item in gathered_targets if item is not None and item.numel() > 0]
            
            # Average loss and metric for this dataset
            avg_dataset_loss = dataset_loss / num_batches if num_batches > 0 else 0.0
            if task_name == 'segmentation' and segmentation_confusion is not None:
                avg_dataset_metric = segmentation_f1_from_confusion(segmentation_confusion)
            elif task_name == 'visibility' and metric_predictions:
                avg_dataset_metric = metric_for_task(
                    task_name,
                    {'visibility_output': torch.cat(metric_predictions, dim=0)},
                    {'visibility': torch.cat(metric_targets, dim=0)}
                )
            else:
                avg_dataset_metric = dataset_metric / dataset_samples if dataset_samples > 0 else 0.0

            target_info = paper_target_for_dataset(task_name, dataset_name)
            paper_target = target_info.get('target')
            normalized_metric, normalized_unit, normalized_gap = normalize_metric_for_report(
                task_name,
                avg_dataset_metric,
                paper_target
            )
            
            results[task_name][dataset_name] = {
                'loss': avg_dataset_loss,
                'metric': avg_dataset_metric,
                'raw_metric': avg_dataset_metric,
                'normalized_metric': normalized_metric,
                'normalized_metric_unit': normalized_unit,
                'normalized_gap_metric_minus_target': normalized_gap,
                'paper_metric': target_info.get('metric', ''),
                'paper_target': paper_target,
                'samples': dataset_samples,
                'batches': num_batches,
            }
            
            task_total_loss += dataset_loss
            task_total_metric += avg_dataset_metric * dataset_samples
            task_total_samples += num_batches
            metric_samples = results[task_name].setdefault('_metric_samples', 0)
            results[task_name]['_metric_samples'] = metric_samples + dataset_samples
            
            if rank == 0:
                display_metric = normalized_metric if normalized_metric is not None else avg_dataset_metric
                display_unit = f" {normalized_unit}" if normalized_unit else ""
                print(f"  ├─ {dataset_name:30s}: Loss={avg_dataset_loss:.6f}, {metric_names[task_name]}={display_metric:.4f}{display_unit}")
        
        # Compute overall task loss and metric
        avg_task_loss = task_total_loss / task_total_samples if task_total_samples > 0 else 0.0
        metric_samples = results[task_name].pop('_metric_samples', 0)
        avg_task_metric = task_total_metric / metric_samples if metric_samples > 0 else 0.0
        normalized_metric, normalized_unit, _ = normalize_metric_for_report(task_name, avg_task_metric, None)
        
        results[task_name]['overall'] = {
            'loss': avg_task_loss,
            'metric': avg_task_metric,
            'raw_metric': avg_task_metric,
            'normalized_metric': normalized_metric,
            'normalized_metric_unit': normalized_unit,
            'samples': metric_samples,
            'batches': task_total_samples,
        }
        
        total_loss += task_total_loss
        total_samples += task_total_samples
        
        if rank == 0:
            display_metric = normalized_metric if normalized_metric is not None else avg_task_metric
            display_unit = f" {normalized_unit}" if normalized_unit else ""
            print(f"  └─ {'OVERALL':30s}: Loss={avg_task_loss:.6f}, {metric_names[task_name]}={display_metric:.4f}{display_unit}")
    
    # Compute total loss across all tasks
    avg_total_loss = total_loss / total_samples if total_samples > 0 else 0.0
    results['total'] = avg_total_loss
    
    if rank == 0:
        print(f"\n{'─'*80}")
        print(f"🎯 TOTAL LOSS (All Tasks): {avg_total_loss:.6f}")
        print(f"{'='*80}\n")
    
    return results


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath, rank=0):
    """Save model checkpoint."""
    if rank != 0:
        return
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train FaceXFormer-main')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"FaceXFormer-main Training")
        print(f"{'='*60}")
        print(f"Number of GPUs: {world_size}")
        print(f"Batch size per GPU: {config.BATCH_SIZE}")
        print(f"Effective batch size: {config.BATCH_SIZE * world_size}")
        print(f"Learning rate: {config.LEARNING_RATE}")
        print(f"{'='*60}\n")
    
    # Set device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets (NO expression datasets)
    # NOTE: All GPUs load datasets, but data is split via DistributedSampler
    if rank == 0:
        print("Loading datasets (all GPUs are loading, only rank 0 prints)...")
    
    try:
        if rank == 0:
            print("  Loading CelebAMask-HQ...")
        celebamask_train = CelebAMaskHQDataset('train')
        celebamask_test = CelebAMaskHQDataset('test')
        
        if rank == 0:
            print("  Loading 300W...")
        w300_train = W300Dataset('train')
        w300_test = W300Dataset('test')
        
        if rank == 0:
            print("  Loading 300W-LP...")
        w300lp_train = W300LPDataset('train')
        
        if rank == 0:
            print("  Loading CelebA (may take a while for first run)...")
        celeba_train = CelebADataset('train', rank=rank, world_size=world_size)
        celeba_test = CelebADataset('test', rank=rank, world_size=world_size)
        
        if rank == 0:
            print("  Loading UTKFace...")
        utkface_train = UTKFaceDataset('train')
        utkface_test = UTKFaceDataset('test')
        
        if rank == 0:
            print("  Loading FairFace...")
        fairface_train = FairFaceDataset('train')
        fairface_test = FairFaceDataset('test')
        
        if rank == 0:
            print("  Loading COFW...")
        cofw_train = COFWDataset('train')
        cofw_test = COFWDataset('test')
        
        # Test-only datasets
        if rank == 0:
            print("  Loading 300VW (test only)...")
        w300vw_test = W300VWDataset('test')
        
        if rank == 0:
            print("  Loading BIWI (test only)...")
        biwi_test = BIWIDataset('test')
        
        if rank == 0:
            print("  Loading LFWA (test only)...")
        lfwa_test = LFWADataset('test')
        
        train_datasets = {
            'segmentation': [celebamask_train],
            'landmark': [w300_train],
            'headpose': [w300lp_train],
            'attribute': [celeba_train],
            'age': [MultiLabelDatasetWrapper(utkface_train, 'age'), 
                    MultiLabelDatasetWrapper(fairface_train, 'age')],
            'gender': [MultiLabelDatasetWrapper(utkface_train, 'gender'),
                       MultiLabelDatasetWrapper(fairface_train, 'gender')],
            'race': [MultiLabelDatasetWrapper(utkface_train, 'race'),
                     MultiLabelDatasetWrapper(fairface_train, 'race')],
            'visibility': [cofw_train]
        }
        
        test_datasets = {
            'segmentation': [celebamask_test],
            'landmark': [w300_test, w300vw_test],
            'headpose': [biwi_test],
            'attribute': [celeba_test, lfwa_test],
            'age': [MultiLabelDatasetWrapper(utkface_test, 'age'),
                    MultiLabelDatasetWrapper(fairface_test, 'age')],
            'gender': [MultiLabelDatasetWrapper(utkface_test, 'gender'),
                       MultiLabelDatasetWrapper(fairface_test, 'gender')],
            'race': [MultiLabelDatasetWrapper(utkface_test, 'race'),
                     MultiLabelDatasetWrapper(fairface_test, 'race')],
            'visibility': [cofw_test]
        }
        
        if rank == 0:
            print(f"✓ All datasets loaded successfully")
            total_train = sum(sum(len(ds) for ds in datasets) for datasets in train_datasets.values())
            print(f"  Total training samples: {total_train:,}")
    
    except FileNotFoundError as e:
        if rank == 0:
            print(f"\n❌ Dataset loading error: {e}")
            print(f"\nPlease ensure datasets are downloaded and extracted to {config.DATASET_ROOT}/")
            print("See README for dataset download instructions.")
        cleanup_distributed()
        return
    
    # Create data loaders
    train_loader = create_multi_task_dataloader(
        train_datasets,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        use_upsampling=True,
        rank=rank,
        world_size=world_size
    )
    
    if rank == 0:
        print(f"\n🔍 DATALOADER INFO:")
        print(f"  Batches per GPU: {len(train_loader)}")
        print(f"  Batch size per GPU: {config.BATCH_SIZE}")
        print(f"  World size: {world_size}")
        print(f"  Total samples per epoch: {len(train_loader) * config.BATCH_SIZE * world_size}")
    
    test_loader = create_multi_task_dataloader(
        test_datasets,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        use_upsampling=True,
        rank=rank,
        world_size=world_size
    )
    
    # Create model
    if rank == 0:
        print("\nCreating model...")
    
    model = FaceXFormer().to(device)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created")
        print(f"  Total parameters: {total_params:,}")
    
    # Wrap model with DDP
    # Note: find_unused_parameters=True is required for multi-task learning
    # where not all task heads receive gradients in every batch
    if world_size > 1:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=True
        )
    
    # Create loss function
    criterion = MultiTaskLoss(config.LOSS_WEIGHTS).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Create learning rate scheduler
    def lr_lambda(epoch):
        if epoch < config.LR_DECAY_EPOCHS[0]:
            return 1.0
        elif epoch < config.LR_DECAY_EPOCHS[1]:
            return config.LR_DECAY_FACTOR
        else:
            return config.LR_DECAY_FACTOR ** 2
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        if rank == 0:
            print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if rank == 0:
            print(f"✓ Resumed from epoch {checkpoint['epoch']}")
    
    # Create checkpoint directory
    if rank == 0:
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Training loop
    best_train_loss = float('inf')
    
    try:
        for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
            # Set epoch for proper shuffling in distributed training
            if world_size > 1:
                # Handle both batch_sampler and regular sampler
                if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
                    train_loader.batch_sampler.set_epoch(epoch)
                elif hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
            
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
                print(f"{'='*60}")
            
            # Synchronize before training to ensure all ranks are ready
            if world_size > 1:
                dist.barrier()
            
            if rank == 0:
                print(f"[rank{rank}] About to call train_one_epoch...")
            
            # Train
            train_loss, train_task_losses = train_one_epoch(
                model, train_loader, criterion, optimizer, epoch, device, rank
            )
            
            if rank == 0:
                print(f"\nTrain Loss: {train_loss:.4f}")
                print("Train Task Losses:")
                for task_name, task_loss in train_task_losses.items():
                    print(f"  {task_name}: {task_loss:.4f}")
            
            # Learning rate scheduling
            scheduler.step()
            
            # Save checkpoint based on training loss
            if rank == 0:
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, train_loss,
                        os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'),
                        rank
                    )
                
                if epoch % config.SAVE_FREQ == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, train_loss,
                        os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth'),
                        rank
                    )
        
        # =====================================================================
        # FINAL TEST SET EVALUATION (ONLY ONCE AFTER TRAINING)
        # =====================================================================
        if rank == 0:
            print(f"\n{'='*60}")
            print("Training completed!")
            print(f"Best training loss: {best_train_loss:.4f}")
            print(f"{'='*60}\n")
        
        # Synchronize all ranks before final evaluation
        if world_size > 1:
            dist.barrier()
        
        # Perform detailed test set evaluation
        test_results = validate_per_dataset(
            model, test_datasets, criterion, device, rank, world_size
        )
        
        # Save test results to file (rank 0 only)
        if rank == 0:
            results_file = os.path.join(config.CHECKPOINT_DIR, 'test_results.json')
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2, cls=NumpyEncoder)
            print(f"\n✓ Test results saved to: {results_file}")
            
            # Print summary table
            print(f"\n{'='*100}")
            print("📊 TEST SET SUMMARY")
            print(f"{'='*100}")
            print(f"{'Task':<20} {'Dataset':<30} {'Loss':<15} {'Metric':<15} {'Value':<15}")
            print(f"{'─'*100}")
            
            for task_name, task_results in test_results.items():
                if task_name == 'total':
                    continue
                
                metric_name = {
                    'segmentation': 'F1-Score',
                    'landmark': 'NME (%)',
                    'headpose': 'MAE (deg)',
                    'attribute': 'Accuracy',
                    'age': 'MAE (years)',
                    'gender': 'Accuracy',
                    'race': 'Accuracy',
                    'visibility': 'Recall@P80'
                }[task_name]
                
                first = True
                for dataset_name, metrics in task_results.items():
                    if dataset_name == 'overall':
                        continue
                    value = metrics.get('normalized_metric', metrics['metric'])
                    unit = metrics.get('normalized_metric_unit', '')
                    gap = metrics.get('normalized_gap_metric_minus_target')
                    suffix = f" {unit}" if unit else ""
                    if gap is not None:
                        suffix += f" gap={gap:.4f}"
                    
                    if first:
                        print(f"{task_name:<20} {dataset_name:<30} {metrics['loss']:.6f}      {metric_name:<15} {value:.4f}{suffix}")
                        first = False
                    else:
                        print(f"{'':<20} {dataset_name:<30} {metrics['loss']:.6f}      {metric_name:<15} {value:.4f}{suffix}")
                
                # Print overall for this task
                if 'overall' in task_results:
                    overall = task_results['overall']
                    value = overall.get('normalized_metric', overall['metric'])
                    unit = overall.get('normalized_metric_unit', '')
                    suffix = f" {unit}" if unit else ""
                    print(f"{'':<20} {'[OVERALL]':<30} {overall['loss']:.6f}      {metric_name:<15} {value:.4f}{suffix}")
                print(f"{'─'*100}")
            
            print(f"{'TOTAL (All Tasks)':<20} {'':<30} {test_results['total']:.6f}")
            print(f"{'='*100}\n")
    
    finally:
        # Cleanup distributed training (always executed, even on exceptions)
        cleanup_distributed()


if __name__ == "__main__":
    main()

"""
Training script for FaceXFormer-main with multi-GPU distributed data parallel support.
Adapted from facexformer-my, excluding expression recognition.

Features:
- Multi-GPU training using DistributedDataParallel
- Upsampled multi-task co-training
- Task-specific loss weighting
- Automatic GPU memory configuration
- Checkpoint saving and loading
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import argparse
from pathlib import Path
import warnings
import json
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
import numpy as np

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
    
    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_task_losses = {k: v / len(dataloader) for k, v in task_losses.items()}
    
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
    
    Args:
        predictions: [B, 136] tensor (68 landmarks * 2)
        ground_truth: [B, 136] tensor
        image_size: image size used when coordinates are stored as pixels
    
    Returns:
        float: NME percentage
    """
    predictions = predictions.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    
    # Reshape to [B, 68, 2]
    pred_pts = predictions.reshape(-1, 68, 2)
    gt_pts = ground_truth.reshape(-1, 68, 2)
    
    # Compute Euclidean distance for each landmark
    distances = np.sqrt(np.sum((pred_pts - gt_pts) ** 2, axis=2))  # [B, 68]
    
    # The current 300W loader stores landmarks normalized to [0, 1]. Older
    # helpers assumed pixel coordinates, which made NME artificially tiny.
    max_abs_coord = max(float(np.max(np.abs(pred_pts))), float(np.max(np.abs(gt_pts))))
    if max_abs_coord <= 2.0:
        diagonal = np.sqrt(2.0)
    else:
        diagonal = np.sqrt(image_size ** 2 + image_size ** 2)
    nme = np.mean(distances) / diagonal * 100  # Convert to percentage
    
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


def compute_age_targets_to_years(ground_truth, device=None):
    """
    Convert age bucket labels to their representative age values.

    UTKFace/FairFace loaders in this repo return age as bucket IDs 0..7, while
    the model predicts logits over the same 8 buckets. For MAE in years, both
    sides need to be represented in years.
    """

    if device is None:
        device = ground_truth.device
    age_bins = torch.tensor(
        [4.5, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 75.0],
        device=device,
        dtype=torch.float32,
    )
    bucket_ids = torch.clamp(ground_truth.long(), 0, 7)
    return age_bins[bucket_ids]


def compute_mae(predictions, ground_truth):
    """
    Compute Mean Absolute Error.
    
    Args:
        predictions: tensor - can be [B], [B, 1], or [B, C] (for age logits)
        ground_truth: tensor - should be [B] or [B, 1]
    
    Returns:
        float: MAE value
    """
    # Handle age logits [B, 8] - convert predictions and bucket targets to years.
    if len(predictions.shape) == 2 and predictions.shape[1] == 8:
        predictions = compute_age_from_logits(predictions)
        ground_truth = compute_age_targets_to_years(ground_truth, device=predictions.device)
    elif len(predictions.shape) > 1:
        predictions = predictions.squeeze()
    
    if len(ground_truth.shape) > 1:
        ground_truth = ground_truth.squeeze()
    
    predictions = predictions.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    
    mae = np.mean(np.abs(predictions - ground_truth))
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
    Compute recall at a given precision threshold for visibility prediction.
    
    Args:
        predictions: [B, 29] tensor (visibility scores per landmark)
        ground_truth: [B, 29] tensor (visibility labels, may be continuous)
        target_precision: desired precision threshold (default 0.8)
    
    Returns:
        float: recall at target precision
    """
    predictions = torch.sigmoid(predictions).cpu().numpy().flatten()
    ground_truth = ground_truth.cpu().numpy().flatten()
    
    # Ensure ground_truth is binary (some datasets may have continuous values)
    ground_truth = (ground_truth > 0.5).astype(int)
    
    # Compute precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(ground_truth, predictions)
    
    # Find the best recall among thresholds that satisfy the target precision.
    idx = np.where(precisions >= target_precision)[0]
    if len(idx) > 0:
        recall_at_prec = np.max(recalls[idx])
    else:
        recall_at_prec = 0.0
    
    return recall_at_prec


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
            
            # Create dataloader for single dataset
            if world_size > 1:
                from torch.utils.data.distributed import DistributedSampler
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False
                )
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=config.BATCH_SIZE,
                    sampler=sampler,
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
                    loss, _ = criterion(
                        predictions,
                        targets,
                        task_ids,
                        compute_individual=True
                    )
                    
                    dataset_loss += loss.item()
                    num_batches += 1
                    
                    # Compute task-specific metric
                    if task_name == 'segmentation':
                        if 'segmentation' in targets:
                            metric = compute_f1_score(seg_out, targets['segmentation'])
                            dataset_metric += metric
                        elif rank == 0 and not first_batch_checked:
                            print(f"      ⚠️ WARNING: 'segmentation' not in targets. Available keys: {list(targets.keys())}")
                            first_batch_checked = True
                    
                    elif task_name == 'landmark':
                        if 'landmark' in targets:
                            metric = compute_nme(landmark_out, targets['landmark'])
                            dataset_metric += metric
                        elif rank == 0 and not first_batch_checked:
                            print(f"      ⚠️ WARNING: 'landmark' not in targets. Available keys: {list(targets.keys())}")
                            first_batch_checked = True
                    
                    elif task_name == 'headpose':
                        if 'headpose' in targets:
                            # Headpose is [yaw, pitch, roll]
                            metric = compute_mae(headpose_out, targets['headpose'])
                            dataset_metric += metric
                    
                    elif task_name == 'attribute':
                        if 'attribute' in targets:
                            metric = compute_accuracy(attribute_out, targets['attribute'])
                            dataset_metric += metric
                        elif rank == 0 and not first_batch_checked:
                            print(f"      ⚠️ WARNING: 'attribute' not in targets. Available keys: {list(targets.keys())}")
                            first_batch_checked = True
                    
                    elif task_name == 'age':
                        if 'age' in targets:
                            # age_out is [B, 8] logits, compute_mae will convert to continuous
                            metric = compute_mae(age_out, targets['age'].float())
                            dataset_metric += metric
                    
                    elif task_name == 'gender':
                        if 'gender' in targets:
                            metric = compute_accuracy(gender_out, targets['gender'])
                            dataset_metric += metric
                    
                    elif task_name == 'race':
                        if 'race' in targets:
                            metric = compute_accuracy(race_out, targets['race'])
                            dataset_metric += metric
                    
                    elif task_name == 'visibility':
                        if 'visibility' in targets:
                            metric = compute_recall_at_precision(visibility_out, targets['visibility'])
                            dataset_metric += metric
                    
                    if rank == 0:
                        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Synchronize across GPUs
            if world_size > 1:
                # Gather losses from all ranks
                dataset_loss_tensor = torch.tensor(dataset_loss, device=device)
                dataset_metric_tensor = torch.tensor(dataset_metric, device=device)
                num_batches_tensor = torch.tensor(num_batches, device=device)
                
                torch.distributed.all_reduce(dataset_loss_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(dataset_metric_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(num_batches_tensor, op=torch.distributed.ReduceOp.SUM)
                
                dataset_loss = dataset_loss_tensor.item()
                dataset_metric = dataset_metric_tensor.item()
                num_batches = num_batches_tensor.item()
            
            # Average loss and metric for this dataset
            avg_dataset_loss = dataset_loss / num_batches if num_batches > 0 else 0.0
            avg_dataset_metric = dataset_metric / num_batches if num_batches > 0 else 0.0
            
            results[task_name][dataset_name] = {
                'loss': avg_dataset_loss,
                'metric': avg_dataset_metric
            }
            
            task_total_loss += dataset_loss
            task_total_metric += dataset_metric
            task_total_samples += num_batches
            
            if rank == 0:
                print(f"  ├─ {dataset_name:30s}: Loss={avg_dataset_loss:.6f}, {metric_names[task_name]}={avg_dataset_metric:.4f}")
        
        # Compute overall task loss and metric
        avg_task_loss = task_total_loss / task_total_samples if task_total_samples > 0 else 0.0
        avg_task_metric = task_total_metric / task_total_samples if task_total_samples > 0 else 0.0
        
        results[task_name]['overall'] = {
            'loss': avg_task_loss,
            'metric': avg_task_metric
        }
        
        total_loss += task_total_loss
        total_samples += task_total_samples
        
        if rank == 0:
            print(f"  └─ {'OVERALL':30s}: Loss={avg_task_loss:.6f}, {metric_names[task_name]}={avg_task_metric:.4f}")
    
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
            print("\nPlease ensure datasets are downloaded and extracted to ./datasets/")
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
    
    # DEBUG: Verify distributed sampling is working
    if rank == 0:
        print(f"\n🔍 DATALOADER VERIFICATION:")
        print(f"  Batches per GPU (rank 0): {len(train_loader)}")
        print(f"  Batch size per GPU: {config.BATCH_SIZE}")
        print(f"  World size: {world_size}")
        print(f"  Total samples per epoch: {len(train_loader) * config.BATCH_SIZE * world_size}")
    
    # All ranks check their first batch to ensure different data
    if world_size > 1:
        for batch_idx, (images, targets) in enumerate(train_loader):
            if batch_idx == 0:
                # Get first sample index or task_id as identifier
                task_ids = targets['task_id']
                first_task_id = task_ids[0].item()
                
                # Gather from all ranks to verify they're different
                all_first_ids = [None] * world_size
                dist.all_gather_object(all_first_ids, first_task_id)
                
                if rank == 0:
                    print(f"\n  First batch task_ids from each GPU: {all_first_ids}")
                    if len(set(all_first_ids)) < world_size:
                        print(f"  ⚠️ WARNING: Some GPUs have identical first batches!")
                        print(f"  DistributedSampler may not be working correctly!")
                    else:
                        print(f"  ✅ All GPUs have different data - DistributedSampler working!")
                
                dist.barrier()
            break
    
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
                    
                    if first:
                        print(f"{task_name:<20} {dataset_name:<30} {metrics['loss']:.6f}      {metric_name:<15} {metrics['metric']:.4f}")
                        first = False
                    else:
                        print(f"{'':<20} {dataset_name:<30} {metrics['loss']:.6f}      {metric_name:<15} {metrics['metric']:.4f}")
                
                # Print overall for this task
                if 'overall' in task_results:
                    print(f"{'':<20} {'[OVERALL]':<30} {task_results['overall']['loss']:.6f}      {metric_name:<15} {task_results['overall']['metric']:.4f}")
                print(f"{'─'*100}")
            
            print(f"{'TOTAL (All Tasks)':<20} {'':<30} {test_results['total']:.6f}")
            print(f"{'='*100}\n")
    
    finally:
        # Cleanup distributed training (always executed, even on exceptions)
        cleanup_distributed()


if __name__ == "__main__":
    main()

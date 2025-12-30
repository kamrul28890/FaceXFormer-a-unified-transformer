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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import argparse
from pathlib import Path

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
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    else:
        pbar = dataloader
    
    for batch_idx, (images, targets) in enumerate(pbar):
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
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for task_name, task_loss in individual_losses.items():
            if task_name not in task_losses:
                task_losses[task_name] = 0.0
            task_losses[task_name] += task_loss.item()
        
        # Update progress bar
        if rank == 0:
            pbar.set_postfix({'loss': loss.item()})
    
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
        print(f"World size: {world_size}")
        print(f"Batch size per GPU: {config.BATCH_SIZE}")
        print(f"Effective batch size: {config.BATCH_SIZE * world_size}")
        print(f"{'='*60}\n")
    
    # Set device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets (NO expression datasets)
    if rank == 0:
        print("Loading datasets...")
    
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
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
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
    best_test_loss = float('inf')
    
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
        
        # Train
        train_loss, train_task_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, device, rank
        )
        
        if rank == 0:
            print(f"\nTrain Loss: {train_loss:.4f}")
            print("Train Task Losses:")
            for task_name, task_loss in train_task_losses.items():
                print(f"  {task_name}: {task_loss:.4f}")
        
        # Test evaluation
        test_loss, test_task_losses = validate(
            model, test_loader, criterion, epoch, device, rank
        )
        
        if rank == 0:
            print(f"\nTest Loss: {test_loss:.4f}")
            print("Test Task Losses:")
            for task_name, task_loss in test_task_losses.items():
                print(f"  {task_name}: {task_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save checkpoint
        if rank == 0:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                save_checkpoint(
                    model, optimizer, scheduler, epoch, test_loss,
                    os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'),
                    rank
                )
            
            if epoch % config.SAVE_FREQ == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, test_loss,
                    os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth'),
                    rank
                )
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Best test loss: {best_test_loss:.4f}")
        print(f"{'='*60}\n")
    
    # Cleanup distributed training
    cleanup_distributed()


if __name__ == "__main__":
    main()

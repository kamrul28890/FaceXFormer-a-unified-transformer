"""
Simple single-GPU training script for FaceXFormer-main.
Use this for testing or single-GPU training.
For multi-GPU training, use train.py with torchrun.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

from config import config
from network.models.facexformer import FaceXFormer
from losses import MultiTaskLoss
from datasets import (
    CelebAMaskHQDataset, W300Dataset, W300LPDataset,
    CelebADataset, UTKFaceDataset, FairFaceDataset, COFWDataset,
    MultiLabelDatasetWrapper, create_multi_task_dataloader
)


def collate_fn(batch):
    """Custom collate to handle mixed-task batches."""
    images = torch.stack([item[0] for item in batch])
    
    all_keys = set()
    for item in batch:
        all_keys.update(item[1].keys())
    
    targets = {}
    for key in all_keys:
        values = []
        for item in batch:
            if key in item[1]:
                values.append(item[1][key])
            else:
                # Dummy values for missing targets
                if key == 'segmentation':
                    values.append(torch.zeros(224, 224, dtype=torch.long))
                elif key == 'landmark':
                    values.append(torch.zeros(136))
                elif key == 'headpose':
                    values.append(torch.zeros(3))
                elif key == 'attribute':
                    values.append(torch.zeros(40))
                elif key == 'task_id':
                    values.append(torch.tensor(-1))
                else:
                    values.append(torch.tensor(0))
        
        if len(values) > 0:
            targets[key] = torch.stack(values)
    
    return images, targets


def main():
    """Main training function."""
    print(f"\n{'='*60}")
    print(f"FaceXFormer-main Training (Single GPU)")
    print(f"{'='*60}\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    
    try:
        utkface_train = UTKFaceDataset('train')
        fairface_train = FairFaceDataset('train')
        
        train_datasets = {
            'segmentation': [CelebAMaskHQDataset('train')],
            'landmark': [W300Dataset('train')],
            'headpose': [W300LPDataset('train')],
            'attribute': [CelebADataset('train')],
            'age': [MultiLabelDatasetWrapper(utkface_train, 'age'), 
                    MultiLabelDatasetWrapper(fairface_train, 'age')],
            'gender': [MultiLabelDatasetWrapper(utkface_train, 'gender'),
                       MultiLabelDatasetWrapper(fairface_train, 'gender')],
            'race': [MultiLabelDatasetWrapper(utkface_train, 'race'),
                     MultiLabelDatasetWrapper(fairface_train, 'race')],
            'visibility': [COFWDataset('train')]
        }
        
        print("✓ Datasets loaded")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease download and extract datasets to ./datasets/")
        return
    
    # Create data loader
    train_loader = create_multi_task_dataloader(
        train_datasets,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        use_upsampling=True
    )
    
    print(f"✓ DataLoader created ({len(train_loader)} batches)")
    
    # Create model
    print("\nCreating model...")
    model = FaceXFormer().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created ({total_params:,} parameters)")
    
    # Create loss and optimizer
    criterion = MultiTaskLoss(config.LOSS_WEIGHTS).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
        print(f"{'='*60}")
        
        model.train()
        total_loss = 0.0
        task_losses = {}
        
        pbar = tqdm(train_loader, desc=f'Training')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            task_ids = targets['task_id'].to(device)
            
            for key in targets:
                targets[key] = targets[key].to(device)
            
            # Forward pass
            landmark_out, headpose_out, attribute_out, visibility_out, \
            age_out, gender_out, race_out, seg_out = model(images, targets, task_ids)
            
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
                predictions, targets, task_ids, compute_individual=True
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
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Print epoch results
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} - Average Loss: {avg_loss:.4f}")
        print("Task Losses:")
        for task_name, task_loss in task_losses.items():
            print(f"  {task_name}: {task_loss / len(train_loader):.4f}")
        
        # Save checkpoint
        if epoch % config.SAVE_FREQ == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'checkpoint_epoch_{epoch}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

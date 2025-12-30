"""
Multi-task loss function for FaceXFormer-main.
Adapted from facexformer-my, excluding expression recognition loss.

Loss components:
- L_seg: Face parsing (Dice + CrossEntropy)
- L_ind: Landmark detection (L1)
- L_hpe: Head pose (L1 on Euler angles)
- L_attr: Attributes (Binary CrossEntropy)
- L_a: Age (L1 or CrossEntropy)
- L_g/r: Gender/Race (CrossEntropy)
- L_vis: Visibility (MSE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class DiceLoss(nn.Module):
    """Dice loss for segmentation with ignore index support."""
    def __init__(self, smooth: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions [B, C, H, W]
            target: Ground truth [B, H, W] (class indices)
        """
        # Create mask for valid pixels (not ignore_index)
        valid_mask = (target != self.ignore_index).unsqueeze(1)  # [B, 1, H, W]
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device)
        
        num_classes = pred.shape[1]
        pred_soft = F.softmax(pred, dim=1)
        
        # Convert target to one-hot, but only for valid pixels
        target_one_hot = torch.zeros_like(pred_soft)
        valid_target = target.clone()
        valid_target[target == self.ignore_index] = 0  # temporary for one_hot
        # Clamp to prevent CUDA device-side asserts in scatter_ operation
        valid_target = torch.clamp(valid_target, 0, num_classes - 1)
        target_one_hot.scatter_(1, valid_target.unsqueeze(1), 1)
        target_one_hot = target_one_hot * valid_mask  # zero out ignored pixels (non-in-place)
        
        # Apply mask to predictions (non-in-place to avoid breaking gradient graph)
        pred_soft_masked = pred_soft * valid_mask
        
        # Compute dice only on valid pixels
        intersection = (pred_soft_masked * target_one_hot).sum(dim=(2, 3))
        union = pred_soft_masked.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class MultiTaskLoss(nn.Module):
    """Multi-task loss for FaceXFormer-main (NO expression loss)."""
    
    def __init__(self, loss_weights: Dict[str, float]):
        super().__init__()
        self.loss_weights = loss_weights
        
        # Segmentation losses
        self.dice_loss = DiceLoss(ignore_index=-100)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Regression losses
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Classification losses
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        task_ids: torch.Tensor,
        compute_individual: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Compute multi-task loss with full-batch computation for DDP consistency.
        
        Args:
            predictions: Dict with full batch predictions for all tasks:
                - 'landmark_output': [B, 136] for all samples
                - 'headpose_output': [B, 3] for all samples
                - 'attribute_output': [B, 40] for all samples
                - 'visibility_output': [B, 29] for all samples
                - 'age_output': [B, 8] for all samples
                - 'gender_output': [B, 2] for all samples
                - 'race_output': [B, 5] for all samples
                - 'seg_output': [B, 11, H, W] for all samples
            labels: Ground truth dictionary (full batch)
            task_ids: Task IDs for each sample [B]
            compute_individual: Whether to return individual task losses
            
        Returns:
            total_loss and optionally individual_losses dict
        """
        losses = {}
        total_loss = None
        device = next(iter(predictions.values())).device
        
        batch_size = task_ids.shape[0]
        
        # Segmentation loss - full batch with ignore index
        if 'seg_output' in predictions:
            seg_pred = predictions['seg_output']  # [B, 11, H, W]
            if 'segmentation' in labels:
                seg_target = labels['segmentation'].to(device)  # [B, H, W]
                # Clamp to valid class range to prevent CUDA device-side asserts
                seg_target = torch.clamp(seg_target, 0, 10)  # 11 classes: 0-10
                
                # Create mask for segmentation samples (task_id=0)
                seg_mask = (task_ids == 0).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
                
                if seg_mask.any():
                    # Apply ignore index to non-segmentation samples
                    seg_target_masked = seg_target.clone()
                    seg_target_masked[~seg_mask.squeeze(-1).squeeze(-1)] = -100  # ignore_index
                    
                    # Compute loss only on masked regions
                    dice = self.dice_loss(seg_pred, seg_target_masked)
                    ce = self.ce_loss(seg_pred, seg_target_masked)
                    seg_loss = (dice + ce) / 2.0
                    
                    losses['seg'] = seg_loss
                    weighted_loss = self.loss_weights['seg'] * seg_loss
                    total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        
        # Landmark loss - full batch with masking
        if 'landmark_output' in predictions:
            lm_pred = predictions['landmark_output']  # [B, 136]
            if 'landmark' in labels:
                lm_target = labels['landmark'].to(device)  # [B, 136]
                
                # Create mask for landmark samples (task_id=1)
                lm_mask = (task_ids == 1).unsqueeze(-1)  # [B, 1]
                
                if lm_mask.any():
                    # Compute L1 loss only on landmark samples
                    lm_loss = self.l1_loss(lm_pred * lm_mask, lm_target * lm_mask) / lm_mask.sum().clamp(min=1.0)
                    
                    losses['ind'] = lm_loss
                    weighted_loss = self.loss_weights['ind'] * lm_loss
                    total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        
        # Head pose loss - full batch with masking
        if 'headpose_output' in predictions:
            pose_pred = predictions['headpose_output']  # [B, 3]
            if 'headpose' in labels:
                pose_target = labels['headpose'].to(device)  # [B, 3]
                
                # Create mask for head pose samples (task_id=2)
                pose_mask = (task_ids == 2).unsqueeze(-1)  # [B, 1]
                
                if pose_mask.any():
                    # Compute L1 loss only on head pose samples
                    pose_loss = self.l1_loss(pose_pred * pose_mask, pose_target * pose_mask) / pose_mask.sum().clamp(min=1.0)
                    
                    losses['hpe'] = pose_loss
                    weighted_loss = self.loss_weights['hpe'] * pose_loss
                    total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        
        # Attribute loss
        if 'attribute_output' in predictions and predictions['attribute_output'].shape[0] > 0:
            attr_pred = predictions['attribute_output']
            attr_indices = (task_ids == 3).nonzero(as_tuple=True)[0]
            if 'attribute' in labels and len(attr_indices) > 0:
                attr_target = labels['attribute'][attr_indices].to(device)
                
                # Handle batch size mismatch
                if attr_pred.shape[0] != attr_target.shape[0]:
                    min_batch = min(attr_pred.shape[0], attr_target.shape[0])
                    attr_pred = attr_pred[:min_batch]
                    attr_target = attr_target[:min_batch]
                
                attr_loss = self.bce_loss(attr_pred, attr_target.float())
                losses['attr'] = attr_loss
                weighted_loss = self.loss_weights['attr'] * attr_loss
                total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        
        # Age loss
        if 'age_output' in predictions and predictions['age_output'].shape[0] > 0:
            age_pred = predictions['age_output']
            age_indices = (task_ids == 4).nonzero(as_tuple=True)[0]
            if 'age' in labels and len(age_indices) > 0:
                age_target = labels['age'][age_indices].to(device)
                
                # Handle batch size mismatch
                if age_pred.shape[0] != age_target.shape[0]:
                    min_batch = min(age_pred.shape[0], age_target.shape[0])
                    age_pred = age_pred[:min_batch]
                    age_target = age_target[:min_batch]
                
                # Age can be regression or classification
                if age_target.dtype in [torch.long, torch.int]:
                    # Clamp to valid range to prevent CUDA device-side asserts
                    age_target = torch.clamp(age_target.long(), 0, 7)  # 8 age groups (0-7)
                    # Use F.cross_entropy with ignore_index for robustness
                    age_loss = F.cross_entropy(age_pred, age_target, ignore_index=-1)
                else:
                    age_loss = self.l1_loss(age_pred, age_target.float())
                
                losses['a'] = age_loss
                weighted_loss = self.loss_weights['a'] * age_loss
                total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        
        # Gender loss
        if 'gender_output' in predictions and predictions['gender_output'].shape[0] > 0:
            gender_pred = predictions['gender_output']
            gender_indices = (task_ids == 5).nonzero(as_tuple=True)[0]
            if 'gender' in labels and len(gender_indices) > 0:
                gender_target = labels['gender'][gender_indices].to(device)
                
                # Handle batch size mismatch
                if gender_pred.shape[0] != gender_target.shape[0]:
                    min_batch = min(gender_pred.shape[0], gender_target.shape[0])
                    gender_pred = gender_pred[:min_batch]
                    gender_target = gender_target[:min_batch]
                
                # Clamp to valid range to prevent CUDA device-side asserts
                gender_target = torch.clamp(gender_target.long(), 0, 1)  # 2 classes
                # Use F.cross_entropy with ignore_index for robustness
                gender_loss = F.cross_entropy(gender_pred, gender_target, ignore_index=-1)
                
                losses['gender'] = gender_loss
                weighted_loss = self.loss_weights['g/r'] * gender_loss * 0.5
                total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        
        # Race loss
        if 'race_output' in predictions and predictions['race_output'].shape[0] > 0:
            race_pred = predictions['race_output']
            race_indices = (task_ids == 6).nonzero(as_tuple=True)[0]
            if 'race' in labels and len(race_indices) > 0:
                race_target = labels['race'][race_indices].to(device)
                
                # Handle batch size mismatch
                if race_pred.shape[0] != race_target.shape[0]:
                    min_batch = min(race_pred.shape[0], race_target.shape[0])
                    race_pred = race_pred[:min_batch]
                    race_target = race_target[:min_batch]
                
                # Clamp to valid range to prevent CUDA device-side asserts
                race_target = torch.clamp(race_target.long(), 0, 4)  # 5 classes
                # Use F.cross_entropy with ignore_index for robustness
                race_loss = F.cross_entropy(race_pred, race_target, ignore_index=-1)
                
                losses['race'] = race_loss
                weighted_loss = self.loss_weights['g/r'] * race_loss * 0.5
                total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        
        # Visibility loss
        if 'visibility_output' in predictions and predictions['visibility_output'].shape[0] > 0:
            vis_pred = predictions['visibility_output']
            vis_indices = (task_ids == 7).nonzero(as_tuple=True)[0]
            if 'visibility' in labels and len(vis_indices) > 0:
                vis_target = labels['visibility'][vis_indices].to(device)
                
                # Handle batch size mismatch
                if vis_pred.shape[0] != vis_target.shape[0]:
                    min_batch = min(vis_pred.shape[0], vis_target.shape[0])
                    vis_pred = vis_pred[:min_batch]
                    vis_target = vis_target[:min_batch]
                
                vis_loss = self.mse_loss(vis_pred, vis_target.float())
                losses['vis'] = vis_loss
                weighted_loss = self.loss_weights['vis'] * vis_loss
                total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        
        # If no losses were computed, return a zero tensor with gradient
        if total_loss is None:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        if compute_individual:
            return total_loss, losses
        
        return total_loss, {}


if __name__ == "__main__":
    # Test the loss
    from config import config
    
    loss_fn = MultiTaskLoss(config.LOSS_WEIGHTS)
    
    # Create dummy predictions
    batch_size = 8
    predictions = {
        'seg_output': torch.randn(batch_size, 11, 56, 56),
        'landmark_output': torch.randn(batch_size, 136),
        'headpose_output': torch.randn(batch_size, 3),
        'attribute_output': torch.randn(batch_size, 40),
        'age_output': torch.randn(batch_size, 8),
        'gender_output': torch.randn(batch_size, 2),
        'race_output': torch.randn(batch_size, 5),
        'visibility_output': torch.randn(batch_size, 29),
    }
    
    # Create dummy labels (mix of tasks)
    task_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 5])  # Mix of tasks
    labels = {
        'segmentation': torch.randint(0, 11, (batch_size, 56, 56)),
        'landmark': torch.randn(batch_size, 136),
        'headpose': torch.randn(batch_size, 3),
        'attribute': torch.randn(batch_size, 40),
        'age': torch.randint(0, 8, (batch_size,)),
        'gender': torch.randint(0, 2, (batch_size,)),
        'race': torch.randint(0, 5, (batch_size,)),
        'visibility': torch.randn(batch_size, 29),
    }
    
    total_loss, individual_losses = loss_fn(predictions, labels, task_ids, compute_individual=True)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print("\nIndividual losses:")
    for task_name, loss in individual_losses.items():
        print(f"  {task_name}: {loss.item():.4f}")

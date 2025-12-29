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
    """Dice loss for segmentation."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions [B, C, H, W]
            target: Ground truth [B, H, W] (class indices)
        """
        num_classes = pred.shape[1]
        pred_soft = F.softmax(pred, dim=1)
        
        # Convert target to one-hot
        target_one_hot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
        
        # Compute dice
        intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class MultiTaskLoss(nn.Module):
    """Multi-task loss for FaceXFormer-main (NO expression loss)."""
    
    def __init__(self, loss_weights: Dict[str, float]):
        super().__init__()
        self.loss_weights = loss_weights
        
        # Segmentation losses
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
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
        Compute multi-task loss.
        
        NOTE: The model already filters outputs by task_id internally, so predictions
        only contain samples for their respective tasks. We extract corresponding labels
        using task_ids but don't filter predictions again.
        
        Args:
            predictions: Dict with keys matching task IDs (already filtered by model):
                - 'seg_output': [N, 11, H, W] for segmentation (N = num seg samples)
                - 'landmark_output': [N, 136] for landmarks (N = num landmark samples)
                - 'headpose_output': [N, 3] for pose
                - 'attribute_output': [N, 40] for attributes
                - 'age_output': [N, 8] for age groups
                - 'gender_output': [N, 2] for gender
                - 'race_output': [N, 5] for race
                - 'visibility_output': [N, 29] for visibility
            labels: Ground truth dictionary (full batch)
            task_ids: Task IDs for each sample in original batch [B]
            compute_individual: Whether to return individual task losses
            
        Returns:
            total_loss and optionally individual_losses dict
        """
        losses = {}
        total_loss = None
        device = next(iter(predictions.values())).device
        
        # Segmentation loss
        if 'seg_output' in predictions and predictions['seg_output'].shape[0] > 0:
            seg_pred = predictions['seg_output']
            # Extract labels for segmentation samples (task_id=0)
            seg_indices = (task_ids == 0).nonzero(as_tuple=True)[0]
            if 'segmentation' not in labels or len(seg_indices) == 0:
                pass
            else:
                seg_target = labels['segmentation'][seg_indices].to(device)
                
                # Handle batch size mismatch
                if seg_pred.shape[0] != seg_target.shape[0]:
                    min_batch = min(seg_pred.shape[0], seg_target.shape[0])
                    seg_pred = seg_pred[:min_batch]
                    seg_target = seg_target[:min_batch]
                
                if seg_target.dtype != torch.long:
                    seg_target = seg_target.long()
                seg_target = torch.clamp(seg_target, 0, 10)  # 11 classes (0-10)
                
                dice = self.dice_loss(seg_pred, seg_target)
                ce = self.ce_loss(seg_pred, seg_target)
                seg_loss = (dice + ce) / 2.0
                
                losses['seg'] = seg_loss
                weighted_loss = self.loss_weights['seg'] * seg_loss
                total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        
        # Landmark loss
        if 'landmark_output' in predictions and predictions['landmark_output'].shape[0] > 0:
            lm_pred = predictions['landmark_output']
            lm_indices = (task_ids == 1).nonzero(as_tuple=True)[0]
            if 'landmark' in labels and len(lm_indices) > 0:
                lm_target = labels['landmark'][lm_indices].to(device)
                
                # Handle batch size mismatch
                if lm_pred.shape[0] != lm_target.shape[0]:
                    min_batch = min(lm_pred.shape[0], lm_target.shape[0])
                    lm_pred = lm_pred[:min_batch]
                    lm_target = lm_target[:min_batch]
                
                lm_loss = self.l1_loss(lm_pred, lm_target)
                losses['ind'] = lm_loss
                weighted_loss = self.loss_weights['ind'] * lm_loss
                total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        
        # Head pose loss
        if 'headpose_output' in predictions and predictions['headpose_output'].shape[0] > 0:
            pose_pred = predictions['headpose_output']
            pose_indices = (task_ids == 2).nonzero(as_tuple=True)[0]
            if 'headpose' in labels and len(pose_indices) > 0:
                pose_target = labels['headpose'][pose_indices].to(device)
                
                # Handle batch size mismatch
                if pose_pred.shape[0] != pose_target.shape[0]:
                    min_batch = min(pose_pred.shape[0], pose_target.shape[0])
                    pose_pred = pose_pred[:min_batch]
                    pose_target = pose_target[:min_batch]
                
                pose_loss = self.l1_loss(pose_pred, pose_target)
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
                    age_target = torch.clamp(age_target.long(), 0, 7)  # 8 age groups (0-7)
                    age_loss = self.ce_loss(age_pred, age_target)
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
                
                gender_target = torch.clamp(gender_target.long(), 0, 1)  # 2 classes
                gender_loss = self.ce_loss(gender_pred, gender_target)
                
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
                
                race_target = torch.clamp(race_target.long(), 0, 4)  # 5 classes
                race_loss = self.ce_loss(race_pred, race_target)
                
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

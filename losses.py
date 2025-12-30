"""
Multi-task loss function for FaceXFormer-main.
Adapted from facexformer-my, excluding expression recognition loss.

Loss components:
- L_seg: Face parsing (Dice + CrossEntropy)
- L_ind: Landmark detection (STAR loss or L1)
- L_hpe: Head pose (Geodesic loss or L1 on Euler angles)
- L_attr: Attributes (Binary CrossEntropy)
- L_a: Age (mean of L1 + CE)
- L_g/r: Gender/Race (CrossEntropy)
- L_vis: Visibility (Binary CrossEntropy)
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
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        
        # Regression losses
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        
        # Classification losses
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        
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
            predictions: Dict with full batch predictions for all tasks
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
        
        # =====================================================================
        # 1. SEGMENTATION LOSS (L_seg = mean of Dice + CE)
        # =====================================================================
        if 'seg_output' in predictions:
            seg_pred = predictions['seg_output']  # [B, 11, H, W]
            if 'segmentation' in labels:
                seg_target = labels['segmentation'].to(device)  # [B, H, W]
                # Clamp to valid class range
                seg_target = torch.clamp(seg_target, 0, 10)  # 11 classes: 0-10
                
                # Create mask for segmentation samples (task_id=0)
                seg_mask = (task_ids == 0)
                
                if seg_mask.any():
                    # Apply ignore index to non-segmentation samples
                    seg_target_masked = seg_target.clone()
                    seg_target_masked[~seg_mask] = -100  # ignore_index
                    
                    # Compute loss (Dice and CE are already averaged internally)
                    dice = self.dice_loss(seg_pred, seg_target_masked)
                    ce = self.ce_loss(seg_pred, seg_target_masked)
                    seg_loss = (dice + ce) / 2.0  # Mean of dice and CE
                    
                    losses['seg'] = seg_loss
                    weighted_loss = self.loss_weights['seg'] * seg_loss
                    total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        
        # =====================================================================
        # 2. LANDMARK LOSS (L_ind = L1 or STAR loss)
        # =====================================================================
        if 'landmark_output' in predictions:
            lm_pred = predictions['landmark_output']  # [B, 136]
            if 'landmark' in labels:
                lm_target = labels['landmark'].to(device)  # [B, 136]
                
                # Create mask for landmark samples (task_id=1)
                lm_mask = (task_ids == 1)
                
                if lm_mask.any():
                    # Extract only landmark samples
                    lm_pred_masked = lm_pred[lm_mask]  # [N, 136]
                    lm_target_masked = lm_target[lm_mask]  # [N, 136]
                    
                    # L1 loss on landmarks (coordinates in pixel space 0-224)
                    # NOTE: Landmarks should ideally be normalized to [0, 1]
                    # If they're in pixel coordinates, divide by image size for stability
                    lm_loss = self.l1_loss(lm_pred_masked / 224.0, lm_target_masked / 224.0)
                    
                    losses['ind'] = lm_loss
                    weighted_loss = self.loss_weights['ind'] * lm_loss
                    total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        
        # =====================================================================
        # 3. HEAD POSE LOSS (L_hpe = L1 on Euler angles)
        # =====================================================================
        if 'headpose_output' in predictions:
            pose_pred = predictions['headpose_output']  # [B, 3]
            if 'headpose' in labels:
                pose_target = labels['headpose'].to(device)  # [B, 3]
                
                # Create mask for head pose samples (task_id=2)
                pose_mask = (task_ids == 2)
                
                if pose_mask.any():
                    # Extract only head pose samples
                    pose_pred_masked = pose_pred[pose_mask]  # [N, 3]
                    pose_target_masked = pose_target[pose_mask]  # [N, 3]
                    
                    # L1 loss on Euler angles (in radians, typically -π to π)
                    # Normalize by π for stability
                    import math
                    pose_loss = self.l1_loss(pose_pred_masked / math.pi, pose_target_masked / math.pi)
                    
                    losses['hpe'] = pose_loss
                    weighted_loss = self.loss_weights['hpe'] * pose_loss
                    total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        
        # =====================================================================
        # 4. ATTRIBUTE LOSS (L_attr = BCE)
        # =====================================================================
        if 'attribute_output' in predictions:
            attr_pred = predictions['attribute_output']  # [B, 40]
            if 'attribute' in labels:
                # Create mask for attribute samples (task_id=3)
                attr_mask = (task_ids == 3)
                
                if attr_mask.any():
                    # Extract only attribute samples
                    attr_pred_masked = attr_pred[attr_mask]
                    attr_target_masked = labels['attribute'][attr_mask].to(device)
                    
                    # BCE loss (already averaged over 40 attributes)
                    attr_loss = self.bce_loss(attr_pred_masked, attr_target_masked.float())
                    
                    losses['attr'] = attr_loss
                    weighted_loss = self.loss_weights['attr'] * attr_loss
                    total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        
        # =====================================================================
        # 5. AGE LOSS (L_a = mean of L1 + CE)
        # =====================================================================
        if 'age_output' in predictions:
            age_pred = predictions['age_output']  # [B, 8] - logits for 8 age groups
            if 'age' in labels:
                # Create mask for age samples (task_id=4 or 5 depending on dataset)
                age_mask = (task_ids == 4) | (task_ids == 5)  # Support both task IDs
                
                if age_mask.any():
                    # Extract only age samples
                    age_pred_masked = age_pred[age_mask]
                    age_target_masked = labels['age'][age_mask].to(device)
                    
                    # Clamp to valid range
                    age_target_masked = torch.clamp(age_target_masked.long(), 0, 7)  # 8 age groups
                    
                    # Classification loss (CE)
                    age_ce = F.cross_entropy(age_pred_masked, age_target_masked, reduction='mean')
                    
                    # Regression loss (L1) - convert class to age value
                    # Age groups: [0-2, 4-6, 8-13, 15-20, 25-32, 38-43, 48-53, 60+]
                    age_bins = torch.tensor([1, 5, 10, 17, 28, 40, 50, 65], 
                                           device=device, dtype=torch.float32)
                    age_gt_value = age_bins[age_target_masked]  # [N]
                    
                    # Predicted age value (weighted sum of bins)
                    age_probs = F.softmax(age_pred_masked, dim=1)  # [N, 8]
                    age_pred_value = (age_probs * age_bins).sum(dim=1)  # [N]
                    
                    # Normalize L1 loss by age range (100 years) for stability
                    age_l1 = self.l1_loss(age_pred_value / 100.0, age_gt_value / 100.0)
                    
                    # Mean of CE and normalized L1
                    age_loss = (age_ce + age_l1) / 2.0
                    
                    losses['a'] = age_loss
                    weighted_loss = self.loss_weights['a'] * age_loss
                    total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        
        # =====================================================================
        # 6. GENDER LOSS (L_g/r part 1 = CE)
        # =====================================================================
        if 'gender_output' in predictions:
            gender_pred = predictions['gender_output']  # [B, 2]
            if 'gender' in labels:
                # Create mask for gender samples (task_id=5 or 6)
                gender_mask = (task_ids == 5) | (task_ids == 6)
                
                if gender_mask.any():
                    # Extract only gender samples
                    gender_pred_masked = gender_pred[gender_mask]
                    gender_target_masked = labels['gender'][gender_mask].to(device)
                    
                    # Clamp to valid range
                    gender_target_masked = torch.clamp(gender_target_masked.long(), 0, 1)  # 2 classes
                    
                    # CE loss
                    gender_loss = F.cross_entropy(gender_pred_masked, gender_target_masked, reduction='mean')
                    
                    losses['gender'] = gender_loss
                    weighted_loss = self.loss_weights['g/r'] * gender_loss
                    total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        
        # =====================================================================
        # 7. RACE LOSS (L_g/r part 2 = CE)
        # =====================================================================
        if 'race_output' in predictions:
            race_pred = predictions['race_output']  # [B, 5]
            if 'race' in labels:
                # Create mask for race samples (task_id=6 or 7)
                race_mask = (task_ids == 6) | (task_ids == 7)
                
                if race_mask.any():
                    # Extract only race samples
                    race_pred_masked = race_pred[race_mask]
                    race_target_masked = labels['race'][race_mask].to(device)
                    
                    # Clamp to valid range
                    race_target_masked = torch.clamp(race_target_masked.long(), 0, 4)  # 5 classes
                    
                    # CE loss
                    race_loss = F.cross_entropy(race_pred_masked, race_target_masked, reduction='mean')
                    
                    losses['race'] = race_loss
                    weighted_loss = self.loss_weights['g/r'] * race_loss
                    total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        
        # =====================================================================
        # 8. VISIBILITY LOSS (L_vis = BCE)
        # =====================================================================
        if 'visibility_output' in predictions:
            vis_pred = predictions['visibility_output']  # [B, 29]
            if 'visibility' in labels:
                # Create mask for visibility samples (task_id=7 or 8)
                vis_mask = (task_ids == 7) | (task_ids == 8)
                
                if vis_mask.any():
                    # Extract only visibility samples
                    vis_pred_masked = vis_pred[vis_mask]
                    vis_target_masked = labels['visibility'][vis_mask].to(device)
                    
                    # BCE loss
                    vis_loss = self.bce_loss(vis_pred_masked, vis_target_masked.float())
                    
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
        'landmark_output': torch.randn(batch_size, 136) * 224,  # Pixel coordinates
        'headpose_output': torch.randn(batch_size, 3) * 3.14,  # Radians
        'attribute_output': torch.randn(batch_size, 40),
        'age_output': torch.randn(batch_size, 8),
        'gender_output': torch.randn(batch_size, 2),
        'race_output': torch.randn(batch_size, 5),
        'visibility_output': torch.randn(batch_size, 29),
    }
    
    # Create dummy labels (mix of tasks)
    task_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])  # One of each task
    labels = {
        'segmentation': torch.randint(0, 11, (batch_size, 56, 56)),
        'landmark': torch.randn(batch_size, 136) * 224,  # Pixel coordinates
        'headpose': torch.randn(batch_size, 3) * 3.14,  # Radians
        'attribute': torch.randint(0, 2, (batch_size, 40)),
        'age': torch.randint(0, 8, (batch_size,)),
        'gender': torch.randint(0, 2, (batch_size,)),
        'race': torch.randint(0, 5, (batch_size,)),
        'visibility': torch.randint(0, 2, (batch_size, 29)).float(),
    }
    
    total_loss, individual_losses = loss_fn(predictions, labels, task_ids, compute_individual=True)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print("\nIndividual losses:")
    for task_name, loss in individual_losses.items():
        print(f"  {task_name}: {loss.item():.4f}")
    
    print("\n✅ Expected loss range: 2-10")
    print(f"{'⚠️ ISSUE' if total_loss.item() > 50 else '✅ OK'}: Current loss is {total_loss.item():.2f}")

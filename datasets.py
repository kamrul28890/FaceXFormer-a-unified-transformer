"""
Multi-task dataset loader for FaceXFormer-main.
Adapted from facexformer-my, removing expression recognition task.

Handles:
- Face parsing (CelebAMaskHQ)
- Landmark detection (300W, 300W-LP for pose)
- Head pose (300W-LP)
- Attributes (CelebA)
- Age/Gender/Race (UTKFace, FairFace)
- Visibility (COFW)

Expression datasets (RAF-DB, AffectNet) are EXCLUDED.
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.dataloader import default_collate
import numpy as np
from typing import Dict, List, Optional
import random
import math
import cv2
from PIL import Image, ImageOps, ImageFilter, ImageDraw
from pathlib import Path
import pickle
import pandas as pd
import time

# Import config for metadata checking
try:
    from config import config
except ImportError:
    # Fallback if config not available during import
    config = None


def age_to_bucket(age):
    """
    Convert continuous age to bucket index (0-7).
    
    Buckets (from authors):
        0: 0-9
        1: 10-19
        2: 20-29
        3: 30-39
        4: 40-49
        5: 50-59
        6: 60-69
        7: 70+
    
    Args:
        age: int or float, age in years
    
    Returns:
        int: bucket index 0-7
    """
    if age < 10:
        return 0
    elif age < 20:
        return 1
    elif age < 30:
        return 2
    elif age < 40:
        return 3
    elif age < 50:
        return 4
    elif age < 60:
        return 5
    elif age < 70:
        return 6
    else:
        return 7


def simple_augmentation(image, target_size=None):
    """Simple augmentation: resize and normalize."""
    if target_size is None:
        target_size = config.IMG_SIZE if config else 224
    image = image.resize((target_size, target_size), Image.BILINEAR)
    image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image - mean) / std
    return image


# ---------------------------------------------------------------------------
# ArcFace 5-point template (112×112), scaled to target_size at runtime.
# Points: left-eye-centre, right-eye-centre, nose-tip, left-mouth, right-mouth
# ---------------------------------------------------------------------------
_ARCFACE_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# Flip index mapping for 68-point IBUG/300W landmarks.
# FLIP_INDICES_68[i] is the symmetric counterpart of landmark i.
_FLIP_INDICES_68 = [
    16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,   # jaw
    26, 25, 24, 23, 22, 21, 20, 19, 18, 17,                        # eyebrows
    27, 28, 29, 30,                                                  # nose bridge
    35, 34, 33, 32, 31,                                             # nose base
    45, 44, 43, 42, 47, 46,                                         # left eye  → right
    39, 38, 37, 36, 41, 40,                                         # right eye → left
    54, 53, 52, 51, 50, 49, 48,                                     # outer upper lip
    59, 58, 57, 56, 55,                                             # outer lower lip
    64, 63, 62, 61, 60,                                             # inner upper lip
    67, 66, 65,                                                     # inner lower lip
]


# ---------------------------------------------------------------------------
# Low-level geometric helpers (image = PIL, lm = np.ndarray (N,2) in pixels)
# ---------------------------------------------------------------------------

def _pil_to_tensor(pil_img):
    """Convert PIL image to normalised ImageNet tensor."""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std


def _gamma_adjust_pil(pil_img, gamma):
    """Apply gamma correction to a PIL image."""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    arr = np.power(np.clip(arr, 0, 1), gamma)
    return Image.fromarray((arr * 255).astype(np.uint8))


def _rotate_img_lm(pil_img, lm, angle, size):
    """Rotate PIL image and (N,2) landmarks by *angle* degrees CCW around centre."""
    pil_img = pil_img.rotate(angle, expand=False)
    cx, cy = size / 2.0, size / 2.0
    rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    xc = lm[:, 0] - cx
    yc = lm[:, 1] - cy
    lm_out = lm.copy()
    lm_out[:, 0] = cos_a * xc - sin_a * yc + cx
    lm_out[:, 1] = sin_a * xc + cos_a * yc + cy
    return pil_img, lm_out


def _scale_img_lm(pil_img, lm, scale, size):
    """Scale PIL image and landmarks by *scale* around image centre."""
    cx, cy = size / 2.0, size / 2.0
    new_size = max(1, int(size * scale))
    scaled = pil_img.resize((new_size, new_size), Image.BILINEAR)
    result = Image.new('RGB', (size, size), (0, 0, 0))
    px = (size - new_size) // 2
    py = (size - new_size) // 2
    result.paste(scaled, (px, py))
    lm_out = lm.copy()
    lm_out[:, 0] = (lm[:, 0] - cx) * scale + cx
    lm_out[:, 1] = (lm[:, 1] - cy) * scale + cy
    return result, lm_out


def _translate_img_lm(pil_img, lm, tx, ty, size):
    """Translate PIL image and landmarks by (tx, ty) pixels."""
    result = Image.new('RGB', (size, size), (0, 0, 0))
    result.paste(pil_img, (int(tx), int(ty)))
    lm_out = lm.copy()
    lm_out[:, 0] = lm[:, 0] + tx
    lm_out[:, 1] = lm[:, 1] + ty
    return result, lm_out


def _flip_img_lm(pil_img, lm, size):
    """Flip PIL image and 68-point landmarks horizontally."""
    pil_img = ImageOps.mirror(pil_img)
    lm_out = lm[_FLIP_INDICES_68].copy()
    lm_out[:, 0] = size - 1 - lm_out[:, 0]
    return pil_img, lm_out


def _random_occlusion(pil_img, size):
    """Paste a random-colour rectangle onto the image."""
    draw = ImageDraw.Draw(pil_img)
    occ = random.randint(int(size * 0.1), int(size * 0.4))
    x1 = random.randint(0, size - occ)
    y1 = random.randint(0, size - occ)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    draw.rectangle([x1, y1, x1 + occ, y1 + occ], fill=color)
    return pil_img


# ---------------------------------------------------------------------------
# Task-specific augmentation functions
# ---------------------------------------------------------------------------

def align_face_arcface(pil_image, landmarks_68, target_size):
    """
    Align a face image using 5 keypoints derived from 68-point annotations and
    the scaled ArcFace template.  Returns (aligned_PIL_image, aligned_landmarks).
    aligned_landmarks are in [0, target_size] pixel space.
    Falls back to a plain resize if the transform cannot be estimated.
    """
    img_np = np.array(pil_image)
    H, W = img_np.shape[:2]

    # Derive 5 keypoints from 68 landmarks
    left_eye   = landmarks_68[36:42].mean(axis=0)
    right_eye  = landmarks_68[42:48].mean(axis=0)
    nose_tip   = landmarks_68[30]
    left_mouth = landmarks_68[48]
    right_mouth = landmarks_68[54]
    src_pts = np.array([left_eye, right_eye, nose_tip, left_mouth, right_mouth],
                       dtype=np.float32)

    dst_pts = _ARCFACE_TEMPLATE_112 * (target_size / 112.0)

    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if M is None:
        # Fallback: simple resize, scale landmarks proportionally
        img_aligned = cv2.resize(img_np, (target_size, target_size))
        lm = landmarks_68.copy().astype(np.float32)
        lm[:, 0] = lm[:, 0] / W * target_size
        lm[:, 1] = lm[:, 1] / H * target_size
    else:
        img_aligned = cv2.warpAffine(img_np, M, (target_size, target_size))
        ones = np.ones((len(landmarks_68), 1), dtype=np.float32)
        lm_h = np.concatenate([landmarks_68.astype(np.float32), ones], axis=1)
        lm = (M @ lm_h.T).T  # (68, 2)

    return Image.fromarray(img_aligned), lm


def augment_landmark_detection(pil_image, landmarks_68, target_size, is_train=True):
    """
    Full pipeline for the landmark detection task:
      Train: ArcFace alignment + geometric augmentations + photometric augmentations
      Test:  ArcFace alignment only (no augmentation)

    Paper augmentations: rotation ±18°, scaling ±10%, translation 5%×size,
    horizontal flip 50%, gray 20%, Gaussian blur 30%, occlusion 40%, gamma 20%.

    Returns:
        img_tensor  – (3, target_size, target_size) normalised tensor
        lm_flat     – (136,) landmark tensor in [0, 1] space
    """
    pil_img, lm = align_face_arcface(pil_image, landmarks_68, target_size)

    if is_train:
        # Random rotation ±18°
        angle = random.uniform(-18, 18)
        pil_img, lm = _rotate_img_lm(pil_img, lm, angle, target_size)

        # Random scaling ±10%
        scale = random.uniform(0.9, 1.1)
        pil_img, lm = _scale_img_lm(pil_img, lm, scale, target_size)

        # Random translation 5% × target_size
        tx = random.uniform(-0.05, 0.05) * target_size
        ty = random.uniform(-0.05, 0.05) * target_size
        pil_img, lm = _translate_img_lm(pil_img, lm, tx, ty, target_size)

        # Random horizontal flip 50%
        if random.random() < 0.5:
            pil_img, lm = _flip_img_lm(pil_img, lm, target_size)

        # Random gray 20%
        if random.random() < 0.2:
            pil_img = ImageOps.grayscale(pil_img).convert('RGB')

        # Random Gaussian blur 30%
        if random.random() < 0.3:
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))

        # Random occlusion 40%
        if random.random() < 0.4:
            pil_img = _random_occlusion(pil_img, target_size)

        # Random gamma 20%
        if random.random() < 0.2:
            pil_img = _gamma_adjust_pil(pil_img, random.uniform(0.7, 1.3))

    lm_norm = np.clip(lm / target_size, 0.0, 1.0)
    return _pil_to_tensor(pil_img), torch.from_numpy(lm_norm.flatten()).float()


def augment_headpose(pil_image, target_size, is_train=True):
    """
    Head pose pipeline: resize + loose random-resized crop (train only).
    Paper augmentations: random resized crop 80–100%, gray 10%, blur 10%, gamma 10%.
    """
    pil_image = pil_image.resize((target_size, target_size), Image.BILINEAR)

    if is_train:
        # Random resized crop 80–100% (loose crop proxy)
        crop_scale = random.uniform(0.8, 1.0)
        crop_size = int(target_size * crop_scale)
        x1 = random.randint(0, target_size - crop_size)
        y1 = random.randint(0, target_size - crop_size)
        pil_image = pil_image.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        pil_image = pil_image.resize((target_size, target_size), Image.BILINEAR)

        # Random gray 10%
        if random.random() < 0.1:
            pil_image = ImageOps.grayscale(pil_image).convert('RGB')

        # Random Gaussian blur 10%
        if random.random() < 0.1:
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # Random gamma 10%
        if random.random() < 0.1:
            pil_image = _gamma_adjust_pil(pil_image, random.uniform(0.7, 1.3))

    return _pil_to_tensor(pil_image)


def augment_attributes(pil_image, target_size, is_train=True):
    """
    Attribute prediction pipeline.
    Paper augmentations: rotation ±18°, scaling ±10%, translation 1%×size,
    flip 50%, gray 10%, blur 10%, gamma 20%.
    """
    pil_image = pil_image.resize((target_size, target_size), Image.BILINEAR)

    if is_train:
        # Random rotation ±18°
        angle = random.uniform(-18, 18)
        pil_image = pil_image.rotate(angle, expand=False)

        # Random scaling ±10%
        scale = random.uniform(0.9, 1.1)
        new_size = max(1, int(target_size * scale))
        scaled = pil_image.resize((new_size, new_size), Image.BILINEAR)
        result = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        px = (target_size - new_size) // 2
        py = (target_size - new_size) // 2
        result.paste(scaled, (px, py))
        pil_image = result

        # Random translation 1% × target_size
        tx = int(random.uniform(-0.01, 0.01) * target_size)
        ty = int(random.uniform(-0.01, 0.01) * target_size)
        result = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        result.paste(pil_image, (tx, ty))
        pil_image = result

        # Random horizontal flip 50%
        if random.random() < 0.5:
            pil_image = ImageOps.mirror(pil_image)

        # Random gray 10%
        if random.random() < 0.1:
            pil_image = ImageOps.grayscale(pil_image).convert('RGB')

        # Random Gaussian blur 10%
        if random.random() < 0.1:
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # Random gamma 20%
        if random.random() < 0.2:
            pil_image = _gamma_adjust_pil(pil_image, random.uniform(0.7, 1.3))

    return _pil_to_tensor(pil_image)


def augment_agr(pil_image, target_size, is_train=True):
    """
    Age / Gender / Race pipeline.
    Paper augmentations: rotation ±18°, scaling ±10%, translation 1%×size,
    flip 50%, gray 10%, blur 10%, gamma 10%.
    """
    pil_image = pil_image.resize((target_size, target_size), Image.BILINEAR)

    if is_train:
        # Random rotation ±18°
        angle = random.uniform(-18, 18)
        pil_image = pil_image.rotate(angle, expand=False)

        # Random scaling ±10%
        scale = random.uniform(0.9, 1.1)
        new_size = max(1, int(target_size * scale))
        scaled = pil_image.resize((new_size, new_size), Image.BILINEAR)
        result = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        px = (target_size - new_size) // 2
        py = (target_size - new_size) // 2
        result.paste(scaled, (px, py))
        pil_image = result

        # Random translation 1% × target_size
        tx = int(random.uniform(-0.01, 0.01) * target_size)
        ty = int(random.uniform(-0.01, 0.01) * target_size)
        result = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        result.paste(pil_image, (tx, ty))
        pil_image = result

        # Random horizontal flip 50%
        if random.random() < 0.5:
            pil_image = ImageOps.mirror(pil_image)

        # Random gray 10%
        if random.random() < 0.1:
            pil_image = ImageOps.grayscale(pil_image).convert('RGB')

        # Random Gaussian blur 10%
        if random.random() < 0.1:
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # Random gamma 10%
        if random.random() < 0.1:
            pil_image = _gamma_adjust_pil(pil_image, random.uniform(0.7, 1.3))

    return _pil_to_tensor(pil_image)


# COFW 29-point symmetric flip indices (0-based).
# _COFW_FLIP_IDX[i] is the landmark that landmark i maps to after a horizontal flip.
# Pairs: (0,3), (1,2), (4,5), (6,9), (7,8), (11,12), (13,14), (19,22), (20,23),
#         (21,24), (26,27).  Midline points map to themselves.
_COFW_FLIP_IDX = [
    3, 2, 1, 0,          # 0-3  right/left eye corners
    5, 4,                # 4-5  right/left eye pupils
    9, 8, 7, 6,          # 6-9  right/left eyebrow corners
    10,                  # 10   nose tip (midline)
    12, 11,              # 11-12 right/left nostrils
    14, 13,              # 13-14 right/left mouth corners
    15, 16, 17, 18,      # 15-18 lip centres (midline)
    22, 23, 24, 19, 20, 21,  # 19-24 right→left ear swap
    25,                  # 25   chin (midline)
    27, 26,              # 26-27 additional symmetric pair
    28,                  # 28   remaining midline point
]
assert len(_COFW_FLIP_IDX) == 29, "COFW flip index length must be 29"


def augment_visibility(pil_image, visibility, target_size, is_train=True):
    """
    Visibility prediction pipeline.
    Paper augmentations: flip 50%, gray 10%, blur 10%, gamma 10%.

    Args:
        pil_image   – PIL image
        visibility  – np.ndarray of shape (29,) with visibility labels
        target_size – output spatial size
        is_train    – whether to apply random augmentations

    Returns:
        img_tensor  – (3, target_size, target_size) normalised tensor
        vis_tensor  – (29,) float32 tensor, reordered if a flip was applied
    """
    pil_image = pil_image.resize((target_size, target_size), Image.BILINEAR)
    vis = np.array(visibility, dtype=np.float32)

    if is_train:
        # Random horizontal flip 50% — reorder visibility labels to match
        if random.random() < 0.5:
            pil_image = ImageOps.mirror(pil_image)
            vis = vis[_COFW_FLIP_IDX]

        # Random gray 10%
        if random.random() < 0.1:
            pil_image = ImageOps.grayscale(pil_image).convert('RGB')

        # Random Gaussian blur 10%
        if random.random() < 0.1:
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # Random gamma 10%
        if random.random() < 0.1:
            pil_image = _gamma_adjust_pil(pil_image, random.uniform(0.7, 1.3))

    return _pil_to_tensor(pil_image), torch.from_numpy(vis)


class UpsampledMultiTaskDataset(Dataset):
    """Multi-task dataset with upsampling to ensure balanced representation."""
    
    def __init__(self, datasets: Dict[str, List[Dataset]], target_size: Optional[int] = None):
        self.datasets = datasets
        self.task_names = list(datasets.keys())
        
        # Calculate dataset sizes
        self.dataset_sizes = {}
        self.max_size = 0
        
        for task_name, task_datasets in datasets.items():
            total_size = sum(len(ds) for ds in task_datasets)
            self.dataset_sizes[task_name] = total_size
            self.max_size = max(self.max_size, total_size)
        
        self.target_size = target_size if target_size else self.max_size
        self._create_upsampled_indices()
        
    def _create_upsampled_indices(self):
        """Create upsampled indices for smaller datasets."""
        self.upsampled_indices = {}
        for task_name in self.task_names:
            task_size = self.dataset_sizes[task_name]
            num_repeats = self.target_size // task_size
            remainder = self.target_size % task_size
            
            base_indices = list(range(task_size))
            upsampled = base_indices * num_repeats
            if remainder > 0:
                upsampled.extend(random.sample(base_indices, remainder))
            random.shuffle(upsampled)
            
            self.upsampled_indices[task_name] = upsampled
    
    def __len__(self):
        return self.target_size * len(self.task_names)
    
    def __getitem__(self, idx):
        task_idx = idx % len(self.task_names)
        task_name = self.task_names[task_idx]
        sample_idx = (idx // len(self.task_names)) % self.target_size
        upsampled_idx = self.upsampled_indices[task_name][sample_idx]
        
        task_datasets = self.datasets[task_name]
        cumulative = 0
        for dataset in task_datasets:
            if upsampled_idx < cumulative + len(dataset):
                local_idx = upsampled_idx - cumulative
                return dataset[local_idx]
            cumulative += len(dataset)
        
        return task_datasets[0][0]


class BalancedMultiTaskBatchSampler:
    """
    Custom batch sampler that ensures each batch has balanced task representation.
    
    For multi-GPU training, each GPU gets batches with the same task distribution.
    Randomly assigns which tasks get extra samples when batch_size is not evenly divisible by num_tasks.
    This prevents bias toward specific tasks.
    
    Strategy (matching facexformer-my):
    1. Upsample each task to match the size of the largest task
    2. Shuffle the upsampled indices for each task
    3. For each batch, randomly decide which tasks get the extra sample
    4. Sample equal number of samples from each task
    5. Distribute batches across multiple GPUs/processes
    """
    
    def __init__(self, datasets, batch_size, num_replicas=1, rank=0, shuffle=True, seed=0, drop_last=True):
        self.datasets = datasets
        self.task_names = list(datasets.keys())
        self.num_tasks = len(self.task_names)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        # Calculate samples per task in each batch
        self.base_samples_per_task = batch_size // self.num_tasks
        self.remainder = batch_size % self.num_tasks
        
        # Build task structure and upsample
        self._build_task_structure()
        self._upsample_tasks()
        self._calculate_num_batches()
    
    def _build_task_structure(self):
        """Build task information: sizes."""
        self.task_sizes = {}
        
        for task_name in self.task_names:
            task_datasets = self.datasets[task_name]
            task_size = sum(len(ds) for ds in task_datasets)
            self.task_sizes[task_name] = task_size
        
        self.max_task_size = max(self.task_sizes.values())
    
    def _upsample_tasks(self):
        """Upsample smaller tasks to match the largest task size."""
        self.upsampled_task_indices = {}
        
        for task_name in self.task_names:
            task_size = self.task_sizes[task_name]
            
            # Create base indices for this task
            base_indices = list(range(task_size))
            
            # Upsample to match max_task_size
            num_repeats = self.max_task_size // task_size
            remainder = self.max_task_size % task_size
            
            upsampled = base_indices * num_repeats
            if remainder > 0:
                # Add random samples for remainder (using fixed seed)
                rng = random.Random(self.seed)
                upsampled.extend(rng.sample(base_indices * ((remainder // task_size) + 1), remainder))
            
            self.upsampled_task_indices[task_name] = upsampled
    
    def _calculate_num_batches(self):
        """Calculate number of batches based on upsampled task size."""
        max_samples_per_task = self.base_samples_per_task + (1 if self.remainder > 0 else 0)
        max_batches_global = self.max_task_size // max_samples_per_task
        
        if self.drop_last:
            self.num_batches_per_replica = max_batches_global // self.num_replicas
        else:
            self.num_batches_per_replica = math.ceil(max_batches_global / self.num_replicas)
    
    def __iter__(self):
        """Generate task-balanced batches by randomly sampling from upsampled task pools."""
        # Shuffle upsampled indices for each task
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
        
        shuffled_task_indices = {}
        for task_name in self.task_names:
            upsampled = self.upsampled_task_indices[task_name].copy()
            
            if self.shuffle:
                perm = torch.randperm(len(upsampled), generator=g).tolist()
                shuffled_task_indices[task_name] = [upsampled[i] for i in perm]
            else:
                shuffled_task_indices[task_name] = upsampled
        
        # Generate batches for this replica
        for local_batch_idx in range(self.num_batches_per_replica):
            global_batch_idx = self.rank + local_batch_idx * self.num_replicas
            
            # CRITICAL: Randomly assign which tasks get the extra sample for THIS batch
            # This prevents bias toward specific tasks
            if self.remainder > 0:
                batch_rng = random.Random(self.seed + self.epoch * 1000000 + global_batch_idx)
                task_indices_shuffled = list(range(self.num_tasks))
                batch_rng.shuffle(task_indices_shuffled)
                tasks_with_extra = set(task_indices_shuffled[:self.remainder])
            else:
                tasks_with_extra = set()
            
            batch = []
            
            # Sample from each task
            for task_idx, task_name in enumerate(self.task_names):
                task_indices = shuffled_task_indices[task_name]
                
                # Randomly selected tasks get base+1, rest get base
                samples_for_this_task = (self.base_samples_per_task + 1 
                                        if task_idx in tasks_with_extra 
                                        else self.base_samples_per_task)
                
                # Get samples from the upsampled shuffled pool
                start_pos = global_batch_idx * samples_for_this_task
                end_pos = start_pos + samples_for_this_task
                
                batch.extend(task_indices[start_pos:end_pos])
            
            # Shuffle within batch for additional randomness
            if self.shuffle:
                random.Random(self.seed + self.epoch + global_batch_idx).shuffle(batch)
            
            yield batch
    
    def __len__(self):
        return self.num_batches_per_replica
    
    def set_epoch(self, epoch):
        self.epoch = epoch


def multi_task_collate_fn(batch):
    """Custom collate function to handle mixed-task batches."""
    images = torch.stack([item[0] for item in batch])
    
    # Collect all possible target keys
    all_keys = set()
    for item in batch:
        all_keys.update(item[1].keys())
    
    targets = {}
    for key in all_keys:
        # Stack tensors if present, else use dummy values
        values = []
        for item in batch:
            if key in item[1]:
                val = item[1][key]
                # Ensure it's a tensor
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val)
                values.append(val)
            else:
                # Create dummy tensor of appropriate shape
                if key == 'segmentation':
                    img_size = config.IMG_SIZE if config else 224
                    values.append(torch.zeros(img_size, img_size, dtype=torch.long))
                elif key == 'landmark':
                    landmark_dim = config.LANDMARK_DIM if config else 136
                    values.append(torch.zeros(landmark_dim))
                elif key == 'headpose':
                    headpose_dim = config.HEADPOSE_DIM if config else 3
                    values.append(torch.zeros(headpose_dim))
                elif key == 'attribute':
                    attribute_dim = config.ATTRIBUTE_DIM if config else 40
                    values.append(torch.zeros(attribute_dim))
                elif key == 'age':
                    values.append(torch.tensor(0, dtype=torch.long))
                elif key == 'gender':
                    values.append(torch.tensor(0, dtype=torch.long))
                elif key == 'race':
                    values.append(torch.tensor(0, dtype=torch.long))
                elif key == 'visibility':
                    visibility_dim = config.VISIBILITY_DIM if config else 29
                    values.append(torch.zeros(visibility_dim))
                elif key == 'task_id':
                    values.append(torch.tensor(-1, dtype=torch.long))
                else:
                    values.append(torch.tensor(0))
        
        if len(values) > 0:
            targets[key] = torch.stack(values)
    
    return images, targets


def create_multi_task_dataloader(
    datasets: Dict[str, List[Dataset]],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    use_upsampling: bool = True,
    rank: int = 0,
    world_size: int = 1,
    use_balanced_batches: bool = True
) -> DataLoader:
    """
    Create DataLoader for multi-task co-training with upsampling.
    
    Args:
        datasets: Dictionary mapping task names to lists of datasets
        batch_size: Batch size per GPU
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        use_upsampling: Whether to use upsampling to balance tasks
        rank: Current process rank for distributed training
        world_size: Total number of processes for distributed training
        use_balanced_batches: If True, ensures each batch has balanced task representation
    
    Returns:
        DataLoader with proper distributed sampling if world_size > 1
        If use_balanced_batches=True, each batch will have exactly batch_size/num_tasks
        samples from each task (or as close as possible if not evenly divisible)
    """
    from torch.utils.data.distributed import DistributedSampler
    
    if use_upsampling:
        multi_task_dataset = UpsampledMultiTaskDataset(datasets)
        num_tasks = len(datasets)
        
        # Use custom balanced batch sampler for guaranteed task balance within batches
        if use_balanced_batches:
            batch_sampler = BalancedMultiTaskBatchSampler(
                datasets=datasets,
                batch_size=batch_size,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                seed=0,
                drop_last=True
            )
            dataloader = DataLoader(
                multi_task_dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=multi_task_collate_fn
            )
        # Use DistributedSampler for multi-GPU training (approximate task balance)
        elif world_size > 1:
            sampler = DistributedSampler(
                multi_task_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=True
            )
            dataloader = DataLoader(
                multi_task_dataset,
                batch_size=batch_size,
                sampler=sampler,  # Use sampler instead of shuffle
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=multi_task_collate_fn
            )
        else:
            # Single GPU training
            dataloader = DataLoader(
                multi_task_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=multi_task_collate_fn
            )
    else:
        all_datasets = []
        for task_datasets in datasets.values():
            all_datasets.extend(task_datasets)
        combined_dataset = ConcatDataset(all_datasets)
        
        # Use DistributedSampler for multi-GPU training
        if world_size > 1:
            sampler = DistributedSampler(
                combined_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=True
            )
            dataloader = DataLoader(
                combined_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=multi_task_collate_fn
            )
        else:
            dataloader = DataLoader(
                combined_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=multi_task_collate_fn
            )
    
    return dataloader


# Dataset implementations (simplified versions)
class TaskDataset(Dataset):
    """Base dataset class."""
    def __init__(self, dataset_name: str, task_name: str, split: str = 'train', dataset_root: str = None):
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.split = split
        if dataset_root is None:
            dataset_root = config.DATASET_ROOT
        self.dataset_root = Path(dataset_root)
        self.target_size = config.IMG_SIZE if config else 224
        self.data = []
    
    def __len__(self):
        return len(self.data)


class CelebAMaskHQDataset(TaskDataset):
    """CelebAMask-HQ for face parsing. Maps to task 0 (segmentation)."""
    def __init__(self, split='train', dataset_root=None):
        super().__init__('CelebAMaskHQ', 'segmentation', split, dataset_root)
        self.data_root = self.dataset_root / 'CelebAMask-HQ'
        img_dir = self.data_root / 'CelebA-HQ-img'
        all_images = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
        self.data = all_images[:28000] if split == 'train' else all_images[28000:30000]
        if len(self.data) == 0:
            raise FileNotFoundError(f"CelebAMaskHQ not found at {img_dir}")
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        img_idx = int(img_path.stem)
        
        # Load and combine mask from individual files
        mask_folder = self.data_root / 'CelebAMask-HQ-mask-anno' / str(img_idx // 2000)
        mask = np.zeros((512, 512), dtype=np.int64)
        mask_names = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear',
                     'ear_r', 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        for i, label_name in enumerate(mask_names, start=1):
            mask_file = mask_folder / f'{img_idx:05d}_{label_name}.png'
            if mask_file.exists():
                label_mask = np.array(Image.open(mask_file).convert('L'))
                # Limit to SEGMENTATION_CLASSES (0 to SEGMENTATION_CLASSES-1)
                max_class = (config.SEGMENTATION_CLASSES - 1) if config else 10
                mask[label_mask == 255] = min(i, max_class)
        
        # Resize image and mask to IMG_SIZE
        image = simple_augmentation(image, self.target_size)
        # Segmentation output is upsampled to IMG_SIZE in model forward pass
        mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize((self.target_size, self.target_size), Image.NEAREST))
        mask = torch.from_numpy(mask).long()
        # Clamp to valid range [0, SEGMENTATION_CLASSES-1] to prevent CUDA device-side asserts
        max_class_idx = (config.SEGMENTATION_CLASSES - 1) if config else 10
        mask = torch.clamp(mask, 0, max_class_idx)
        
        return (image, {'segmentation': mask, 'task_id': torch.tensor(0)})


class W300Dataset(TaskDataset):
    """300W for landmarks. Maps to task 1."""
    def __init__(self, split='train', dataset_root=None):
        super().__init__('300W', 'landmark', split, dataset_root)
        self.data_root = self.dataset_root / '300w'
        base_path = self.data_root / '300W' if (self.data_root / '300W').exists() else self.data_root
        self.data = []
        for subdir in ['01_Indoor', '02_Outdoor']:
            subdir_path = base_path / subdir
            if subdir_path.exists():
                self.data.extend(list(subdir_path.glob('*.jpg')) + list(subdir_path.glob('*.png')))
        total = len(self.data)
        self.data = self.data[:int(total * 0.82)] if split == 'train' else self.data[int(total * 0.82):]
        if len(self.data) == 0:
            raise FileNotFoundError(f"300W not found at {self.data_root}")
    
    def _load_pts(self, pts_path):
        with open(pts_path, 'r') as f:
            lines = f.readlines()
        landmarks = []
        for line in lines[3:-1]:
            coords = line.strip().split()
            if len(coords) == 2:
                landmarks.append([float(coords[0]), float(coords[1])])
        return np.array(landmarks, dtype=np.float32)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        pts_path = img_path.with_suffix('.pts')
        landmarks = self._load_pts(pts_path)  # (68, 2) pixel coords

        image, landmarks_flat = augment_landmark_detection(
            image, landmarks, self.target_size, is_train=(self.split == 'train')
        )
        return (image, {'landmark': landmarks_flat, 'task_id': torch.tensor(1)})


class W300LPDataset(TaskDataset):
    """300W-LP for head pose. Maps to task 2."""
    def __init__(self, split='train', dataset_root=None):
        super().__init__('300W-LP', 'headpose', split, dataset_root)
        self.data_root = self.dataset_root / '300W_LP'
        self.data = []
        for subdir in self.data_root.iterdir():
            if subdir.is_dir() and subdir.name not in ['Code', 'landmarks', 'code']:
                self.data.extend(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
        if len(self.data) == 0 and split == 'train':
            raise FileNotFoundError(f"300W-LP not found at {self.data_root}")
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        mat_path = img_path.with_suffix('.mat')
        import scipy.io
        pose_data = scipy.io.loadmat(str(mat_path))
        rotation = pose_data['Pose_Para'][0][:3]  # Euler angles

        image = augment_headpose(image, self.target_size, is_train=(self.split == 'train'))
        rotation = torch.from_numpy(rotation).float()
        return (image, {'headpose': rotation, 'task_id': torch.tensor(2)})


class CelebADataset(TaskDataset):
    """CelebA for attributes. Maps to task 3."""
    def __init__(self, split='train', dataset_root=None, rank=0, world_size=1):
        super().__init__('CelebA', 'attribute', split, dataset_root)
        self.data_root = self.dataset_root / 'CelebA'
        self.rank = rank
        self.world_size = world_size
        self.is_master = rank == 0
        cache_file = self.data_root / f'celeba_{split}_cache.pkl'
        
        if cache_file.exists():
            # Check cache metadata for compatibility
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                    meta = cached.get('meta', {})
                    
                    # Check if cache metadata matches current config
                    cache_valid = True
                    if meta.get('img_size') != config.IMG_SIZE:
                        cache_valid = False
                        if self.is_master:
                            print(f"Cache {cache_file} img_size={meta.get('img_size')} != config.IMG_SIZE={config.IMG_SIZE}; rebuilding")
                    if meta.get('dataset_root') != str(self.dataset_root.resolve()):
                        cache_valid = False
                        if self.is_master:
                            print(f"Cache {cache_file} dataset_root={meta.get('dataset_root')} != {self.dataset_root.resolve()}; rebuilding")
                    
                    if cache_valid:
                        self.data = cached['data']
                        self.attributes = cached['attributes']
                    else:
                        # Force rebuild by removing cache file
                        if self.is_master:
                            cache_file.unlink(missing_ok=True)
                        raise ValueError("Cache metadata mismatch")
            except (KeyError, ValueError):
                # Cache is invalid or missing metadata, rebuild it
                if self.is_master:
                    cache_file.unlink(missing_ok=True)
                # Fall through to rebuild logic
                pass
        else:
            # Cache doesn't exist, will be built below
            pass
        
        # Re-check if cache exists after potential removal
        if not cache_file.exists():
            # Only master process builds cache in distributed training
            if self.is_master:
                attr_file = self.data_root / 'list_attr_celeba.csv'
                split_file = self.data_root / 'list_eval_partition.csv'
                self.data = []
                self.attributes = []
                
                if attr_file.exists() and split_file.exists():
                    attr_df = pd.read_csv(attr_file)
                    split_df = pd.read_csv(split_file)
                    merged = attr_df.merge(split_df, on='image_id')
                    filtered = merged[merged['partition'] == (0 if split == 'train' else 2)]
                    attr_cols = [col for col in attr_df.columns if col != 'image_id']
                    
                    img_dir1 = self.data_root / 'img_align_celeba'
                    img_dir2 = img_dir1 / 'img_align_celeba'
                    img_dir = img_dir2 if img_dir2.exists() else img_dir1
                    
                    for i, (img_name, attr_values) in enumerate(zip(filtered['image_id'].values, filtered[attr_cols].values)):
                        img_path = img_dir / img_name
                        if img_path.exists():
                            self.data.append(img_path)
                            attrs = ((attr_values + 1) // 2).tolist()
                            self.attributes.append(attrs)
                    
                    # Save cache with metadata
                    cache_data = {
                        'data': self.data,
                        'attributes': self.attributes,
                        'meta': {
                            'img_size': config.IMG_SIZE,
                            'dataset_root': str(self.dataset_root.resolve()),
                            'cache_version': '1.0',
                            'created_by': f'rank_{self.rank}'
                        }
                    }
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cache_data, f)
            
            # In distributed training, non-master processes wait for cache
            if self.world_size > 1 and not self.is_master:
                cache_poll_timeout = 2 * 60 * 60.0  # 2 hours
                cache_poll_interval = 5.0  # Check every 5 seconds
                start_time = time.time()
                
                while not cache_file.exists():
                    elapsed = time.time() - start_time
                    if elapsed >= cache_poll_timeout:
                        raise TimeoutError(f"Timeout waiting for cache file {cache_file} after {cache_poll_timeout}s")
                    time.sleep(cache_poll_interval)
                
                # Load from cache and validate metadata
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                    meta = cached.get('meta', {})
                    if meta.get('img_size') != config.IMG_SIZE:
                        raise ValueError(f"Cache metadata mismatch: img_size {meta.get('img_size')} != {config.IMG_SIZE}")
                    if meta.get('dataset_root') != str(self.dataset_root.resolve()):
                        raise ValueError(f"Cache metadata mismatch: dataset_root {meta.get('dataset_root')} != {self.dataset_root.resolve()}")
                    self.data = cached['data']
                    self.attributes = cached['attributes']
        
        if len(self.data) == 0:
            raise FileNotFoundError(f"CelebA not found at {self.data_root}")
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        attributes = torch.tensor(self.attributes[idx], dtype=torch.float32)

        image = augment_attributes(image, self.target_size, is_train=(self.split == 'train'))
        return (image, {'attribute': attributes, 'task_id': torch.tensor(3)})


class MultiLabelDatasetWrapper:
    """Wrapper for UTKFace/FairFace to extract single label (age/gender/race)."""
    def __init__(self, base_dataset, target_task):
        self.base_dataset = base_dataset
        self.target_task = target_task
        # Map to task IDs: age=4, gender=5, race=6
        self.task_id_map = {'age': 4, 'gender': 5, 'race': 6}
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, targets = self.base_dataset[idx]
        return (image, {self.target_task: targets[self.target_task], 
                       'task_id': torch.tensor(self.task_id_map[self.target_task])})
    
    def get_name(self):
        """Return the name of the underlying dataset for display purposes."""
        base_name = self.base_dataset.__class__.__name__
        return f"{base_name} ({self.target_task})"


class UTKFaceDataset(TaskDataset):
    """UTKFace for age/gender/race."""
    def __init__(self, split='train', dataset_root=None):
        super().__init__('UTKFace', 'age_gender_race', split, dataset_root)
        self.data_root = self.dataset_root / 'UTKFace'
        utkface_dir = self.data_root / 'UTKFace' if (self.data_root / 'UTKFace').exists() else self.data_root
        all_images = sorted(list(utkface_dir.glob('*.jpg')))
        
        valid_images = []
        for img_path in all_images:
            parts = img_path.stem.split('_')
            if len(parts) >= 4:
                try:
                    int(parts[0]); int(parts[1]); int(parts[2])
                    valid_images.append(img_path)
                except ValueError:
                    continue
        
        total = len(valid_images)
        split_idx = int(total * 0.85)
        self.data = valid_images[:split_idx] if split == 'train' else valid_images[split_idx:]
        if len(self.data) == 0:
            raise FileNotFoundError(f"UTKFace not found at {self.data_root}")
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        parts = img_path.stem.split('_')
        age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])

        # Convert age to bucket index (0-7)
        age_bucket = age_to_bucket(age)

        image = augment_agr(image, self.target_size, is_train=(self.split == 'train'))
        return (image, {'age': age_bucket, 'gender': gender, 'race': race})


class FairFaceDataset(TaskDataset):
    """FairFace for age/gender/race."""
    def __init__(self, split='train', dataset_root=None):
        super().__init__('FairFace', 'age_gender_race', split, dataset_root)
        self.data_root = self.dataset_root / 'FairFace'
        csv_split = 'train' if split == 'train' else 'val'
        csv_file = self.data_root / 'extracted' / f'{csv_split}_labels.csv'
        
        age_map = {
            '0-2': 0,      # 0-9 bucket
            '3-9': 0,      # 0-9 bucket
            '10-19': 1,    # 10-19 bucket
            '20-29': 2,    # 20-29 bucket
            '30-39': 3,    # 30-39 bucket
            '40-49': 4,    # 40-49 bucket
            '50-59': 5,    # 50-59 bucket
            '60-69': 6,    # 60-69 bucket
            'more than 70': 7  # 70+ bucket
        }
        race_map = {'White': 0, 'Black': 1, 'Asian': 2, 'Indian': 3, 'Others': 4}
        gender_map = {'Male': 0, 'Female': 1}
        
        self.data = []
        self.labels = []
        
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                img_filename = Path(row['file']).name if 'file' in df.columns else row.iloc[0]
                img_path = self.data_root / 'extracted' / csv_split / img_filename
                if img_path.exists():
                    self.data.append(img_path)
                    self.labels.append({
                        'age': age_map.get(row['age'], 3),  # Default to bucket 3 (30-39) if unknown
                        'gender': gender_map.get(row['gender'], 0),
                        'race': race_map.get(row['race'], 4)
                    })
        
        if len(self.data) == 0:
            raise FileNotFoundError(f"FairFace not found at {self.data_root}")
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        labels = self.labels[idx]

        image = augment_agr(image, self.target_size, is_train=(self.split == 'train'))
        return (image, {'age': labels['age'], 'gender': labels['gender'], 'race': labels['race']})


class COFWDataset(TaskDataset):
    """COFW for visibility. Maps to task 5."""
    def __init__(self, split='train', dataset_root=None):
        super().__init__('COFW', 'visibility', split, dataset_root)
        self.data_root = self.dataset_root / 'COFW'
        
        import h5py
        mat_paths = [
            self.data_root / ('COFW_train.mat' if split == 'train' else 'COFW_test.mat'),
            self.data_root / 'common' / 'xpburgos' / 'behavior' / ('COFW_train.mat' if split == 'train' else 'COFW_test.mat'),
            self.data_root / 'common' / 'xpburgos' / 'behavior' / 'code' / 'pose' / ('COFW_train.mat' if split == 'train' else 'COFW_test.mat'),
        ]
        
        mat_file = None
        for path in mat_paths:
            if path.exists():
                mat_file = path
                break
        
        self.images = []
        self.visibility_labels = []
        
        if mat_file:
            with h5py.File(mat_file, 'r') as f:
                img_key = 'IsTr' if split == 'train' else 'IsT'
                phi_key = 'phisTr' if split == 'train' else 'phisT'
                img_refs = f[img_key]
                phi_data = f[phi_key]
                
                for i in range(img_refs.shape[1]):
                    img_ref = img_refs[0, i]
                    img_data = f[img_ref][:]
                    self.images.append(img_data)
                    
                    landmark_phis = phi_data[:, i]
                    visibility_values = landmark_phis[2::3]  # 29 visibility values
                    self.visibility_labels.append(visibility_values)
        
        if len(self.images) == 0:
            raise FileNotFoundError(f"COFW not found at {self.data_root}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_data = self.images[idx]
        if img_data.ndim == 2:
            img_data = np.stack([img_data, img_data, img_data], axis=-1)
        
        image = Image.fromarray(img_data.astype(np.uint8))
        visibility = self.visibility_labels[idx]  # np.ndarray (29,)

        image, visibility_tensor = augment_visibility(
            image, visibility, self.target_size, is_train=(self.split == 'train')
        )
        return (image, {'visibility': visibility_tensor, 'task_id': torch.tensor(7)})


class W300VWDataset(TaskDataset):
    """300VW for landmark detection (test only). Maps to task 1."""
    def __init__(self, split='test', dataset_root=None):
        super().__init__('300VW', 'landmark', split, dataset_root)
        self.data_root = self.dataset_root / '300VW'
        
        # Category 3 test sequences (challenging scenarios)
        test_sequences = ['410', '516', '517', '526', '528', '529', '530', 
                         '531', '533', '557', '558', '559', '562']
        
        self.data = []
        if self.data_root.exists():
            for seq_name in test_sequences:
                video_dir = self.data_root / seq_name
                if not video_dir.exists():
                    continue
                annot_dir = video_dir / 'annot'
                if annot_dir.exists():
                    pts_files = sorted(list(annot_dir.glob('*.pts')))
                    for pts_file in pts_files:
                        frame_name = pts_file.stem + '.jpg'
                        frame_path = video_dir / frame_name
                        if frame_path.exists():
                            self.data.append((frame_path, pts_file))
        
        if len(self.data) == 0 and split == 'test':
            raise FileNotFoundError(f"300VW test frames not found at {self.data_root}")
    
    def _load_pts(self, pts_path):
        with open(pts_path, 'r') as f:
            lines = f.readlines()
        landmarks = []
        for line in lines[3:-1]:
            coords = line.strip().split()
            if len(coords) == 2:
                landmarks.append([float(coords[0]), float(coords[1])])
        return np.array(landmarks, dtype=np.float32)
    
    def __getitem__(self, idx):
        img_path, pts_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        landmarks = self._load_pts(pts_path)  # (68, 2) pixel coords

        # Test set: alignment only, no augmentation
        image, landmarks_flat = augment_landmark_detection(
            image, landmarks, self.target_size, is_train=False
        )
        return (image, {'landmark': landmarks_flat, 'task_id': torch.tensor(1)})


class BIWIDataset(TaskDataset):
    """BIWI for head pose estimation (test only). Maps to task 2."""
    def __init__(self, split='test', dataset_root=None):
        super().__init__('BIWI', 'headpose', split, dataset_root)
        self.data_root = self.dataset_root / 'BIWI'
        
        self.data = []
        faces_dir = self.data_root / 'faces_0'
        if faces_dir.exists():
            for person_dir in sorted(faces_dir.iterdir()):
                if person_dir.is_dir() and person_dir.name.isdigit():
                    rgb_files = sorted(list(person_dir.glob('frame_*_rgb.png')))
                    for rgb_file in rgb_files:
                        parts = rgb_file.stem.split('_')
                        if len(parts) >= 2:
                            frame_num = parts[1]
                            pose_file = person_dir / f'frame_{frame_num}_pose.txt'
                            if pose_file.exists():
                                self.data.append((rgb_file, pose_file))
        
        if len(self.data) == 0 and split == 'test':
            raise FileNotFoundError(f"BIWI dataset not found at {self.data_root}")
    
    def __getitem__(self, idx):
        img_path, pose_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load rotation matrix from pose file
        with open(pose_path, 'r') as f:
            lines = f.readlines()
        rotation_matrix = []
        for line in lines[:3]:
            rotation_matrix.extend([float(x) for x in line.strip().split()])
        
        # Convert 9D rotation matrix to 3D Euler angles (approximate)
        # For simplicity, extract yaw, pitch, roll from rotation matrix
        import math
        R = np.array(rotation_matrix).reshape(3, 3)
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            yaw = math.atan2(R[1, 0], R[0, 0])
            pitch = math.atan2(-R[2, 0], sy)
            roll = math.atan2(R[2, 1], R[2, 2])
        else:
            yaw = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            roll = 0
        
        # Match 300W-LP training convention: [pitch, yaw, roll] in radians
        rotation = torch.tensor([pitch, yaw, roll], dtype=torch.float32)

        image = augment_headpose(image, self.target_size, is_train=False)
        return (image, {'headpose': rotation, 'task_id': torch.tensor(2)})


class LFWADataset(TaskDataset):
    """LFWA for attributes (test only). Maps to task 3."""
    def __init__(self, split='test', dataset_root=None):
        super().__init__('LFWA', 'attribute', split, dataset_root)
        self.data_root = self.dataset_root / 'LFWA'
        
        anno_file = self.data_root / 'lfw_attributes.txt'
        self.data = []
        self.attributes = []
        
        if anno_file.exists():
            with open(anno_file, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    if line.strip().startswith('#'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) >= 75:
                        person_name = parts[0]
                        img_num = parts[1]
                        person_folder = person_name.replace(' ', '_')
                        try:
                            img_name = f"{person_folder}_{int(img_num):04d}.jpg"
                        except ValueError:
                            continue
                        
                        img_paths = [
                            self.data_root / 'lfw' / person_folder / img_name,
                            self.data_root / 'lfw' / 'lfw' / person_folder / img_name,
                            self.data_root / 'lfw-deepfunneled' / person_folder / img_name
                        ]
                        
                        for img_path in img_paths:
                            if img_path.exists():
                                self.data.append(img_path)
                                try:
                                    attrs = [int(float(parts[i]) > 0) for i in range(2, 42)]
                                    self.attributes.append(attrs)
                                except (ValueError, IndexError):
                                    self.data.pop()
                                    continue
                                break
        
        if len(self.data) == 0 and split == 'test':
            raise FileNotFoundError(f"LFWA dataset not found at {self.data_root}")

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        attributes = torch.tensor(self.attributes[idx], dtype=torch.float32)

        # Test set: no augmentation
        image = augment_attributes(image, self.target_size, is_train=False)
        return (image, {'attribute': attributes, 'task_id': torch.tensor(3)})

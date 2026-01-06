"""
Configuration file for FaceXFormer-main model.
Adapted for facexformer-main architecture (simpler, SAM-based).
Expression recognition task is removed.
"""

import torch
import os


def get_gpu_memory_gb():
    """Get available GPU memory in GB."""
    if torch.cuda.is_available():
        # Get total GPU memory
        gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = gpu_memory_bytes / (1024 ** 3)
        return gpu_memory_gb
    return 0


def auto_configure_batch_size():
    """
    Automatically configure batch size and related hyperparameters based on GPU memory.
    
    Note: Paper uses 384 total batch size across 8 GPUs (48 per GPU).
    Single GPU settings are scaled accordingly.
    
    Returns:
        dict with batch_size, learning_rate, gradient_checkpointing
    """
    gpu_memory = get_gpu_memory_gb()
    
    # High-end data center GPUs (GH200: 96-144GB, A100 80GB, H100: 80GB)
    if gpu_memory >= 70:
        return {
            'batch_size': 96,  # Safe for 96GB, allows headroom
            'learning_rate': 2.5e-5,  # 1e-4 * (96/384)
            'gradient_checkpointing': False,
            'reason': f'{gpu_memory:.1f}GB GPU - High-end single GPU'
        }
    # A100 40GB
    elif gpu_memory >= 40:
        return {
            'batch_size': 64,
            'learning_rate': 1.67e-5,  # 1e-4 * (64/384)
            'gradient_checkpointing': False,
            'reason': f'{gpu_memory:.1f}GB GPU - Single A100 40GB'
        }
    # Mid-range GPUs (RTX 3090/4090: 24GB, A10: 24GB)
    elif gpu_memory >= 20:
        return {
            'batch_size': 32,
            'learning_rate': 8.33e-6,  # 1e-4 * (32/384)
            'gradient_checkpointing': False,
            'reason': f'{gpu_memory:.1f}GB GPU - Consumer high-end'
        }
    # Lower mid-range (RTX 3080: 10-12GB, V100: 16GB)
    elif gpu_memory >= 10:
        return {
            'batch_size': 16,
            'learning_rate': 4.17e-6,  # 1e-4 * (16/384)
            'gradient_checkpointing': True,
            'reason': f'{gpu_memory:.1f}GB GPU - With checkpointing'
        }
    # Low-end consumer GPUs (RTX 4060: 8GB, RTX 3060: 8-12GB)
    elif gpu_memory >= 6:
        return {
            'batch_size': 4,
            'learning_rate': 1.04e-6,  # 1e-4 * (4/384)
            'gradient_checkpointing': True,
            'reason': f'{gpu_memory:.1f}GB GPU - Minimal batch with checkpointing'
        }
    # CPU or very small GPU
    else:
        return {
            'batch_size': 2,
            'learning_rate': 5.2e-7,  # 1e-4 * (2/384)
            'gradient_checkpointing': True,
            'reason': 'CPU or <6GB GPU - Minimal settings'
        }


class Config:
    """Configuration for FaceXFormer-main multi-task facial analysis model."""
    
    # Input configuration
    IMG_SIZE = 224
    IN_CHANNELS = 3
    
    # Backbone configuration (Swin-B)
    SWIN_PRETRAINED = True
    
    # Task tokens (facexformer-main uses 18 total)
    # 1 landmark + 1 pose + 1 attribute + 1 visibility + 1 age + 1 gender + 1 race + 11 mask = 18
    TASK_TOKENS = {
        'landmark': 1,      # Single token → MLP → 136 coords (68 * 2)
        'headpose': 1,      # Single token → MLP → 3 (Euler angles)
        'attribute': 1,     # Single token → MLP → 40 attributes
        'visibility': 1,    # Single token → MLP → 29 visibility
        'age': 1,           # Single token → MLP → 8 age groups
        'gender': 1,        # Single token → MLP → 2 classes
        'race': 1,          # Single token → MLP → 5 classes
        'mask': 11          # 11 tokens for segmentation
    }
    
    # Total task tokens
    TOTAL_TASK_TOKENS = sum(TASK_TOKENS.values())  # = 18
    
    # Decoder configuration
    DECODER_DEPTH = 2
    DECODER_EMBEDDING_DIM = 256
    DECODER_MLP_DIM = 2048
    DECODER_NUM_HEADS = 8
    
    # Task output dimensions
    LANDMARK_DIM = 136  # 68 * 2 (x, y coords)
    HEADPOSE_DIM = 3    # Euler angles (yaw, pitch, roll)
    ATTRIBUTE_DIM = 40  # Binary attributes
    VISIBILITY_DIM = 29 # Landmark visibility
    AGE_DIM = 8         # Age groups
    GENDER_DIM = 2      # Male/Female
    RACE_DIM = 5        # 5 race categories
    SEGMENTATION_CLASSES = 11  # Face segmentation classes
    
    # Loss weights (lambda values - adjusted for better performance)
    # 
    # Reasoning based on evaluation results:
    # - Age: Keep at 1.0 (high MAE needs MORE training focus, not less)
    # - Segmentation: Increase from 1.0→2.5 (underperforming: 0.8647 vs 0.9201 target)
    # - Visibility: Keep at 1.0 (loss magnitude is OK when normalized)
    # - Gender/Race: Decrease from 1.0→0.7 (showing perfect scores, possible overfit)
    # - Landmarks: Keep at 1.0 (performing well, 0.0423 vs 0.0467 target)
    # - Headpose: Keep at 1.0 (performing very well)
    # - Attributes: Increase from 1.0→1.2 (slightly underperforming: 0.9127 vs 0.9183)
    LOSS_WEIGHTS = {
        'seg': 2.5,        # λ_seg (segmentation) - increased to improve F1-score
        'ind': 1.0,        # λ_ind (landmarks) - performing well, keep as is
        'hpe': 1.0,        # λ_hpe (head pose) - performing well, keep as is
        'attr': 1.2,       # λ_attr (attributes) - slightly increase for better accuracy
        'a': 1.0,          # λ_a (age) - keep at 1.0, high MAE needs training focus
        'g/r': 0.7,        # λ_g/r (gender/race) - reduced to prevent overfitting
        'vis': 1.0         # λ_vis (visibility) - keep at 1.0
    }
    
    # Training configuration - Auto-configured based on GPU memory
    # Paper settings: 8 GPUs × batch=48 per GPU = 384 total, LR=1e-4
    # 
    # IMPORTANT: With DistributedDataParallel, BATCH_SIZE is the per-GPU batch size.
    # Total effective batch size = BATCH_SIZE × number of GPUs
    
    # ============================================================================
    # MANUAL OVERRIDE: To manually set batch size, uncomment these lines:
    # ============================================================================
    BATCH_SIZE = 96  # Per-GPU batch size
    LEARNING_RATE = 1e-4  # learning rate (AdamW optimizer, 1e-4*sqrt(total_batch/384))
    GRADIENT_CHECKPOINTING = False
    _manual_override = True
    
    # ============================================================================
    # AUTO-CONFIGURATION: Comment this section if using manual override above
    # ============================================================================
    # _auto_config = auto_configure_batch_size()
    # BATCH_SIZE = _auto_config['batch_size']
    # LEARNING_RATE = _auto_config['learning_rate']
    # GRADIENT_CHECKPOINTING = _auto_config['gradient_checkpointing']
    # _manual_override = False
    
    NUM_EPOCHS = 12
    WEIGHT_DECAY = 1e-5
    LR_DECAY_EPOCHS = [6, 10]  # Decay by factor of 10 at epochs 6 and 10
    LR_DECAY_FACTOR = 0.1
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Checkpoint and logging
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'
    SAVE_FREQ = 1  # Save every N epochs
    
    # Multi-GPU distributed training
    DIST_BACKEND = 'nccl'  # 'nccl' for GPU, 'gloo' for CPU
    WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # Data loading
    NUM_WORKERS = 4
    PIN_MEMORY = True


# Create a default config instance
config = Config()

# Print configuration once when module is loaded (only for master process and not in worker processes)
if int(os.environ.get('RANK', 0)) == 0:
    # Check if we're in a dataloader worker (workers have WORKER_ID set by torch)
    import sys
    if 'torch.utils.data._utils.worker' not in sys.modules:
        if config._manual_override:
            print(f"\n{'='*60}")
            print(f"MANUAL training configuration:")
            print(f"  Batch size: {config.BATCH_SIZE} (per-GPU)")
            print(f"  Learning rate: {config.LEARNING_RATE:.2e}")
            print(f"  Gradient checkpointing: {config.GRADIENT_CHECKPOINTING}")
            print(f"  Total task tokens: {config.TOTAL_TASK_TOKENS}")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"Auto-configured training settings:")
            print(f"  Reason: {config._auto_config['reason']}")
            print(f"  Batch size: {config.BATCH_SIZE} (per-GPU)")
            print(f"  Learning rate: {config.LEARNING_RATE:.2e}")
            print(f"  Gradient checkpointing: {config.GRADIENT_CHECKPOINTING}")
            print(f"  Total task tokens: {config.TOTAL_TASK_TOKENS}")
            print(f"{'='*60}\n")

"""
Test script to verify FaceXFormer-main training setup.
Tests model creation, data loading, and forward/backward passes.
"""

import torch
import sys

print("="*60)
print("FaceXFormer-main Training Setup Test")
print("="*60)

# Test 1: Import config
print("\n[1/6] Testing config import...")
try:
    from config import config
    print(f"✓ Config loaded")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Learning rate: {config.LEARNING_RATE:.2e}")
    print(f"  - Total task tokens: {config.TOTAL_TASK_TOKENS}")
except Exception as e:
    print(f"❌ Config import failed: {e}")
    sys.exit(1)

# Test 2: Import model
print("\n[2/6] Testing model import...")
try:
    from network.models.facexformer import FaceXFormer
    print(f"✓ Model imported")
except Exception as e:
    print(f"❌ Model import failed: {e}")
    sys.exit(1)

# Test 3: Create model
print("\n[3/6] Testing model creation...")
try:
    model = FaceXFormer()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Expected: ~92M")
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    sys.exit(1)

# Test 4: Import loss
print("\n[4/6] Testing loss function import...")
try:
    from losses import MultiTaskLoss
    criterion = MultiTaskLoss(config.LOSS_WEIGHTS)
    print(f"✓ Loss function created")
    print(f"  - Loss weights: {list(config.LOSS_WEIGHTS.keys())}")
except Exception as e:
    print(f"❌ Loss import failed: {e}")
    sys.exit(1)

# Test 5: Test forward pass
print("\n[5/6] Testing forward pass...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  - Using device: {device}")
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Create dummy input
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Create dummy labels
    labels = {
        'segmentation': torch.randint(0, 11, (batch_size, 224, 224)).to(device),
        'landmark': torch.randn(batch_size, 136).to(device),
        'headpose': torch.randn(batch_size, 3).to(device),
        'attribute': torch.randn(batch_size, 40).to(device),
        'age': torch.randint(0, 8, (batch_size,)).to(device),
        'gender': torch.randint(0, 2, (batch_size,)).to(device),
        'race': torch.randint(0, 5, (batch_size,)).to(device),
        'visibility': torch.randn(batch_size, 29).to(device),  # 29 landmarks visibility
    }
    
    # Task IDs: 0=seg, 1=landmarks, 2=pose, 3=attr, 4=age, 5=gender, 6=race, 7=visibility
    task_ids = torch.tensor([0, 1]).to(device)
    
    # Forward pass
    with torch.no_grad():
        landmark_out, headpose_out, attribute_out, visibility_out, \
        age_out, gender_out, race_out, seg_out = model(images, labels, task_ids)
    
    print(f"✓ Forward pass successful")
    print(f"  - Segmentation output: {seg_out.shape}")
    print(f"  - Landmark output: {landmark_out.shape}")
    print(f"  - Headpose output: {headpose_out.shape}")
    print(f"  - Attribute output: {attribute_out.shape}")
    print(f"  - Age output: {age_out.shape}")
    print(f"  - Gender output: {gender_out.shape}")
    print(f"  - Race output: {race_out.shape}")
    print(f"  - Visibility output: {visibility_out.shape}")
    
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test backward pass
print("\n[6/6] Testing backward pass (loss + optimization)...")
try:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Forward pass
    landmark_out, headpose_out, attribute_out, visibility_out, \
    age_out, gender_out, race_out, seg_out = model(images, labels, task_ids)
    
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
        labels,
        task_ids,
        compute_individual=True
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✓ Backward pass successful")
    print(f"  - Total loss: {loss.item():.4f}")
    print(f"  - Individual losses:")
    for task_name, task_loss in individual_losses.items():
        print(f"    - {task_name}: {task_loss.item():.4f}")
    
except Exception as e:
    print(f"❌ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("✓ All tests passed!")
print("="*60)
print("\nSetup is ready for training.")
print("\nNext steps:")
print("  1. Download and extract datasets to ./datasets/")
print("  2. Run: python train_simple.py (single GPU)")
print("  3. Or:  launch_train.bat 4 (multi-GPU, Windows)")
print("  4. Or:  bash launch_train.sh (multi-GPU, Linux)")
print("\nSee TRAINING_README.md for detailed instructions.")
print("="*60 + "\n")
